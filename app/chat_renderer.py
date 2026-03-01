"""
app/chat_renderer.py
-----------------------------------------------------------------------------
Unified synchronous HTTP client for the Ollama API.

Provides two interfaces:
  - :meth:`ChatRenderer.render` — fire-and-forget chat call that swallows
    errors and returns ``None`` on any failure (used by the Chat Translation
    page, matching mud_server behaviour).
  - :meth:`ChatRenderer.generate` — same transport but raises on failure
    (used by the Character Description page, where the route handler maps
    each exception to an HTTPException).
  - :meth:`ChatRenderer.list_models` — static helper that queries
    ``/api/tags`` and returns a sorted list of pulled model names.

Both generation methods use Ollama's ``/api/chat`` endpoint with the
OpenAI-compatible messages array (system + user roles), which is what the
production MUD translation layer also uses.  This ensures that any
model-behaviour differences between ``/api/generate`` (flat prompt) and
``/api/chat`` (messages) are visible during lab testing.

Connection pooling
------------------
HTTP connections are reused across requests via a module-level client pool
keyed by ``(host, connect_timeout, read_timeout)``.  This avoids the
overhead of a fresh TCP handshake + TLS negotiation on every call and
prevents cold-start failures when Character B fires immediately after
Character A.  Call :func:`close_all_clients` at shutdown to release
pooled connections cleanly.

Sync rationale
--------------
The lab's route handlers are synchronous (FastAPI runs them in a
thread-pool executor), so a blocking httpx call here does not stall the
async event loop.  Using an async client would require ``asyncio.run()`` or
restructuring the handler, neither of which is worth the complexity for a
single-user tool.

Request structure sent to Ollama
---------------------------------
.. code-block:: json

    {
      "model": "<model-tag>",
      "stream": false,
      "keep_alive": "5m",
      "messages": [
        {"role": "system", "content": "<rendered system prompt>"},
        {"role": "user",   "content": "<ooc message>"}
      ],
      "options": {
        "temperature": <float>,
        "num_predict": <int>,
        "seed": <int>  // only when seed is not None
      }
    }

The ``stream: false`` flag is required to get a single JSON response body
rather than a series of newline-delimited chunks.

The ``keep_alive`` field tells Ollama how long to keep the model loaded in
memory after responding (default ``"5m"``).  This prevents cold-start
latency on back-to-back requests (e.g. Character A then Character B).

Environment variables
---------------------
OLLAMA_HOST – Base URL of the Ollama server (default: http://localhost:11434).
              Read once at import time so the value is consistent for the
              lifetime of the process.
"""

from __future__ import annotations

import logging
import os
import threading

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Strip any trailing slash so we can safely append paths.
# Exported so main.py can pass the default to the template.
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# ── Shared client pool ────────────────────────────────────────────────────────
# Keyed by (host, connect_timeout, read_timeout) so that requests to the same
# Ollama instance reuse TCP connections (HTTP Keep-Alive).  Thread-safe via a
# simple lock — contention is negligible for a single-user tool.

_client_pool: dict[tuple[str, float, float], httpx.Client] = {}
_pool_lock = threading.Lock()


def _get_client(host: str, timeout: httpx.Timeout) -> httpx.Client:
    """Return a shared ``httpx.Client`` for the given host and timeout.

    Creates a new client on first access; subsequent calls with the same
    parameters return the cached instance.
    """
    key = (host, timeout.connect or 10.0, timeout.read or 120.0)
    with _pool_lock:
        client = _client_pool.get(key)
        if client is None:
            client = httpx.Client(timeout=timeout)
            _client_pool[key] = client
        return client


def close_all_clients() -> None:
    """Close all pooled HTTP clients and clear the pool.

    Call this during application shutdown to release TCP connections cleanly.
    """
    with _pool_lock:
        for client in _client_pool.values():
            try:
                client.close()
            except Exception:  # nosec B110 — best-effort shutdown cleanup
                pass
        _client_pool.clear()


class ChatRenderer:
    """Synchronous Ollama client that calls the ``/api/chat`` endpoint.

    Requests reuse a shared ``httpx.Client`` from a module-level pool keyed
    by ``(host, connect_timeout, read_timeout)``.  This enables HTTP
    Keep-Alive across calls and avoids cold-start latency when multiple
    requests target the same Ollama instance in quick succession.

    Args:
        host:            Ollama server base URL, e.g.
                         ``'http://localhost:11434'``.  A trailing slash is
                         stripped automatically.  ``/api/chat`` is appended
                         internally.
        model:           Ollama model tag, e.g. ``'gemma2:2b'``.  Must match
                         a model that has been pulled in Ollama.
        timeout_seconds: HTTP *read* timeout in seconds.  Applies to waiting
                         for the model to finish generating.  Defaults to
                         120 s to accommodate slow hardware or large models.
                         The *connect* timeout is always 10 s.
        temperature:     Sampling temperature forwarded to Ollama's
                         ``options.temperature``.  0.0 is deterministic
                         (greedy decoding); higher values add randomness.
        seed:            Optional integer forwarded to Ollama's
                         ``options.seed``.  When provided, Ollama uses this
                         as the random seed for token sampling, which makes
                         the output reproducible for the same input.
                         When ``None``, the ``seed`` key is omitted from the
                         options object and Ollama chooses its own seed.
        max_tokens:      ``num_predict`` ceiling for the generation.  Ollama
                         stops after this many tokens even if the model would
                         continue.
        keep_alive:      Duration string telling Ollama how long to keep the
                         model loaded in memory after responding (e.g.
                         ``"5m"``, ``"1h"``, ``"0"`` to unload immediately).
                         Defaults to ``"5m"`` to prevent cold-start latency
                         on back-to-back requests.
    """

    def __init__(
        self,
        *,
        host: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 128,
        keep_alive: str = "5m",
    ) -> None:
        self._host = host.rstrip("/")
        self._endpoint = f"{self._host}/api/chat"
        self._model = model
        # httpx.Timeout(default, connect=...) sets read/write/pool to `default`
        # while overriding just the connect timeout.
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        self._keep_alive = keep_alive

    def _build_payload(self, system_prompt: str, user_message: str) -> dict:
        options: dict = {
            "temperature": self._temperature,
            "num_predict": self._max_tokens,
        }
        # Only include seed in options when explicitly provided — Ollama's
        # default behaviour (random seed) is preserved when seed is None.
        if self._seed is not None:
            options["seed"] = self._seed

        return {
            "model": self._model,
            "stream": False,  # single JSON response, not a stream of chunks
            "keep_alive": self._keep_alive,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": options,
        }

    def render(self, system_prompt: str, user_message: str) -> str | None:
        """POST to Ollama /api/chat and return the raw response content.

        Builds the request payload, sends it to ``self._endpoint``, and
        extracts the model's response from ``data["message"]["content"]``.

        The ``system_prompt`` and ``user_message`` are sent as separate
        entries in the ``messages`` array using the ``"system"`` and
        ``"user"`` roles respectively.  This matches the format used by the
        production MUD translation layer.

        No content-level validation is performed here; that is handled
        downstream by :class:`~app.output_validator.OutputValidator`.

        Args:
            system_prompt: Fully-rendered system prompt text.  All
                           ``{{placeholder}}`` variables should already have
                           been substituted before this call.
            user_message:  The OOC message (user turn).  Sent verbatim as
                           the ``"user"`` role message.

        Returns:
            The stripped ``message.content`` string on success, or ``None``
            on any of the following failure conditions:

            - **TimeoutException**: Ollama took longer than
              ``timeout_seconds`` to respond.
            - **ConnectError**: Ollama is not reachable at the configured
              endpoint (wrong host, not running, firewall).
            - **Any other exception**: Unexpected HTTP or JSON parsing error.

            All failure paths log a warning/error via the module logger.
            ``None`` return causes the endpoint to report
            ``"fallback.api_error"`` in the translation result.
        """
        payload = self._build_payload(system_prompt, user_message)
        client = _get_client(self._host, self._timeout)

        try:
            response = client.post(self._endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            # Ollama /api/chat response shape:
            # {"model": ..., "message": {"role": "assistant", "content": "..."}, ...}
            content = data.get("message", {}).get("content", "").strip()
            return content or None

        except httpx.TimeoutException:
            logger.warning(
                "ChatRenderer: request timed out (endpoint=%s, read_timeout=%.0fs)",
                self._endpoint,
                self._timeout.read,
            )
            return None
        except httpx.ConnectError:
            logger.warning("ChatRenderer: cannot connect to Ollama at %s", self._endpoint)
            return None
        except Exception as exc:
            logger.error("ChatRenderer: request failed: %s", exc)
            return None

    def generate(self, system_prompt: str, user_message: str) -> tuple[str, dict]:
        """POST to /api/chat; return (text, usage). Raises on any failure.

        Same payload structure as :meth:`render`, but exceptions propagate to
        the caller instead of being caught.  This matches the contract expected
        by the ``/api/generate`` route handler, which maps each exception type
        to an HTTPException.

        Args:
            system_prompt: Fully-rendered system prompt text.
            user_message:  The user turn (axis JSON string for description
                           generation, or OOC message for chat).

        Returns:
            tuple[str, dict]:
                - ``str`` — Stripped ``message.content``.
                - ``dict`` — ``{"prompt_eval_count": int|None, "eval_count": int|None}``

        Raises:
            httpx.HTTPStatusError:  Non-2xx response from Ollama.
            httpx.TimeoutException: Request timed out.
            ValueError:             Response is missing the ``"message"`` key.
        """
        payload = self._build_payload(system_prompt, user_message)
        client = _get_client(self._host, self._timeout)

        response = client.post(self._endpoint, json=payload)
        response.raise_for_status()
        data = response.json()

        if "message" not in data:
            raise ValueError(
                f"Ollama /api/chat response for model '{self._model}' is missing "
                f"the 'message' key. Got keys: {list(data.keys())}"
            )

        text = data["message"].get("content", "").strip()
        usage = {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        }
        return text, usage

    @staticmethod
    def list_models(host: str | None = None) -> list[str]:
        """Sorted model names from /api/tags. Returns [] on any error.

        Args:
            host: Optional Ollama server base URL.  When ``None``, the
                  module-level :data:`OLLAMA_HOST` constant is used.

        Returns:
            Sorted list of model name strings, e.g. ``["gemma2:2b", "llama3.2:1b"]``.
            Returns an empty list if Ollama is unreachable or returns an error,
            allowing the frontend to degrade gracefully.
        """
        base = host.rstrip("/") if host else OLLAMA_HOST
        url = f"{base}/api/tags"
        timeout = httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url)
                response.raise_for_status()
            data = response.json()
            names = [m["name"] for m in data.get("models", []) if "name" in m]
            return sorted(names)
        except Exception as exc:
            logger.warning("Failed to list Ollama models: %s: %s", type(exc).__name__, exc)
            return []

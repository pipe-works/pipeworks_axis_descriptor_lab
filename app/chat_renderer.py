"""
app/chat_renderer.py
-----------------------------------------------------------------------------
Sync HTTP wrapper for Ollama's /api/chat endpoint.

Mirrors mud_server's translation/renderer.py, using httpx (already a lab
dependency) instead of requests.  The /api/chat endpoint uses a messages
array (system + user roles) rather than the flat prompt string used by
/api/generate, which is what the production translation layer sends.

Sync rationale
--------------
The lab's route handlers are synchronous (FastAPI runs them in a
thread-pool executor), so a blocking httpx call here does not stall the
async event loop.  Using an async client would require ``asyncio.run()`` or
restructuring the handler, neither of which is worth the complexity for a
single-user tool.

Endpoint difference: /api/chat vs /api/generate
-------------------------------------------------
``/api/generate`` accepts a flat ``prompt`` string and is used by the
Character Description page (via ``ollama_client.py``).

``/api/chat`` accepts a ``messages`` array in OpenAI-compatible format and
is what the production MUD translation layer uses.  Using the same endpoint
here ensures that any model behaviour differences between the two APIs are
visible during lab testing.

Request structure sent to Ollama
---------------------------------
.. code-block:: json

    {
      "model": "<model-tag>",
      "stream": false,
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
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class ChatRenderer:
    """Synchronous renderer that calls Ollama's /api/chat endpoint.

    Each call to :meth:`render` opens a short-lived ``httpx.Client``
    context, POSTs the request, and returns the model's response text.
    Connection and read timeouts are configured separately so that a slow
    Ollama instance (long generation) does not fail with a connect timeout.

    Args:
        api_endpoint:    Full ``/api/chat`` URL, e.g.
                         ``'http://localhost:11434/api/chat'``.  Must include
                         the path; the caller is responsible for constructing
                         it from the configured Ollama host.
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
    """

    def __init__(
        self,
        *,
        api_endpoint: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 128,
    ) -> None:
        self._api_endpoint = api_endpoint
        self._model = model
        # httpx.Timeout(default, connect=...) sets read/write/pool to `default`
        # while overriding just the connect timeout.
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens

    def render(self, system_prompt: str, user_message: str) -> str | None:
        """POST to Ollama /api/chat and return the raw response content.

        Builds the request payload, sends it to ``self._api_endpoint``, and
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
        options: dict = {
            "temperature": self._temperature,
            "num_predict": self._max_tokens,
        }
        # Only include seed in options when explicitly provided — Ollama's
        # default behaviour (random seed) is preserved when seed is None.
        if self._seed is not None:
            options["seed"] = self._seed

        payload = {
            "model": self._model,
            "stream": False,  # single JSON response, not a stream of chunks
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": options,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(self._api_endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                # Ollama /api/chat response shape:
                # {"model": ..., "message": {"role": "assistant", "content": "..."}, ...}
                content = data.get("message", {}).get("content", "").strip()
                return content or None

        except httpx.TimeoutException:
            logger.warning(
                "ChatRenderer: request timed out (endpoint=%s, read_timeout=%.0fs)",
                self._api_endpoint,
                self._timeout.read,
            )
            return None
        except httpx.ConnectError:
            logger.warning("ChatRenderer: cannot connect to Ollama at %s", self._api_endpoint)
            return None
        except Exception as exc:
            logger.error("ChatRenderer: request failed: %s", exc)
            return None

"""
app/ollama_client.py
-----------------------------------------------------------------------------
Thin synchronous wrapper around the Ollama HTTP API.

Why synchronous?
----------------
FastAPI can run sync route handlers in a thread-pool executor automatically
(via `def` rather than `async def`), which avoids blocking the event loop
while we wait for the LLM to respond.  For a single-user local lab this is
perfectly adequate and keeps the code simple.

Ollama /api/generate reference
-------------------------------
POST {host}/api/generate
{
    "model":   "<model-name>",
    "prompt":  "<user turn – the axis JSON string>",
    "system":  "<system prompt text>",
    "options": {"temperature": <float>, "num_predict": <int>, "seed": <int>},
    "stream":  false
}

When ``seed`` is provided in the ``options`` object, Ollama pins the random
number generator used during token sampling.  Combined with a fixed
temperature and identical prompt, this makes generation **deterministic** (the
same seed + inputs will produce the same output).  The seed value originates
from the AxisPayload and is part of the Interpretive Provenance Chain (IPC)
hash — so it was always *recorded* for auditability, but it must also be
*forwarded* to the model for it to actually control sampling.

Response (non-streaming):
{
    "model": "...",
    "response": "<generated text>",
    "prompt_eval_count": <int>,
    "eval_count": <int>,
    ...
}

Environment variables
---------------------
OLLAMA_HOST – Base URL of the Ollama server (default: http://localhost:11434).
              Read once at import time so the value is consistent for the
              lifetime of the process.

Logging
-------
Errors in non-critical paths (e.g. model listing) are logged as warnings via
the standard ``logging`` module rather than silently swallowed.  This ensures
that network issues, timeouts, and malformed responses are visible in server
logs without crashing the application.
"""

from __future__ import annotations

import logging
import os

import httpx
from dotenv import load_dotenv

# Load .env if present (no-op if the file doesn't exist)
load_dotenv()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Strip any trailing slash so we can safely append paths.
_OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# How long (seconds) to wait for Ollama to start streaming back a response.
# Small local models should start within a few seconds; larger ones may take
# longer on first call while the model is loaded into VRAM.
_CONNECT_TIMEOUT: float = 10.0

# How long to wait for the *entire* response body.
# 120 s is generous but local inference can be slow for bigger models.
_READ_TIMEOUT: float = 120.0


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def ollama_generate(
    *,
    model: str,
    system_prompt: str,
    user_json_str: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> tuple[str, dict]:
    """
    Call the Ollama /api/generate endpoint and return the generated text.

    The ``seed`` parameter is forwarded to Ollama's ``options.seed`` field,
    which pins the random number generator used during token sampling.  When
    combined with a fixed temperature and identical prompt, this makes the
    generation **deterministic** — the same inputs will produce the same
    output.  The seed originates from the AxisPayload and is already part of
    the Interpretive Provenance Chain (IPC) hash; passing it here ensures the
    model actually *uses* it rather than merely recording it for auditing.

    Parameters
    ----------
    model         : Ollama model identifier, e.g. "gemma2:2b".
    system_prompt : The system-role text that constrains LLM behaviour.
    user_json_str : The axis payload serialised as a pretty-printed JSON string.
                    This becomes the user turn in the conversation.
    temperature   : Sampling temperature; lower = more deterministic.
    max_tokens    : Maximum tokens the model may generate (maps to
                    Ollama's ``num_predict`` option).
    seed          : RNG seed from the AxisPayload.  Forwarded to Ollama's
                    ``options.seed`` to make token sampling deterministic.
                    This is the same seed used in the IPC hash.

    Returns
    -------
    A tuple of:
      - str  : The raw text returned by the model (stripped of leading /
               trailing whitespace).
      - dict : Token-usage information extracted from the Ollama response
               (keys: "prompt_eval_count", "eval_count").  Values may be None
               if the model doesn't report them.

    Raises
    ------
    httpx.HTTPStatusError  : If Ollama returns a non-2xx response.
    httpx.TimeoutException : If the request times out.
    ValueError             : If the Ollama response is missing the "response"
                             key (malformed response guard).
    """
    url = f"{_OLLAMA_HOST}/api/generate"

    # Build the request body.  We always disable streaming so we get a single
    # JSON object back rather than a newline-delimited stream.
    #
    # The "seed" in "options" pins Ollama's RNG for token sampling.  Together
    # with a fixed temperature and identical prompt, this makes the model's
    # output deterministic.  Without it, repeated calls with the same IPC
    # inputs would still produce different text due to random sampling.
    body: dict = {
        "model": model,
        "prompt": user_json_str,
        "system": system_prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed,
        },
        "stream": False,
    }

    timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=5.0, pool=5.0)

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=body)
        # Raise immediately for HTTP errors (4xx / 5xx) so the caller gets a
        # clear exception rather than a confusing KeyError later.
        response.raise_for_status()

    data: dict = response.json()

    # Guard: the "response" key must be present.
    if "response" not in data:
        raise ValueError(
            f"Ollama response for model '{model}' is missing the 'response' key. "
            f"Got keys: {list(data.keys())}"
        )

    generated_text: str = data["response"].strip()

    # Collect token-usage fields; these are informational and optional.
    usage: dict = {
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
    }

    return generated_text, usage


def list_local_models() -> list[str]:
    """
    Return a sorted list of model names that are currently available in the
    local Ollama instance.

    Calls GET {OLLAMA_HOST}/api/tags.  If Ollama is not reachable an empty
    list is returned so the frontend can degrade gracefully.

    Returns
    -------
    list[str] : Sorted model name strings (e.g. ["gemma2:2b", "llama3.2:1b"]).
    """
    url = f"{_OLLAMA_HOST}/api/tags"
    timeout = httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
        data = response.json()
        # Ollama returns {"models": [{"name": "...", ...}, ...]}
        names = [m["name"] for m in data.get("models", []) if "name" in m]
        return sorted(names)
    except Exception as exc:
        # Graceful degradation: return an empty list so the frontend can
        # still render (with no model dropdown options) even when Ollama is
        # unreachable, timing out, or returning unexpected responses.
        # Log the error so operators can diagnose connectivity issues from
        # server logs without having to reproduce the failure interactively.
        logger.warning("Failed to list Ollama models: %s: %s", type(exc).__name__, exc)
        return []

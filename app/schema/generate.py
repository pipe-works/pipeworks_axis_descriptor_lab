"""
Schemas for descriptive generation and structured run logging.

These models cover the character-description page's core generation request,
generation response, and append-only JSONL logging format.  They depend on
the shared axis payload primitives but otherwise contain no business logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schema.axis import AxisPayload


class GenerateRequest(BaseModel):
    """
    Full request body for POST /api/generate.

    The frontend serialises its current in-memory state into this object and
    sends it to the backend, which forwards the payload to Ollama.

    Optional fields allow the frontend to override per-request settings
    (model, temperature, token budget, custom system prompt) without
    restarting the server.
    """

    payload: AxisPayload = Field(
        ...,
        description="The axis payload that will be serialised to JSON and sent as the LLM user turn.",
    )
    model: str = Field(
        ...,
        description="Ollama model name to use (e.g. 'gemma2:2b').",
        examples=["gemma2:2b", "llama3.2:1b"],
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature.  Low values (0.1–0.3) reduce variance "
            "and are preferred for drift comparison experiments."
        ),
    )
    max_tokens: int = Field(
        default=120,
        ge=10,
        le=2048,
        description="Maximum number of tokens the model may produce.",
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Optional override for the system prompt.  When None the server "
            "loads app/prompts/system_prompt_v01.txt."
        ),
    )
    ollama_host: str | None = Field(
        default=None,
        description=(
            "Optional Ollama server URL override.  When None the server uses "
            "the OLLAMA_HOST environment variable (default: http://localhost:11434)."
        ),
        examples=["http://localhost:11434", "http://192.168.1.50:11434"],
    )


class GenerateResponse(BaseModel):
    """
    Response body for POST /api/generate.

    Carries the raw LLM output plus enough metadata to reconstruct the exact
    call for logging, diffing, and repeatability analysis.

    The four hash fields (``input_hash``, ``system_prompt_hash``,
    ``output_hash``, ``ipc_id``) form the **Interpretive Provenance Chain
    (IPC)** — a complete fingerprint of every variable that influenced the
    generation.  Together they enable drift detection and reproducibility
    audits.
    """

    text: str = Field(
        ...,
        description="The raw descriptive paragraph produced by the LLM.",
    )
    model: str = Field(
        ...,
        description="Ollama model name that was used.",
    )
    temperature: float = Field(
        ...,
        description="Sampling temperature that was used.",
    )
    usage: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Token-usage information returned by Ollama, if available.  "
            "Keys vary by model (e.g. 'prompt_eval_count', 'eval_count')."
        ),
    )

    input_hash: str | None = Field(
        default=None,
        description="SHA-256 hex digest of the canonical serialised AxisPayload.",
    )
    system_prompt_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the normalised system prompt used.  "
            "Normalisation strips per-line whitespace and edge blank lines."
        ),
    )
    output_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the normalised output text.  "
            "Normalisation collapses extra spaces and strips edges."
        ),
    )
    ipc_id: str | None = Field(
        default=None,
        description=(
            "Interpretive Provenance Chain identifier — a SHA-256 digest of "
            "input_hash:system_prompt_hash:model:temperature:max_tokens:seed.  "
            "Two generations with the same IPC ID used identical inputs."
        ),
    )


class LogEntry(BaseModel):
    """
    Schema for a single structured log record written by POST /api/log.

    Captures everything needed to reproduce a run and detect drift across
    repeated calls with the same seed / policy hash. `input_hash` groups runs
    that should be identical, `payload` stores the full deterministic input,
    and the remaining fields capture the generated output plus the runtime
    parameters and timestamp needed for later audit or drift analysis.
    """

    input_hash: str = Field(
        ...,
        description="SHA-256 hex digest of the canonical serialised AxisPayload.",
    )
    payload: AxisPayload = Field(
        ...,
        description="Full axis payload for this run.",
    )
    output: str = Field(
        ...,
        description="LLM-generated descriptive text.",
    )
    model: str = Field(..., description="Ollama model used.")
    temperature: float = Field(..., description="Sampling temperature used.")
    max_tokens: int = Field(..., description="Token budget used.")
    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of when the log entry was created.",
    )
    system_prompt_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the normalised system prompt.  "
            "None when the log call did not include the prompt text."
        ),
    )
    output_hash: str | None = Field(
        default=None,
        description="SHA-256 hex digest of the normalised output text.",
    )
    ipc_id: str | None = Field(
        default=None,
        description=(
            "Interpretive Provenance Chain identifier.  None when "
            "system_prompt was not provided to the log endpoint."
        ),
    )

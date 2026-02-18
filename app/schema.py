"""
app/schema.py
-----------------------------------------------------------------------------
Pydantic v2 models for every request / response object in the Axis Descriptor
Lab API.

Design principles
-----------------
• Keep models thin – no business logic here.
• Every field has a docstring-style `description` so FastAPI's auto-generated
  OpenAPI UI is immediately useful.
• Scores are validated to the [0, 1] closed interval; the system is
  authoritative so we want hard boundaries rather than silent clamps.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# -----------------------------------------------------------------------------
# Axis primitives
# -----------------------------------------------------------------------------


class AxisValue(BaseModel):
    """
    A single named axis entry consisting of a human-readable label and a
    normalised score in [0, 1].

    The *label* is the interpretive colour (e.g. "resentful", "weary").
    The *score* is the underlying deterministic value produced by the engine.
    Both are forwarded verbatim to the LLM; the LLM must not treat them as
    facts – they are tonal hints only.
    """

    label: str = Field(
        ...,
        description="Short human-readable descriptor for this axis position.",
        examples=["resentful", "weary", "questioned"],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised axis score in the closed interval [0, 1].",
        examples=[0.5, 0.62],
    )

    @field_validator("label")
    @classmethod
    def label_not_empty(cls, v: str) -> str:
        """Ensure the label is not a blank string."""
        v = v.strip()
        if not v:
            raise ValueError("label must not be empty or whitespace")
        return v


class AxisPayload(BaseModel):
    """
    The complete deterministic state object that drives a single descriptive
    generation.  This is the "source of truth" handed to the LLM as a JSON
    string in the user turn.

    Fields
    ------
    axes        – keyed by axis name (e.g. "demeanor"), value is AxisValue.
    policy_hash – SHA-256 hex digest of the policy rules in force when this
                  payload was produced.  Included for auditability; the LLM is
                  instructed never to mention it.
    seed        – deterministic RNG seed used to produce the axis scores.
    world_id    – identifier for the Pipe-Works world context.
    """

    axes: dict[str, AxisValue] = Field(
        ...,
        description="Map of axis name → AxisValue.  At least one entry required.",
    )
    policy_hash: str = Field(
        ...,
        description=(
            "SHA-256 hex digest of the axis policy in force.  "
            "Forwarded to the LLM but the system prompt instructs the model "
            "never to mention or interpret it."
        ),
    )
    seed: int = Field(
        ...,
        description="RNG seed that produced the axis scores.  Used for reproducibility.",
    )
    world_id: str = Field(
        ...,
        description="Identifier for the Pipe-Works world (e.g. 'pipeworks_web').",
    )

    @field_validator("axes")
    @classmethod
    def axes_not_empty(cls, v: dict[str, AxisValue]) -> dict[str, AxisValue]:
        """Require at least one axis."""
        if not v:
            raise ValueError("axes must contain at least one entry")
        return v


# -----------------------------------------------------------------------------
# /api/generate  request / response
# -----------------------------------------------------------------------------


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

    # -- Interpretive Provenance Chain (IPC) fields -----------------------
    # These four hashes uniquely identify the full generation context.

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


# -----------------------------------------------------------------------------
# /api/log  request
# -----------------------------------------------------------------------------


class LogEntry(BaseModel):
    """
    Schema for a single structured log record written by POST /api/log.

    Captures everything needed to reproduce a run and detect drift across
    repeated calls with the same seed / policy_hash.

    Fields
    ------
    input_hash   – SHA-256 of the canonical JSON-serialised AxisPayload.
                   Use this to group runs that should be identical.
    payload      – Full axis payload (stored for later inspection).
    output       – The descriptive text the LLM produced.
    model        – Model identifier.
    temperature  – Sampling temperature used.
    max_tokens   – Token budget used.
    timestamp    – ISO-8601 UTC timestamp (set by the server, not the client).
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

    # -- Interpretive Provenance Chain (IPC) fields -----------------------
    # Optional so that existing JSONL records (written before this feature)
    # can still be deserialised without error.

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


# -----------------------------------------------------------------------------
# /api/save  request / response
# -----------------------------------------------------------------------------


class SaveRequest(BaseModel):
    """
    Full request body for POST /api/save.

    The frontend collects all in-memory state at the moment the user clicks
    Save and sends it here.  The backend writes individual files to a
    timestamped subfolder under ``data/``.

    Fields
    ------
    payload       – The current AxisPayload (source of truth axes).
    output        – The latest generated text (``state.current``), or None if
                    the user hasn't generated yet.
    baseline      – The stored baseline text (``state.baseline``), or None.
    model         – The Ollama model name used for the last generation.
    temperature   – Sampling temperature used.
    max_tokens    – Token budget used.
    system_prompt – The system prompt actually sent to the LLM.  The frontend
                    must resolve which prompt was in use (custom override or
                    the server default fetched via GET /api/system-prompt) and
                    send it verbatim.  Must not be empty.
    """

    payload: AxisPayload = Field(
        ...,
        description="The full axis payload at save time.",
    )
    output: str | None = Field(
        default=None,
        description="Latest generated text (state.current).  None if not generated.",
    )
    baseline: str | None = Field(
        default=None,
        description="Stored baseline text (state.baseline).  None if not set.",
    )
    model: str = Field(
        ...,
        description="Ollama model name used (e.g. 'gemma2:2b').",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature used.",
    )
    max_tokens: int = Field(
        default=120,
        ge=10,
        le=2048,
        description="Maximum token budget used.",
    )
    system_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "The system prompt actually used in the last generation.  "
            "The frontend must resolve the active prompt (custom override or "
            "server default) and send it verbatim.  Must not be empty."
        ),
    )


class SaveResponse(BaseModel):
    """
    Response body for POST /api/save.

    Returns the save folder name and the list of files written so the
    frontend can display a confirmation message in the status bar.

    Fields
    ------
    folder_name        – The name of the subfolder created inside ``data/``.
    files              – Sorted list of filenames written inside the subfolder.
    input_hash         – SHA-256 of the payload (for traceability).
    timestamp          – ISO-8601 UTC timestamp of when the save occurred.
    system_prompt_hash – SHA-256 of the normalised system prompt.
    output_hash        – SHA-256 of the normalised output (None if no output).
    ipc_id             – IPC identifier (None if no output was generated).
    """

    folder_name: str = Field(
        ...,
        description="Subfolder name under data/ (e.g. '20260218_143022_abc1def2').",
    )
    files: list[str] = Field(
        ...,
        description="Sorted list of filenames written inside the folder.",
    )
    input_hash: str = Field(
        ...,
        description="SHA-256 hex digest of the saved AxisPayload.",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of the save operation.",
    )

    # -- Interpretive Provenance Chain (IPC) fields -----------------------

    system_prompt_hash: str | None = Field(
        default=None,
        description="SHA-256 hex digest of the normalised system prompt.",
    )
    output_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the normalised output text.  "
            "None when no output was generated before saving."
        ),
    )
    ipc_id: str | None = Field(
        default=None,
        description=(
            "Interpretive Provenance Chain identifier.  "
            "None when no output was generated (incomplete chain)."
        ),
    )


# -----------------------------------------------------------------------------
# POST /api/analyze-delta  request / response
# -----------------------------------------------------------------------------


class DeltaRequest(BaseModel):
    """
    Request body for ``POST /api/analyze-delta``.

    Accepts two plain-text strings (baseline and current) for signal
    isolation analysis.  The backend runs both through the NLP pipeline
    (tokenise → lemmatise → filter stopwords) and returns the set
    difference as sorted word lists.
    """

    baseline_text: str = Field(
        ...,
        min_length=1,
        description=(
            "The reference text (A) — typically the stored baseline output.  " "Must not be empty."
        ),
        examples=["The weathered figure stands near the threshold."],
    )
    current_text: str = Field(
        ...,
        min_length=1,
        description=(
            "The comparison text (B) — typically the latest generated output.  "
            "Must not be empty."
        ),
        examples=["A dark goblin lurks beyond the crumbling gate."],
    )


class DeltaResponse(BaseModel):
    """
    Response body for ``POST /api/analyze-delta``.

    Contains two alphabetically sorted lists of content lemmas that
    represent meaningful lexical differences between the baseline and
    current texts, after stopword removal and lemmatisation.

    The lists are **set differences**, not positional diffs:

    - ``removed`` = content lemmas in A but absent from B.
    - ``added``   = content lemmas in B but absent from A.
    """

    removed: list[str] = Field(
        ...,
        description=(
            "Content lemmas present in the baseline (A) but absent from the "
            "current text (B).  Alphabetically sorted."
        ),
    )
    added: list[str] = Field(
        ...,
        description=(
            "Content lemmas present in the current text (B) but absent from "
            "the baseline (A).  Alphabetically sorted."
        ),
    )

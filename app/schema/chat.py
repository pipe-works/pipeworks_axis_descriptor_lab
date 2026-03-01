"""
Schemas for chat translation, chat save, and chat import flows.

These models cover the chat page's translation pipeline, live-mode log
entries, save-package persistence, and import payloads used to rebuild the
frontend state from archived chat sessions.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from app.schema.axis import AxisValue


class ChatCharacterInput(BaseModel):
    """
    Axis profile and OOC message for a single character in a chat translation
    request.
    """

    axes: dict[str, AxisValue] = Field(
        ...,
        description="Map of axis name → AxisValue for this character.",
    )
    ooc_message: str = Field(
        ...,
        min_length=1,
        description="Raw out-of-character message text to translate.",
    )
    channel: str = Field(
        default="say",
        description=(
            "Chat delivery mode injected into the profile as {{channel}}.  "
            "One of 'say', 'yell', or 'whisper'."
        ),
    )
    active_axes: list[str] | None = Field(
        default=None,
        description=(
            "Axis names to include in the rendered profile.  "
            "None means all axes are active.  An empty list disables all axes."
        ),
    )


class ChatTranslationRequest(BaseModel):
    """
    Request body for POST /api/translate_chat.

    Translates OOC messages for one or two characters using the
    OOC→IC translation pipeline.  Each character's profile is built
    from its axes, filtered by active_axes, and rendered into the
    system prompt template before calling Ollama /api/chat.
    """

    character_a: ChatCharacterInput | None = Field(
        default=None,
        description=(
            "Profile and OOC message for Character A.  "
            "When None, only Character B is translated."
        ),
    )
    character_b: ChatCharacterInput | None = Field(
        default=None,
        description="Profile and OOC message for Character B.  When None only Character A is translated.",
    )

    @model_validator(mode="after")
    def at_least_one_character(self) -> "ChatTranslationRequest":
        """Require at least one of character_a or character_b."""
        if self.character_a is None and self.character_b is None:
            raise ValueError("At least one of character_a or character_b must be provided.")
        return self

    model: str = Field(
        ...,
        description="Ollama model tag (e.g. 'gemma2:2b').",
        examples=["gemma2:2b", "llama3.2:1b"],
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.  0.7 is the production default.",
    )
    max_tokens: int = Field(
        default=128,
        ge=10,
        le=512,
        description="Maximum tokens to generate (num_predict).",
    )
    seed: int = Field(
        ...,
        description=(
            "Integer seed forwarded to Ollama options.seed.  "
            "Also used as the seed component of the IPC ID."
        ),
    )
    prompt_name: str | None = Field(
        default=None,
        description=(
            "Name of a server-side prompt file to load from app/prompts/.  "
            "When None and system_prompt is also None, the server loads "
            "ic_v01_undertaking.txt as the default."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Inline system prompt override.  Takes precedence over "
            "prompt_name when both are provided."
        ),
    )
    ollama_host: str | None = Field(
        default=None,
        description=(
            "Optional Ollama server URL override.  "
            "When None the server uses OLLAMA_HOST env var."
        ),
        examples=["http://localhost:11434"],
    )
    strict_mode: bool = Field(
        default=True,
        description=(
            "When True, any output constraint violation returns None "
            "(fallback to OOC).  When False, minor violations are cleaned up."
        ),
    )
    max_output_chars: int = Field(
        default=280,
        ge=50,
        le=2000,
        description="Maximum character count for IC output.",
    )
    world_id: str | None = Field(
        default=None,
        description=(
            "World ID for server-mode translation.  Sent by the frontend "
            "so the backend does not depend on prior select-world state."
        ),
    )


class ChatTranslationResult(BaseModel):
    """
    Result for a single character's OOC→IC translation attempt.

    ic_text is None when translation failed (status explains why).
    The IPC fields form the provenance chain for this translation.
    """

    ic_text: str | None = Field(
        ...,
        description="Translated IC dialogue, or None on failure.",
    )
    status: str = Field(
        ...,
        description="One of 'success', 'fallback.api_error', or 'fallback.validation_failed'.",
    )
    input_hash: str | None = Field(
        default=None,
        description="SHA-256 of the active axis state + OOC message + channel.",
    )
    system_prompt_hash: str | None = Field(
        default=None,
        description="SHA-256 of the normalised rendered system prompt.",
    )
    output_hash: str | None = Field(
        default=None,
        description="SHA-256 of the normalised IC output.  None on failure.",
    )
    ipc_id: str | None = Field(
        default=None,
        description="IPC identifier (input:prompt:model:temp:tokens:seed).  None on failure.",
    )
    model: str | None = Field(
        default=None,
        description="Model used for this translation (server-returned or request echo).",
    )
    error_detail: str | None = Field(
        default=None,
        description="Human-readable error detail when translation failed.",
    )


class ChatTranslationResponse(BaseModel):
    """
    Response body for POST /api/translate_chat.

    Contains results for Character A and optionally Character B.
    """

    character_a: ChatTranslationResult | None = Field(
        default=None,
        description="Translation result for Character A, or None if not requested.",
    )
    character_b: ChatTranslationResult | None = Field(
        default=None,
        description="Translation result for Character B, or None if not requested.",
    )


class ChatLogEntry(BaseModel):
    """
    A single entry in the in-game chat log produced during live mode.

    Captured when a per-character Send succeeds.  Stored in chatState.gameLog
    on the frontend and serialised here for server-side persistence.
    """

    ch: str = Field(
        ...,
        description="Character key: 'a' or 'b'.",
    )
    channel: str = Field(
        ...,
        description="Chat channel used: 'say', 'yell', or 'whisper'.",
    )
    ooc_message: str | None = Field(
        default=None,
        description=(
            "Original out-of-character message typed by the player before translation.  "
            "None for entries created before this field was added."
        ),
    )
    ic_text: str | None = Field(
        default=None,
        description=(
            "Translated IC dialogue text produced by the translation pipeline.  "
            "None for failed entries where no IC text was produced."
        ),
    )
    model: str = Field(
        ...,
        description="Ollama model tag used for this entry.",
    )
    ipc_id: str | None = Field(
        default=None,
        description="IPC identifier from the translation result.  None when unavailable.",
    )
    status: str = Field(
        default="success",
        description=(
            "Translation outcome: 'success', 'fallback.validation_failed', "
            "'fallback.api_error', or 'error' (network/timeout)."
        ),
    )
    error_detail: str | None = Field(
        default=None,
        description=(
            "Human-readable error description when status is not 'success'.  "
            "None for successful translations."
        ),
    )
    sent_at: str | None = Field(
        default=None,
        description=(
            "ISO-8601 timestamp of when the OOC message was sent (captured "
            "before the fetch call on the frontend)."
        ),
    )
    duration_ms: int | None = Field(
        default=None,
        description=(
            "Round-trip translation time in milliseconds, measured from "
            "send to response on the frontend."
        ),
    )
    input_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the canonical input dict (active axes + OOC message + channel) "
            "passed to the translation pipeline.  Matches the 'input' row in the "
            "in-browser IPC meta table."
        ),
    )
    system_prompt_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the fully-rendered IC system prompt (after template variable "
            "substitution with the character's profile summary and OOC message).  "
            "Matches the 'prompt' row in the in-browser IPC meta table."
        ),
    )
    output_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the normalised translated IC text.  "
            "Matches the 'output' row in the in-browser IPC meta table."
        ),
    )


class ChatSaveRequest(BaseModel):
    """
    Request body for POST /api/save_chat.

    Saves a complete in-game chat log session to a timestamped folder under
    ``data/``, including optional character payloads and the system prompt.
    """

    entries: list[ChatLogEntry] = Field(
        ...,
        min_length=1,
        description="In-game log entries to save.  At least one entry required.",
    )
    character_a: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Character A axes dict (axis name → AxisValue-shaped object), "
            "or None if Character A was not used."
        ),
    )
    character_b: dict[str, Any] | None = Field(
        default=None,
        description="Character B axes dict, or None if not used.",
    )
    model: str = Field(
        ...,
        description="Ollama model tag used during this session.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature used during the session.",
    )
    max_tokens: int = Field(
        default=128,
        ge=10,
        le=512,
        description="Token budget used during the session.",
    )
    seed: int = Field(
        ...,
        description="Seed value used during the session.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="IC system prompt used, or None if the server default was in effect.",
    )


class ChatSaveResponse(BaseModel):
    """
    Response body for POST /api/save_chat.

    Returns the save folder name and the list of files written.
    """

    folder_name: str = Field(
        ...,
        description="Save folder name under data/ (e.g. '20260227_103022_abc1def2').",
    )
    files: list[str] = Field(
        ...,
        description="Sorted list of filenames written inside the folder.",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of the save operation.",
    )


class ChatImportResponse(BaseModel):
    """
    Response body for ``POST /api/import_chat``.

    Contains everything the frontend needs to restore a chat translation
    session from an uploaded chat save-package zip.
    """

    folder_name: str = Field(
        ...,
        description="Original save folder name (e.g. '20260219_101437_5d628967').",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="The full metadata.json content as a dict.",
    )
    character_a: Optional[dict[str, Any]] = Field(
        default=None,
        description="Axes dict for Character A (axis name → {label, score}), or None.",
    )
    character_b: Optional[dict[str, Any]] = Field(
        default=None,
        description="Axes dict for Character B (axis name → {label, score}), or None.",
    )
    system_prompt: str = Field(
        ...,
        description="Plain text of the IC system prompt, fence-stripped from system_prompt.md.",
    )
    game_log_entries: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Parsed rows from game_log.md.  Each entry has keys: "
            "ch, channel, ooc_message, ic_text.  Empty when game_log.md "
            "was not present in the zip."
        ),
    )
    model: str = Field(..., description="Ollama model name from metadata.json.")
    temperature: float = Field(..., description="Sampling temperature from metadata.json.")
    max_tokens: int = Field(..., description="Token budget from metadata.json.")
    seed: int = Field(..., description="Seed value from metadata.json (-1 = random).")
    manifest_valid: bool = Field(
        ...,
        description="True if all manifest checksums passed verification.",
    )
    files: list[str] = Field(
        ...,
        description="Sorted list of filenames found in the uploaded zip.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during import.",
    )

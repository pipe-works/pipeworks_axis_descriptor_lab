"""
Schemas for character-description save, export, and import flows.

These models cover persistence for the main character-description page:
save requests, save responses, manifest entries, and import payloads used
to reconstruct session state from zip packages.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schema.analysis import TransformationMapRow
from app.schema.axis import AxisPayload


class SaveRequest(BaseModel):
    """
    Full request body for POST /api/save.

    The frontend collects all in-memory state at the moment the user clicks
    Save and sends it here.  The backend writes individual files to a
    timestamped subfolder under ``data/``.
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
    payload_name: str | None = Field(
        default=None,
        description=(
            "Selected example payload stem for the current session (for example "
            "'brittle_elite'), or None when the payload did not come from a "
            "named example."
        ),
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
    transformation_map: list[TransformationMapRow] | None = Field(
        default=None,
        description=(
            "Optional clause-level transformation map rows computed "
            "client-side from the word-level LCS diff.  When present, "
            "saved as transformation_map.json alongside other files."
        ),
    )
    diff_change_pct: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Word-level change percentage between baseline and current output, "
            "computed client-side as (insertions + deletions) / total words. "
            "None when no diff was computed (missing baseline or output)."
        ),
    )


class SaveResponse(BaseModel):
    """
    Response body for POST /api/save.

    Returns the save folder name and the list of files written so the
    frontend can display a confirmation message in the status bar.
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


class ManifestFileEntry(BaseModel):
    """
    A single file's entry in the save-package manifest.

    Provides the SHA-256 checksum, role classification, and byte size for
    one file within a save folder.  Used inside the ``manifest`` section
    of ``metadata.json`` to make save packages self-describing and
    verifiable.
    """

    sha256: Optional[str] = Field(
        ...,
        description=(
            "SHA-256 hex digest of the file's raw bytes.  None for "
            "metadata.json (which cannot hash itself without creating "
            "a circular dependency)."
        ),
    )
    role: str = Field(
        ...,
        description=(
            "Human-readable role classification for this file.  "
            "One of: provenance, payload, system_prompt, output, "
            "baseline, delta, transformation_map."
        ),
        examples=["payload", "output", "provenance"],
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="File size in bytes.  0 for metadata.json (sentinel).",
    )


class ImportResponse(BaseModel):
    """
    Response body for ``POST /api/import``.

    Contains everything the frontend needs to fully restore session state
    from an uploaded save-package zip file.  The backend parses the zip,
    validates manifest checksums (if present), and extracts plain text
    from the Markdown files so the frontend can populate the UI directly
    without any further parsing.
    """

    folder_name: str = Field(
        ...,
        description="Original save folder name (e.g. '20260219_101437_5d628967').",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="The full metadata.json content as a dict, including manifest.",
    )
    payload: AxisPayload = Field(
        ...,
        description="Parsed AxisPayload from payload.json.",
    )
    system_prompt: str = Field(
        ...,
        description=(
            "Plain text of the system prompt, extracted from system_prompt.md "
            "with Markdown heading and fenced code block stripped."
        ),
    )
    output: Optional[str] = Field(
        default=None,
        description=(
            "Plain text of the generated output, extracted from output.md "
            "with heading and HTML comment header stripped.  None if output.md "
            "was not present in the zip."
        ),
    )
    baseline: Optional[str] = Field(
        default=None,
        description=(
            "Plain text of the baseline, extracted from baseline.md with "
            "heading and HTML comment header stripped.  None if baseline.md "
            "was not present in the zip."
        ),
    )
    model: str = Field(
        ...,
        description="Ollama model name from metadata.json.",
    )
    temperature: float = Field(
        ...,
        description="Sampling temperature from metadata.json.",
    )
    max_tokens: int = Field(
        ...,
        description="Token budget from metadata.json.",
    )
    manifest_valid: bool = Field(
        ...,
        description=(
            "True if all manifest checksums passed verification, or if "
            "no manifest was present (backward compatibility).  False if "
            "any checksum mismatch was detected but import proceeded."
        ),
    )
    files: list[str] = Field(
        ...,
        description="Sorted list of filenames found in the uploaded zip.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal issues encountered during import (e.g. missing "
            "manifest, skipped unknown files).  Empty list when everything "
            "is clean."
        ),
    )

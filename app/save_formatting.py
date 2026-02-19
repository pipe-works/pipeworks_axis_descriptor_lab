"""
app/save_formatting.py
-----------------------------------------------------------------------------
Formatted Markdown builders and folder-name generator for the save system.

This module produces the human-readable Markdown files and filesystem-safe
folder names that make up a save package.  Each function is a pure formatter:
it takes scalar values and returns a string.  No I/O, no network, no
``app.*`` dependencies — only ``datetime`` from the standard library.

Exports
-------
save_folder_name(timestamp, input_hash) -> str
    Produce a unique folder name for a save operation.

build_output_md(text, model, temperature, max_tokens, seed, timestamp,
                input_hash, *, system_prompt_hash, ipc_id) -> str
    Format the generated LLM output as a Markdown document with a provenance
    header.

build_baseline_md(text, folder_name) -> str
    Format the stored baseline text as a Markdown document.

build_system_prompt_md(prompt_text, folder_name) -> str
    Format the system prompt as a Markdown document with a fenced code block.

Design notes
------------
These functions were extracted from ``main.py`` so that:

1. The save route handler can remain a thin orchestrator.
2. Formatting logic can be unit-tested without the HTTP layer.
3. ``build_output_md`` accepts scalar parameters (not ``SaveRequest``)
   to avoid tight coupling between formatting and the request schema.
"""

from __future__ import annotations

from datetime import datetime

# -----------------------------------------------------------------------------
# Folder name
# -----------------------------------------------------------------------------


def save_folder_name(timestamp: datetime, input_hash: str) -> str:
    """
    Produce a unique folder name for a save operation.

    Format: ``YYYYMMDD_HHMMSS_<8-char-hash-prefix>``

    Example: ``20260218_143022_d845cdcf``

    The 8-character hash prefix provides practical uniqueness even when two
    saves occur within the same second (different payload → different hash
    suffix).  The format uses only digits, underscores, and lowercase hex
    characters, making it safe for all major filesystems.

    Parameters
    ----------
    timestamp  : UTC datetime of the save (passed in so the folder name
                 stays consistent with the ``metadata.json`` timestamp).
    input_hash : Full 64-char SHA-256 hex digest of the AxisPayload.

    Returns
    -------
    str : Folder name safe for all major filesystems.
    """
    date_part = timestamp.strftime("%Y%m%d_%H%M%S")
    hash_part = input_hash[:8]
    return f"{date_part}_{hash_part}"


# -----------------------------------------------------------------------------
# Markdown builders
# -----------------------------------------------------------------------------


def build_output_md(
    text: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    timestamp: datetime,
    input_hash: str,
    *,
    system_prompt_hash: str | None = None,
    ipc_id: str | None = None,
) -> str:
    """
    Format the generated LLM output as a Markdown document.

    Includes an HTML-comment provenance header (model, temperature, seed,
    hashes) so the file is self-documenting when opened in any Markdown
    viewer.  The IPC hashes are included when available so saved files
    carry a complete reproducibility record.

    Parameters
    ----------
    text               : The raw LLM-generated text.
    model              : Ollama model identifier used for the generation.
    temperature        : Sampling temperature used.
    max_tokens         : Token budget used.
    seed               : RNG seed from the AxisPayload.
    timestamp          : UTC datetime of the save (for the provenance header).
    input_hash         : SHA-256 of the payload.
    system_prompt_hash : SHA-256 of the normalised system prompt (optional).
    ipc_id             : Interpretive Provenance Chain identifier (optional).

    Returns
    -------
    str : Markdown string ready to write to disk.
    """
    lines = [
        "# Output",
        "",
        "<!-- Axis Descriptor Lab – generated output -->",
        f"<!-- saved: {timestamp.isoformat()} -->",
        f"<!-- model: {model} | temp: {temperature} | max_tokens: {max_tokens} -->",
        f"<!-- seed: {seed} | input_hash: {input_hash[:16]}... -->",
    ]

    # Append IPC provenance hashes when available so the saved file carries
    # a complete reproducibility record without needing metadata.json.
    if system_prompt_hash:
        lines.append(f"<!-- system_prompt_hash: {system_prompt_hash[:16]}... -->")
    if ipc_id:
        lines.append(f"<!-- ipc_id: {ipc_id[:16]}... -->")

    lines += ["", text, ""]
    return "\n".join(lines)


def build_baseline_md(text: str, folder_name: str) -> str:
    """
    Format the stored baseline text as a Markdown document.

    Parameters
    ----------
    text        : The baseline text (state.baseline from the frontend).
    folder_name : Save folder name (used in the provenance comment).

    Returns
    -------
    str : Markdown string ready to write to disk.
    """
    lines = [
        "# Baseline (A)",
        "",
        f"<!-- Axis Descriptor Lab – baseline text for save {folder_name} -->",
        "",
        text,
        "",
    ]
    return "\n".join(lines)


def build_system_prompt_md(prompt_text: str, folder_name: str) -> str:
    """
    Format the system prompt as a Markdown document with a fenced code block.

    Wrapping in a fenced code block preserves all whitespace and makes the
    prompt clearly machine-readable when opened in a Markdown viewer.

    Parameters
    ----------
    prompt_text : The system prompt string (may be multi-line).
    folder_name : Save folder name (for the provenance comment).

    Returns
    -------
    str : Markdown string ready to write to disk.
    """
    lines = [
        "# System Prompt",
        "",
        f"<!-- Axis Descriptor Lab – system prompt for save {folder_name} -->",
        "",
        "```text",
        prompt_text,
        "```",
        "",
    ]
    return "\n".join(lines)

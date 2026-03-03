"""
app/file_loaders.py
-----------------------------------------------------------------------------
File-loading utilities for the Axis Descriptor Lab.

This module reads example JSON files from ``app/examples/`` and prompt text
files from a purpose-grouped ``app/prompts/`` tree. Prompt files are split by
the page or flow that owns them so the frontend can populate cleaner,
purpose-specific dropdowns instead of relying on filename prefixes.

Current prompt groups
---------------------
- ``app/prompts/character_description/`` — descriptive-generation system
  prompts used by the Character Description page.
- ``app/prompts/chat_translation/`` — standalone IC translation prompt
  templates used by the Chat Translation page.

All path resolution is relative to this file's parent directory (``app/``),
so the loaders work regardless of the working directory from which uvicorn
is launched.

Exports
-------
load_default_prompt() -> str
    Read and return the default Character Description system prompt.

load_chat_default_prompt() -> str
    Read and return the default Chat Translation prompt template.

load_example(name) -> dict
    Load and parse a named example JSON file.

load_prompt(name) -> str
    Load a named prompt text file.

list_example_names() -> list[str]
    Return sorted stems of all ``.json`` files in ``app/examples/``.

list_prompt_names() -> list[str]
    Return sorted stems of all ``.txt`` files in the prompt tree, optionally
    filtered by prompt purpose.

Dependencies
------------
Uses ``fastapi.HTTPException`` for error signalling so that callers (route
handlers in ``main.py``) get properly formatted HTTP error responses without
extra try/except boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import HTTPException

# Resolve directories relative to this file so paths work regardless of
# the current working directory at import time.
_HERE = Path(__file__).parent
PROMPTS_DIR = _HERE / "prompts"
EXAMPLES_DIR = _HERE / "examples"

type PromptPurpose = Literal["character_description", "chat_translation"]

DEFAULT_CHARACTER_DESCRIPTION_PROMPT = "system_prompt_v01"
DEFAULT_CHAT_TRANSLATION_PROMPT = "pipeworks_web_ic_prompt"


def _prompt_dirs() -> dict[PromptPurpose, Path]:
    """
    Return the authoritative prompt-group directory mapping.

    The mapping is produced on demand instead of being frozen at import time
    so tests can patch ``PROMPTS_DIR`` without also needing to patch derived
    child-path constants.
    """

    return {
        "character_description": PROMPTS_DIR / "character_description",
        "chat_translation": PROMPTS_DIR / "chat_translation",
    }


def _iter_prompt_files(purpose: PromptPurpose | None = None) -> list[Path]:
    """
    Return all prompt files for the requested purpose.

    Parameters
    ----------
    purpose : PromptPurpose | None
        When provided, only prompt files under that purpose directory are
        returned. When ``None``, files from every prompt group are returned.

    Returns
    -------
    list[Path]
        Sorted prompt file paths.
    """

    prompt_dirs = _prompt_dirs()
    if purpose is not None:
        return sorted(prompt_dirs[purpose].rglob("*.txt"))

    paths: list[Path] = []
    for path in prompt_dirs.values():
        paths.extend(path.rglob("*.txt"))
    return sorted(paths)


def _build_prompt_index(purpose: PromptPurpose | None = None) -> dict[str, Path]:
    """
    Build a stem → file-path index for prompt lookup.

    Prompt names remain the user-facing API contract, so every prompt stem
    must be unique across whichever scope is being indexed. If two files
    share the same stem inside the same lookup scope, the app raises a 500 so
    the ambiguity is surfaced immediately instead of returning the wrong file.

    Parameters
    ----------
    purpose : PromptPurpose | None
        Optional prompt-purpose filter.

    Returns
    -------
    dict[str, Path]
        Mapping of prompt stem to its source file.
    """

    index: dict[str, Path] = {}
    for path in _iter_prompt_files(purpose):
        stem = path.stem
        existing = index.get(stem)
        if existing is not None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Duplicate prompt name detected: "
                    f"'{stem}' is defined in both {existing} and {path}"
                ),
            )
        index[stem] = path
    return index


# -----------------------------------------------------------------------------
# Default prompt
# -----------------------------------------------------------------------------


def load_default_prompt() -> str:
    """
    Read the default system prompt from disk.

    Returns the text of the default Character Description prompt,
    ``app/prompts/character_description/system_prompt_v01.txt``, stripped of
    leading and trailing whitespace.

    Returns
    -------
    str : The default system prompt text.

    Raises
    ------
    HTTPException(500)
        If the file is missing (indicates a broken deployment).
    """
    try:
        return load_prompt(
            DEFAULT_CHARACTER_DESCRIPTION_PROMPT,
            purpose="character_description",
        )
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Default system prompt not found under " "app/prompts/character_description/"
                ),
            ) from exc
        raise


def load_chat_default_prompt() -> str:
    """
    Read the default Chat Translation prompt from disk.

    Returns the text of the default standalone chat prompt,
    ``app/prompts/chat_translation/pipeworks_web_ic_prompt.txt``, stripped
    of leading and trailing whitespace.

    Returns
    -------
    str
        The default Chat Translation prompt text.

    Raises
    ------
    HTTPException(500)
        If the default chat prompt is missing.
    """

    try:
        return load_prompt(
            DEFAULT_CHAT_TRANSLATION_PROMPT,
            purpose="chat_translation",
        )
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Default chat translation prompt not found under "
                    "app/prompts/chat_translation/"
                ),
            ) from exc
        raise


# -----------------------------------------------------------------------------
# Example loading
# -----------------------------------------------------------------------------


def load_example(name: str) -> dict:
    """
    Load and parse a named example JSON file from ``app/examples/``.

    Parameters
    ----------
    name : Bare filename without extension (e.g. ``"proud_operator"``).

    Returns
    -------
    dict : Parsed JSON object.

    Raises
    ------
    HTTPException(404)
        If the file doesn't exist.
    HTTPException(500)
        If the file contains invalid JSON.
    """
    path = EXAMPLES_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Example '{name}' not found.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Example '{name}' contains invalid JSON: {exc}"
        ) from exc


def list_example_names() -> list[str]:
    """
    Return a sorted list of example names (without ``.json`` extension).

    Scans ``app/examples/`` for all ``.json`` files and returns their
    stems in alphabetical order.  Used by the ``GET /api/examples`` route
    to populate the frontend dropdown.

    Returns
    -------
    list[str] : Sorted example name stems.
    """
    return sorted(p.stem for p in EXAMPLES_DIR.glob("*.json"))


# -----------------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------------


def load_prompt(name: str, purpose: PromptPurpose | None = None) -> str:
    """
    Load a named prompt text file from the grouped ``app/prompts/`` tree.

    Unlike :func:`load_example` which parses structured JSON, this simply
    reads the file as plain UTF-8 text and returns it stripped of
    leading/trailing whitespace.  Prompts are natural-language instructions
    for the LLM, not structured data.

    Parameters
    ----------
    name : Bare filename without extension (e.g. ``"system_prompt_v01"``).
    purpose : PromptPurpose | None
        Optional prompt-purpose filter. When provided, the lookup is limited
        to that prompt group.

    Returns
    -------
    str : The prompt text content, stripped of surrounding whitespace.

    Raises
    ------
    HTTPException(404)
        If the file doesn't exist.
    """
    path = _build_prompt_index(purpose).get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found.")
    return path.read_text(encoding="utf-8").strip()


def list_prompt_names(purpose: PromptPurpose | None = None) -> list[str]:
    """
    Return a sorted list of prompt names (without ``.txt`` extension).

    Scans the grouped ``app/prompts/`` tree and returns prompt stems in
    alphabetical order. Used by ``GET /api/prompts`` to populate prompt
    dropdowns. The optional ``purpose`` filter lets each page request only
    the prompt family it actually uses.

    Returns
    -------
    list[str] : Sorted prompt name stems.
    """
    return sorted(_build_prompt_index(purpose).keys())

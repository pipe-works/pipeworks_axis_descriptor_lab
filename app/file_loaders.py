"""
app/file_loaders.py
-----------------------------------------------------------------------------
File-loading utilities for the Axis Descriptor Lab.

This module reads example JSON files from ``app/examples/`` and prompt text
files from ``app/prompts/``.  It also provides listing functions that return
sorted lists of available file names (stems without extensions).

All path resolution is relative to this file's parent directory (``app/``),
so the loaders work regardless of the working directory from which uvicorn
is launched.

Exports
-------
load_default_prompt() -> str
    Read and return the default system prompt (``system_prompt_v01.txt``).

load_example(name) -> dict
    Load and parse a named example JSON file.

load_prompt(name) -> str
    Load a named prompt text file.

list_example_names() -> list[str]
    Return sorted stems of all ``.json`` files in ``app/examples/``.

list_prompt_names() -> list[str]
    Return sorted stems of all ``.txt`` files in ``app/prompts/``.

Dependencies
------------
Uses ``fastapi.HTTPException`` for error signalling so that callers (route
handlers in ``main.py``) get properly formatted HTTP error responses without
extra try/except boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

# Resolve directories relative to this file so paths work regardless of
# the current working directory at import time.
_HERE = Path(__file__).parent
PROMPTS_DIR = _HERE / "prompts"
EXAMPLES_DIR = _HERE / "examples"


# -----------------------------------------------------------------------------
# Default prompt
# -----------------------------------------------------------------------------


def load_default_prompt() -> str:
    """
    Read the default system prompt from disk.

    Returns the text of ``app/prompts/system_prompt_v01.txt``, stripped of
    leading and trailing whitespace.

    Returns
    -------
    str : The default system prompt text.

    Raises
    ------
    HTTPException(500)
        If the file is missing (indicates a broken deployment).
    """
    prompt_path = PROMPTS_DIR / "system_prompt_v01.txt"
    if not prompt_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Default system prompt not found at {prompt_path}",
        )
    return prompt_path.read_text(encoding="utf-8").strip()


# -----------------------------------------------------------------------------
# Example loading
# -----------------------------------------------------------------------------


def load_example(name: str) -> dict:
    """
    Load and parse a named example JSON file from ``app/examples/``.

    Parameters
    ----------
    name : Bare filename without extension (e.g. ``"example_a"``).

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


def load_prompt(name: str) -> str:
    """
    Load a named prompt text file from ``app/prompts/``.

    Unlike :func:`load_example` which parses structured JSON, this simply
    reads the file as plain UTF-8 text and returns it stripped of
    leading/trailing whitespace.  Prompts are natural-language instructions
    for the LLM, not structured data.

    Parameters
    ----------
    name : Bare filename without extension (e.g. ``"system_prompt_v01"``).

    Returns
    -------
    str : The prompt text content, stripped of surrounding whitespace.

    Raises
    ------
    HTTPException(404)
        If the file doesn't exist.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found.")
    return path.read_text(encoding="utf-8").strip()


def list_prompt_names() -> list[str]:
    """
    Return a sorted list of prompt names (without ``.txt`` extension).

    Scans ``app/prompts/`` for all ``.txt`` files and returns their stems
    in alphabetical order.  Used by the ``GET /api/prompts`` route to
    populate the frontend prompt library dropdown.

    Returns
    -------
    list[str] : Sorted prompt name stems.
    """
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))

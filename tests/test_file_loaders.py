"""
Tests for app/file_loaders.py — file loading and listing utilities.

These tests were migrated from test_main.py (where they tested private helpers)
and adapted to the new public-function signatures and module-level patch targets.

Test strategy
-------------
1. Happy-path loading from the real ``app/examples/`` and ``app/prompts/`` dirs.
2. Error cases (missing files, invalid JSON) using ``tmp_path`` + ``patch``.
3. Listing functions return sorted names from the real directories.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.file_loaders import (
    list_example_names,
    list_prompt_names,
    load_default_prompt,
    load_example,
    load_prompt,
)

# ── load_default_prompt ─────────────────────────────────────────────────────


class TestLoadDefaultPrompt:
    """Tests for the load_default_prompt() function."""

    def test_loads_prompt(self) -> None:
        """The default prompt file must exist and contain meaningful text."""
        prompt = load_default_prompt()
        assert "authoritative" in prompt.lower() or "ornamental" in prompt.lower()
        assert len(prompt) > 50

    def test_missing_prompt_raises(self, tmp_path: Path) -> None:
        """A missing prompt file must raise an exception."""
        with patch("app.file_loaders.PROMPTS_DIR", tmp_path):
            with pytest.raises(Exception):
                load_default_prompt()


# ── load_example ────────────────────────────────────────────────────────────


class TestLoadExample:
    """Tests for the load_example() function."""

    def test_loads_example_a(self) -> None:
        """Loading 'example_a' must return a dict with axes and seed."""
        data = load_example("example_a")
        assert "axes" in data
        assert "seed" in data

    def test_missing_example_raises_404(self) -> None:
        """A non-existent example name must raise HTTPException(404)."""
        with pytest.raises(HTTPException) as exc_info:
            load_example("nonexistent_example")
        assert exc_info.value.status_code == 404

    def test_invalid_json_raises_500(self, tmp_path: Path) -> None:
        """An example file with invalid JSON must raise HTTPException(500)."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{", encoding="utf-8")
        with patch("app.file_loaders.EXAMPLES_DIR", tmp_path):
            with pytest.raises(HTTPException) as exc_info:
                load_example("bad")
            assert exc_info.value.status_code == 500


# ── load_prompt ─────────────────────────────────────────────────────────────


class TestLoadPrompt:
    """Tests for the load_prompt() function."""

    def test_loads_default_prompt(self) -> None:
        """Loading system_prompt_v01 should return the known default prompt."""
        text = load_prompt("system_prompt_v01")
        assert "ornamental" in text.lower()
        assert len(text) > 50

    def test_missing_prompt_raises_404(self) -> None:
        """A non-existent prompt name must raise HTTPException(404)."""
        with pytest.raises(HTTPException) as exc_info:
            load_prompt("nonexistent_prompt_xyz")
        assert exc_info.value.status_code == 404

    def test_returns_stripped_text(self, tmp_path: Path) -> None:
        """Loaded prompt text must be stripped of leading/trailing whitespace."""
        prompt_file = tmp_path / "padded.txt"
        prompt_file.write_text("  \n  Hello world  \n  ", encoding="utf-8")
        with patch("app.file_loaders.PROMPTS_DIR", tmp_path):
            text = load_prompt("padded")
        assert text == "Hello world"


# ── list_example_names ──────────────────────────────────────────────────────


class TestListExampleNames:
    """Tests for the list_example_names() function."""

    def test_returns_sorted_list(self) -> None:
        """Must return a sorted list containing at least example_a and example_b."""
        names = list_example_names()
        assert isinstance(names, list)
        assert "example_a" in names
        assert "example_b" in names
        assert names == sorted(names)


# ── list_prompt_names ───────────────────────────────────────────────────────


class TestListPromptNames:
    """Tests for the list_prompt_names() function."""

    def test_returns_sorted_list(self) -> None:
        """Must return a sorted list containing at least system_prompt_v01."""
        names = list_prompt_names()
        assert isinstance(names, list)
        assert "system_prompt_v01" in names
        assert names == sorted(names)

    def test_includes_variant_prompts(self) -> None:
        """All known prompt variants must appear."""
        names = list_prompt_names()
        assert len(names) >= 4
        assert "system_prompt_v02_terse" in names
        assert "system_prompt_v03_environmental" in names
        assert "system_prompt_v04_contrast" in names

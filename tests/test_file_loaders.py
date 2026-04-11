"""
Tests for app/file_loaders.py — file loading and listing utilities.

These tests were migrated from test_main.py (where they tested private helpers)
and adapted to the new public-function signatures and module-level patch targets.

Test strategy
-------------
1. Happy-path loading from the supported lab/world local asset roots.
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
    load_chat_default_prompt,
    load_default_prompt,
    load_example,
    load_prompt,
)
from app.path_resolver import PathResolutionError

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
        with (
            patch("app.file_loaders.PROMPTS_DIR", tmp_path),
            patch("app.file_loaders.WORLD_ASSET_ROOT", tmp_path / "worlds"),
            patch("app.file_loaders.LAB_ONLY_ASSET_ROOT", tmp_path / "lab_only"),
        ):
            with pytest.raises(Exception):
                load_default_prompt()


class TestLoadChatDefaultPrompt:
    """Tests for the load_chat_default_prompt() function."""

    def test_loads_prompt(self) -> None:
        """The default standalone chat prompt must exist and contain template text."""
        prompt = load_chat_default_prompt()
        assert "{{profile_summary}}" in prompt
        assert len(prompt) > 50

    def test_missing_prompt_raises(self, tmp_path: Path) -> None:
        """A missing chat default prompt must raise an exception."""
        with (
            patch("app.file_loaders.PROMPTS_DIR", tmp_path),
            patch("app.file_loaders.WORLD_ASSET_ROOT", tmp_path / "worlds"),
            patch("app.file_loaders.LAB_ONLY_ASSET_ROOT", tmp_path / "lab_only"),
        ):
            with pytest.raises(Exception):
                load_chat_default_prompt()


# ── load_example ────────────────────────────────────────────────────────────


class TestLoadExample:
    """Tests for the load_example() function."""

    def test_loads_proud_operator(self) -> None:
        """Loading 'proud_operator' must return a dict with axes and seed."""
        data = load_example("proud_operator")
        assert "axes" in data
        assert "seed" in data

    def test_missing_example_raises_404(self) -> None:
        """A non-existent example name must raise HTTPException(404)."""
        with pytest.raises(HTTPException) as exc_info:
            load_example("nonexistent_example")
        assert exc_info.value.status_code == 404

    def test_invalid_json_raises_500(self, tmp_path: Path) -> None:
        """An example file with invalid JSON must raise HTTPException(500)."""
        bad_file = tmp_path / "axis" / "examples" / "bad.json"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("not json {{{", encoding="utf-8")
        with patch("app.file_loaders.LAB_ONLY_ASSET_ROOT", tmp_path):
            with pytest.raises(HTTPException) as exc_info:
                load_example("bad")
            assert exc_info.value.status_code == 500

    def test_resolution_error_raises_500(self) -> None:
        """Resolver ambiguity must surface as HTTPException(500)."""
        with patch(
            "app.file_loaders.resolve_axis_payload_paths",
            side_effect=PathResolutionError("ambiguous axis payload"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                load_example("proud_operator")
        assert exc_info.value.status_code == 500
        assert "ambiguous axis payload" in str(exc_info.value.detail)

    def test_prefers_lab_only_example_when_world_example_absent(self, tmp_path: Path) -> None:
        """Lab-only examples must remain available when no world example exists."""
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        (lab_root / "axis" / "examples").mkdir(parents=True)
        (lab_root / "axis" / "examples" / "proud_operator.json").write_text(
            '{"axes":{"demeanor":{"label":"lab","score":0.9}},"policy_hash":"x","seed":1,"world_id":"pipeworks_web"}',
            encoding="utf-8",
        )

        with (
            patch("app.file_loaders.WORLD_ASSET_ROOT", world_root),
            patch("app.file_loaders.LAB_ONLY_ASSET_ROOT", lab_root),
            patch("app.file_loaders.EXAMPLES_DIR", lab_root / "axis" / "examples"),
        ):
            data = load_example("proud_operator")
        assert data["axes"]["demeanor"]["label"] == "lab"


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
        prompt_dir = tmp_path / "prompts" / "character_description"
        prompt_dir.mkdir(parents=True)
        prompt_file = prompt_dir / "padded.txt"
        prompt_file.write_text("  \n  Hello world  \n  ", encoding="utf-8")
        with patch("app.file_loaders.LAB_ONLY_ASSET_ROOT", tmp_path):
            text = load_prompt("padded")
        assert text == "Hello world"

    def test_respects_purpose_filter(self) -> None:
        """Purpose-filtered prompt lookup must only search that prompt group."""
        text = load_prompt("pipeworks_web_ic_prompt", purpose="chat_translation")
        assert "{{profile_summary}}" in text

        with pytest.raises(HTTPException) as exc_info:
            load_prompt("pipeworks_web_ic_prompt", purpose="character_description")
        assert exc_info.value.status_code == 404

    def test_prefers_world_prompt_over_lab_prompt_on_stem_collision(self, tmp_path: Path) -> None:
        """World-scoped prompt paths must win over lab-only paths for same stem."""
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        (
            world_root
            / "pipeworks_web"
            / "policies"
            / "translation"
            / "prompts"
            / "ic"
            / "pipeworks_web_ic_prompt.txt"
        ).parent.mkdir(parents=True, exist_ok=True)
        (
            world_root
            / "pipeworks_web"
            / "policies"
            / "translation"
            / "prompts"
            / "ic"
            / "pipeworks_web_ic_prompt.txt"
        ).write_text("world prompt", encoding="utf-8")
        (lab_root / "prompts" / "chat_translation").mkdir(parents=True, exist_ok=True)
        (lab_root / "prompts" / "chat_translation" / "pipeworks_web_ic_prompt.txt").write_text(
            "lab prompt",
            encoding="utf-8",
        )

        with (
            patch("app.file_loaders.WORLD_ASSET_ROOT", world_root),
            patch("app.file_loaders.LAB_ONLY_ASSET_ROOT", lab_root),
            patch("app.file_loaders.PROMPTS_DIR", lab_root / "prompts"),
        ):
            text = load_prompt("pipeworks_web_ic_prompt", purpose="chat_translation")
        assert text == "world prompt"


# ── list_example_names ──────────────────────────────────────────────────────


class TestListExampleNames:
    """Tests for the list_example_names() function."""

    def test_returns_sorted_list(self) -> None:
        """Must return a sorted list containing the shipped archetype examples."""
        names = list_example_names()
        assert isinstance(names, list)
        assert "proud_operator" in names
        assert "resentful_debtor" in names
        assert names == sorted(names)

    def test_resolution_error_raises_500(self) -> None:
        """Resolver ambiguity must surface as HTTPException(500)."""
        with patch(
            "app.file_loaders.resolve_axis_payload_paths",
            side_effect=PathResolutionError("ambiguous examples"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                list_example_names()
        assert exc_info.value.status_code == 500
        assert "ambiguous examples" in str(exc_info.value.detail)


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
        assert len(names) >= 6
        assert "system_prompt_v02_terse" in names
        assert "system_prompt_v03_environmental" in names
        assert "system_prompt_v04_contrast" in names
        assert "pipeworks_web_ic_prompt" in names
        assert "daily_undertaking_ic_prompt" in names

    def test_filters_character_description_prompts(self) -> None:
        """Character Description listing must exclude chat translation prompts."""
        names = list_prompt_names("character_description")
        assert "system_prompt_v01" in names
        assert "system_prompt_v04_contrast" in names
        assert "pipeworks_web_ic_prompt" not in names

    def test_filters_chat_translation_prompts(self) -> None:
        """Chat Translation listing must exclude character description prompts."""
        names = list_prompt_names("chat_translation")
        assert "pipeworks_web_ic_prompt" in names
        assert "daily_undertaking_ic_prompt" in names
        assert "system_prompt_v01" not in names

    def test_resolution_error_raises_500_for_filtered_listing(self) -> None:
        """Prompt resolver ambiguity must surface as HTTPException(500)."""
        with patch(
            "app.file_loaders.resolve_prompt_paths",
            side_effect=PathResolutionError("ambiguous prompts"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                list_prompt_names("chat_translation")
        assert exc_info.value.status_code == 500
        assert "ambiguous prompts" in str(exc_info.value.detail)

    def test_resolution_error_raises_500_for_unfiltered_listing(self) -> None:
        """Unfiltered listing must also surface resolver ambiguities."""
        with patch(
            "app.file_loaders.resolve_prompt_paths",
            side_effect=PathResolutionError("ambiguous mixed listing"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                list_prompt_names()
        assert exc_info.value.status_code == 500
        assert "ambiguous mixed listing" in str(exc_info.value.detail)

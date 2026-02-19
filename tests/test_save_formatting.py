"""
Tests for app/save_formatting.py — folder name generation and Markdown builders.

These tests were migrated from test_main.py (where they tested private helpers)
and adapted to the new scalar-parameter signatures.  The formatting functions
are pure (no I/O, no network) so these tests are fast and deterministic.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.save_formatting import (
    build_baseline_md,
    build_output_md,
    build_system_prompt_md,
    save_folder_name,
)

# ── save_folder_name ────────────────────────────────────────────────────────


class TestSaveFolderName:
    """Tests for the save_folder_name() function."""

    def test_format_matches_expected_pattern(self) -> None:
        """Folder name must be YYYYMMDD_HHMMSS_<8 hex chars>."""
        now = datetime(2026, 2, 18, 14, 30, 22, tzinfo=timezone.utc)
        hash_str = "d845cdcf" + "a" * 56  # 64-char hex string
        name = save_folder_name(now, hash_str)

        assert name == "20260218_143022_d845cdcf"
        assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", name)

    def test_uses_first_eight_chars_of_hash(self) -> None:
        """Only the first 8 characters of the hash should appear."""
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        name = save_folder_name(now, "abcdef01" + "0" * 56)
        assert name.endswith("_abcdef01")

    def test_different_hashes_produce_different_names(self) -> None:
        """Same timestamp but different hashes must produce different names."""
        now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        name_a = save_folder_name(now, "aaaa" * 16)
        name_b = save_folder_name(now, "bbbb" * 16)
        assert name_a != name_b


# ── build_output_md ─────────────────────────────────────────────────────────


class TestBuildOutputMd:
    """Tests for the build_output_md() Markdown builder.

    These tests use scalar parameters (not SaveRequest) matching the
    extracted function's signature.
    """

    def test_contains_text_and_provenance(self) -> None:
        """Output MD must include the generated text and provenance comments."""
        now = datetime(2026, 2, 18, 14, 0, 0, tzinfo=timezone.utc)
        md = build_output_md(
            text="A weathered figure.",
            model="gemma2:2b",
            temperature=0.2,
            max_tokens=120,
            seed=42,
            timestamp=now,
            input_hash="d845" + "0" * 60,
        )

        assert "# Output" in md
        assert "A weathered figure." in md
        assert "gemma2:2b" in md
        assert "2026-02-18" in md
        assert "d845" in md

    def test_includes_ipc_provenance_when_provided(self) -> None:
        """When system_prompt_hash and ipc_id are passed, they appear in the output."""
        now = datetime(2026, 2, 18, 14, 0, 0, tzinfo=timezone.utc)
        sp_hash = "ab" * 32  # 64-char hex
        ipc = "cd" * 32  # 64-char hex
        md = build_output_md(
            text="A weathered figure.",
            model="gemma2:2b",
            temperature=0.2,
            max_tokens=120,
            seed=42,
            timestamp=now,
            input_hash="d845" + "0" * 60,
            system_prompt_hash=sp_hash,
            ipc_id=ipc,
        )

        assert sp_hash[:16] in md
        assert ipc[:16] in md

    def test_omits_ipc_provenance_when_not_provided(self) -> None:
        """When no IPC hashes are passed, their labels must not appear."""
        now = datetime(2026, 2, 18, 14, 0, 0, tzinfo=timezone.utc)
        md = build_output_md(
            text="A weathered figure.",
            model="gemma2:2b",
            temperature=0.2,
            max_tokens=120,
            seed=42,
            timestamp=now,
            input_hash="d845" + "0" * 60,
        )

        assert "system_prompt_hash" not in md
        assert "ipc_id" not in md


# ── build_baseline_md ───────────────────────────────────────────────────────


class TestBuildBaselineMd:
    """Tests for the build_baseline_md() Markdown builder."""

    def test_contains_text_and_folder_ref(self) -> None:
        """Baseline MD must include the text and reference the save folder."""
        md = build_baseline_md("Old description text.", "20260218_140000_abcd1234")

        assert "# Baseline (A)" in md
        assert "Old description text." in md
        assert "20260218_140000_abcd1234" in md


# ── build_system_prompt_md ──────────────────────────────────────────────────


class TestBuildSystemPromptMd:
    """Tests for the build_system_prompt_md() Markdown builder."""

    def test_contains_prompt_in_code_block(self) -> None:
        """System prompt MD must wrap the text in a fenced code block."""
        md = build_system_prompt_md("You are a descriptive layer.", "20260218_test")

        assert "# System Prompt" in md
        assert "```text" in md
        assert "You are a descriptive layer." in md
        assert "20260218_test" in md

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
    build_game_log_md,
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


# ── build_game_log_md ────────────────────────────────────────────────────────


class TestBuildGameLogMd:
    """Tests for the build_game_log_md() Markdown builder.

    Covers the five-column table format introduced when OOC message recording
    was added (Char | OOC | Channel | IC Text).
    """

    # ── Fixtures ──────────────────────────────────────────────────────────── #

    @staticmethod
    def _ts() -> datetime:
        return datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)

    @staticmethod
    def _entry(
        ch: str = "a",
        channel: str = "say",
        ooc: str = "she looks around",
        ic: str = "She surveys the chamber with wary eyes.",
        model: str = "gemma2:2b",
        ipc_id: str | None = None,
    ) -> dict:
        """Construct a minimal serialised ChatLogEntry dict for testing."""
        return {
            "ch": ch,
            "channel": channel,
            "ooc_message": ooc,
            "ic_text": ic,
            "model": model,
            "ipc_id": ipc_id,
        }

    # ── Structure tests ───────────────────────────────────────────────────── #

    def test_contains_heading_and_provenance(self) -> None:
        """Output must have the # In-Game Log heading and a provenance comment."""
        md = build_game_log_md(
            entries=[self._entry()],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=42,
            timestamp=self._ts(),
        )

        assert "# In-Game Log" in md
        assert "<!-- Axis Descriptor Lab" in md
        assert "gemma2:2b" in md
        assert "2026-02-26" in md

    def test_five_column_header_present(self) -> None:
        """Table header must have five columns: #, Char, OOC, Channel, IC Text."""
        md = build_game_log_md(
            entries=[self._entry()],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=42,
            timestamp=self._ts(),
        )

        assert "| # | Char | OOC | Channel | IC Text |" in md
        assert "| --- | --- | --- | --- | --- |" in md

    def test_entry_row_contains_all_five_columns(self) -> None:
        """Each data row must carry index, char, OOC, channel, and IC text."""
        ooc = "she glances toward the door"
        ic = "Her eyes drift to the entrance."
        md = build_game_log_md(
            entries=[self._entry(ch="a", channel="whisper", ooc=ooc, ic=ic)],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=99,
            timestamp=self._ts(),
        )

        # Verify all five values appear in the rendered table row.
        assert "| 1 |" in md
        assert "A" in md
        assert ooc in md
        assert "whisper" in md
        assert ic in md

    def test_character_uppercased_in_output(self) -> None:
        """The 'ch' field ('a' / 'b') must be uppercased to 'A' / 'B'."""
        md = build_game_log_md(
            entries=[self._entry(ch="b")],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=0,
            timestamp=self._ts(),
        )

        # 'B' should appear in the table; lowercase 'b' in the data row would
        # indicate the uppercasing step was skipped.
        assert "| B |" in md

    # ── Pipe-escaping tests ────────────────────────────────────────────────── #

    def test_pipe_in_ooc_is_escaped(self) -> None:
        """Pipe characters in the OOC field must be backslash-escaped."""
        md = build_game_log_md(
            entries=[self._entry(ooc="A|B split")],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=0,
            timestamp=self._ts(),
        )

        assert "A\\|B split" in md
        # The raw unescaped pipe with surrounding spaces would break the table.
        assert "A|B split" not in md.split("| --- |")[1]  # only check data rows

    def test_pipe_in_ic_text_is_escaped(self) -> None:
        """Pipe characters in the IC text field must be backslash-escaped."""
        md = build_game_log_md(
            entries=[self._entry(ic="left | right")],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=0,
            timestamp=self._ts(),
        )

        assert "left \\| right" in md

    # ── Legacy / backward-compat tests ───────────────────────────────────── #

    def test_missing_ooc_message_renders_as_empty_cell(self) -> None:
        """Entries without ooc_message (legacy) must render with an empty OOC cell."""
        entry = {
            "ch": "a",
            "channel": "say",
            # ooc_message deliberately omitted — simulates a legacy entry.
            "ic_text": "She glances around.",
            "model": "gemma2:2b",
            "ipc_id": None,
        }
        md = build_game_log_md(
            entries=[entry],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=0,
            timestamp=self._ts(),
        )

        # The OOC cell should be empty (two consecutive pipes with only spaces).
        assert "|  |" in md or "| |" in md

    def test_none_ooc_message_renders_as_empty_cell(self) -> None:
        """Entries with ooc_message=None must render with an empty OOC cell."""
        entry = self._entry(ooc=None)  # type: ignore[arg-type]
        # Manually override because _entry enforces a str default.
        entry["ooc_message"] = None
        md = build_game_log_md(
            entries=[entry],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=0,
            timestamp=self._ts(),
        )

        # The OOC cell should be empty string, resulting in two adjacent pipes.
        assert "|  |" in md or "| |" in md

    # ── Multi-entry test ──────────────────────────────────────────────────── #

    def test_multiple_entries_indexed_correctly(self) -> None:
        """Row indices must be 1-based and sequential across entries."""
        entries = [
            self._entry(ch="a", ooc="walks in", ic="She enters the room."),
            self._entry(ch="b", ooc="nods at her", ic="He dips his head slightly."),
            self._entry(ch="a", ooc="smiles back", ic="A faint smile crosses her lips."),
        ]
        md = build_game_log_md(
            entries=entries,
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=128,
            seed=0,
            timestamp=self._ts(),
        )

        # All three row indices must appear.
        assert "| 1 |" in md
        assert "| 2 |" in md
        assert "| 3 |" in md
        # All three OOC messages must appear.
        assert "walks in" in md
        assert "nods at her" in md
        assert "smiles back" in md

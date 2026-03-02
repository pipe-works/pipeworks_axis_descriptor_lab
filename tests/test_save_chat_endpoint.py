"""
Tests for POST /api/save_chat — in-game chat log save endpoint.

Each test uses a temporary ``data/`` directory (via ``tmp_path`` and
monkeypatching ``app.main.BASE_DIR``) so no real files are written to the
working tree.  The existing ``GET /api/save/{folder}/export`` endpoint is
also tested to confirm the chat save folder can be exported as a zip.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def axes_a() -> dict:
    return {
        "health": {"label": "weary", "score": 0.3},
        "age": {"label": "old", "score": 0.75},
    }


@pytest.fixture()
def axes_b() -> dict:
    return {
        "health": {"label": "vigorous", "score": 0.8},
        "age": {"label": "young", "score": 0.25},
    }


@pytest.fixture()
def base_entry(axes_a: dict) -> dict:
    """
    Minimal single log entry with all IPC provenance hash fields.

    The four hash fields (input_hash, system_prompt_hash, output_hash, ipc_id)
    mirror what the frontend stores in chatState.gameLog after a successful
    translation and what the browser's IPC meta table displays.
    """
    return {
        "ch": "a",
        "channel": "say",
        "ooc_message": "she looks around cautiously",
        "ic_text": "She peers cautiously about the chamber.",
        "system_prompt": "Server prompt A: {{profile_summary}}",
        "model": "gemma2:2b",
        "ipc_id": "a1e28854078de8dd" + "a" * 48,
        "input_hash": "cd712615b3b3670a" + "b" * 48,
        "system_prompt_hash": "dab5aabaa2474094" + "c" * 48,
        "output_hash": "ea0c157dd83dbffb" + "d" * 48,
    }


@pytest.fixture()
def base_request(base_entry: dict) -> dict:
    """Minimal valid ChatSaveRequest body."""
    return {
        "entries": [base_entry],
        "model": "gemma2:2b",
        "temperature": 0.7,
        "max_tokens": 128,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestSaveChatSuccess:
    """Successful save scenarios."""

    def test_returns_200(self, client: TestClient, base_request: dict) -> None:
        resp = client.post("/api/save_chat", json=base_request)
        assert resp.status_code == 200

    def test_response_has_folder_name(self, client: TestClient, base_request: dict) -> None:
        data = client.post("/api/save_chat", json=base_request).json()
        assert "folder_name" in data
        assert data["folder_name"]  # non-empty

    def test_folder_name_format(self, client: TestClient, base_request: dict) -> None:
        """Folder name matches YYYYMMDD_HHMMSS_<8hex> pattern."""
        import re

        data = client.post("/api/save_chat", json=base_request).json()
        assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", data["folder_name"])

    def test_response_has_files_list(self, client: TestClient, base_request: dict) -> None:
        data = client.post("/api/save_chat", json=base_request).json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_game_log_md_always_written(self, client: TestClient, base_request: dict) -> None:
        data = client.post("/api/save_chat", json=base_request).json()
        assert "game_log.md" in data["files"]

    def test_metadata_json_always_written(self, client: TestClient, base_request: dict) -> None:
        data = client.post("/api/save_chat", json=base_request).json()
        assert "metadata.json" in data["files"]

    def test_response_has_timestamp(self, client: TestClient, base_request: dict) -> None:
        data = client.post("/api/save_chat", json=base_request).json()
        assert "timestamp" in data
        # Should look like an ISO-8601 datetime
        assert "T" in data["timestamp"]


# ---------------------------------------------------------------------------
# Conditional file writing
# ---------------------------------------------------------------------------


class TestConditionalFiles:
    """Conditional file-writing behaviour for payload and prompt files.

    char_a_payload.json / char_b_payload.json are written only when the
    corresponding axes dict is present in the request.

    system_prompt.md is always written: either from the explicit
    ``system_prompt`` field in the request, or by loading the server's
    default IC prompt file (``ic_v01_undertaking.txt``) when no prompt was
    provided.  This ensures the save package always documents the prompt
    that was actually used during translation.
    """

    def test_char_a_payload_written_when_provided(
        self, client: TestClient, base_request: dict, axes_a: dict
    ) -> None:
        req = {**base_request, "character_a": axes_a}
        data = client.post("/api/save_chat", json=req).json()
        assert "char_a_payload.json" in data["files"]

    def test_char_b_payload_written_when_provided(
        self, client: TestClient, base_request: dict, axes_b: dict
    ) -> None:
        req = {**base_request, "character_b": axes_b}
        data = client.post("/api/save_chat", json=req).json()
        assert "char_b_payload.json" in data["files"]

    def test_char_a_payload_absent_when_null(self, client: TestClient, base_request: dict) -> None:
        """When character_a is None (default), char_a_payload.json must not be written."""
        data = client.post("/api/save_chat", json=base_request).json()
        assert "char_a_payload.json" not in data["files"]

    def test_char_b_payload_absent_when_null(self, client: TestClient, base_request: dict) -> None:
        data = client.post("/api/save_chat", json=base_request).json()
        assert "char_b_payload.json" not in data["files"]

    def test_system_prompt_md_written_when_provided(
        self, client: TestClient, base_request: dict
    ) -> None:
        req = {
            **base_request,
            "system_prompt": "Translate the user's OOC message using this profile.",
        }
        data = client.post("/api/save_chat", json=req).json()
        assert "system_prompt.md" in data["files"]

    def test_system_prompt_md_written_from_default_when_no_prompt(
        self, client: TestClient, base_request: dict
    ) -> None:
        """When no system_prompt is sent, system_prompt.md is still written
        by loading the server default IC prompt file (ic_v01_undertaking.txt).
        """
        # base_request has no system_prompt key.
        data = client.post("/api/save_chat", json=base_request).json()
        assert "system_prompt.md" in data["files"]

    def test_prompt_history_files_written_for_distinct_prompts(
        self, client: TestClient, base_entry: dict
    ) -> None:
        """Each distinct prompt used during the chat must be written as its own md file."""
        req = {
            "entries": [
                base_entry,
                {
                    **base_entry,
                    "ch": "b",
                    "system_prompt": "Server prompt B: {{profile_summary}}",
                    "system_prompt_hash": "bbb5aabaa2474094" + "e" * 48,
                    "ipc_id": "b1e28854078de8dd" + "f" * 48,
                    "input_hash": "dd712615b3b3670a" + "1" * 48,
                    "output_hash": "fa0c157dd83dbffb" + "2" * 48,
                },
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 42,
            "system_prompt": "Server prompt B: {{profile_summary}}",
        }
        data = client.post("/api/save_chat", json=req).json()

        assert "system_prompt_001.md" in data["files"]
        assert "system_prompt_002.md" in data["files"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestSaveChatValidation:
    """Pydantic validation errors."""

    def test_empty_entries_returns_422(self, client: TestClient, base_request: dict) -> None:
        req = {**base_request, "entries": []}
        resp = client.post("/api/save_chat", json=req)
        assert resp.status_code == 422

    def test_missing_model_returns_422(self, client: TestClient, base_request: dict) -> None:
        req = {k: v for k, v in base_request.items() if k != "model"}
        resp = client.post("/api/save_chat", json=req)
        assert resp.status_code == 422

    def test_missing_seed_returns_422(self, client: TestClient, base_request: dict) -> None:
        req = {k: v for k, v in base_request.items() if k != "seed"}
        resp = client.post("/api/save_chat", json=req)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Export (zip download) works for chat save folders
# ---------------------------------------------------------------------------


class TestChatSaveExport:
    """The existing /api/save/{folder}/export endpoint serves chat save folders."""

    def test_export_returns_200(self, client: TestClient, base_request: dict) -> None:
        folder = client.post("/api/save_chat", json=base_request).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        assert resp.status_code == 200

    def test_export_content_type_is_zip(self, client: TestClient, base_request: dict) -> None:
        folder = client.post("/api/save_chat", json=base_request).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        assert "zip" in resp.headers["content-type"]

    def test_export_zip_contains_game_log_md(self, client: TestClient, base_request: dict) -> None:
        """The downloaded zip must contain game_log.md."""
        folder = client.post("/api/save_chat", json=base_request).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            assert "game_log.md" in zf.namelist()

    def test_export_zip_contains_metadata_json(
        self, client: TestClient, base_request: dict
    ) -> None:
        folder = client.post("/api/save_chat", json=base_request).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            assert "metadata.json" in zf.namelist()

    def test_metadata_json_content(self, client: TestClient, base_request: dict) -> None:
        """metadata.json must record key session fields."""
        folder = client.post("/api/save_chat", json=base_request).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            meta = json.loads(zf.read("metadata.json"))
        assert meta["model"] == "gemma2:2b"
        assert meta["entry_count"] == 1
        assert meta["seed"] == 42
        assert meta["has_character_a"] is False
        assert meta["has_character_b"] is False
        assert meta["character_a_name"] is None
        assert meta["character_b_name"] is None

    def test_metadata_json_includes_character_payload_names(
        self, client: TestClient, base_request: dict, axes_a: dict, axes_b: dict
    ) -> None:
        """metadata.json must preserve the selected example names for both characters."""
        req = {
            **base_request,
            "character_a": axes_a,
            "character_b": axes_b,
            "character_a_name": "brittle_elite",
            "character_b_name": "arrogant_patron",
        }
        folder = client.post("/api/save_chat", json=req).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            meta = json.loads(zf.read("metadata.json"))

        assert meta["has_character_a"] is True
        assert meta["has_character_b"] is True
        assert meta["character_a_name"] == "brittle_elite"
        assert meta["character_b_name"] == "arrogant_patron"

    def test_export_zip_contains_system_prompt_md(
        self, client: TestClient, base_request: dict
    ) -> None:
        """The downloaded zip must contain system_prompt.md even when no explicit
        prompt was sent — the server loads the default and includes it.
        """
        folder = client.post("/api/save_chat", json=base_request).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            assert "system_prompt.md" in zf.namelist()

    def test_export_zip_contains_prompt_history_md_files(
        self, client: TestClient, base_entry: dict
    ) -> None:
        """Distinct prompts used during the chat must be exported as separate markdown files."""
        req = {
            "entries": [
                base_entry,
                {
                    **base_entry,
                    "ch": "b",
                    "system_prompt": "Server prompt B: {{profile_summary}}",
                    "system_prompt_hash": "bbb5aabaa2474094" + "e" * 48,
                    "ipc_id": "b1e28854078de8dd" + "f" * 48,
                    "input_hash": "dd712615b3b3670a" + "1" * 48,
                    "output_hash": "fa0c157dd83dbffb" + "2" * 48,
                },
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 42,
            "system_prompt": "Server prompt B: {{profile_summary}}",
        }
        folder = client.post("/api/save_chat", json=req).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            assert "system_prompt_001.md" in zf.namelist()
            assert "system_prompt_002.md" in zf.namelist()


# ---------------------------------------------------------------------------
# OOC message column in game_log.md
# ---------------------------------------------------------------------------


class TestOocInGameLog:
    """Verify that ooc_message is included in the saved game_log.md output.

    The OOC column was added between the Char and Channel columns as part of
    the In-Game Output OOC feature.  These tests confirm the end-to-end flow
    from the HTTP request through to the written Markdown table.
    """

    def _read_game_log_md(self, client: TestClient, req: dict) -> str:
        """Helper: POST /api/save_chat, then read game_log.md from the exported zip."""
        folder = client.post("/api/save_chat", json=req).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            return zf.read("game_log.md").decode("utf-8")

    def test_ooc_column_header_present(self, client: TestClient, base_request: dict) -> None:
        """game_log.md must have a five-column header that includes 'OOC'."""
        md = self._read_game_log_md(client, base_request)
        assert "| # | Char | OOC | Channel | IC Text |" in md

    def test_ooc_message_appears_in_data_row(
        self, client: TestClient, base_request: dict, base_entry: dict
    ) -> None:
        """The OOC message from the entry must appear in the table body."""
        md = self._read_game_log_md(client, base_request)
        assert base_entry["ooc_message"] in md

    def test_ooc_between_char_and_channel(
        self, client: TestClient, base_request: dict, base_entry: dict
    ) -> None:
        """OOC must appear after the Char column and before the Channel column
        on the same data row.
        """
        md = self._read_game_log_md(client, base_request)
        # Find the first data row (after the header separator).
        for line in md.splitlines():
            if line.startswith("| 1 |"):
                cols = [c.strip() for c in line.split("|")]
                # cols[0] = '', cols[1] = '#', cols[2] = 'Char',
                # cols[3] = 'OOC', cols[4] = 'Channel', cols[5] = 'IC Text', cols[6] = ''
                assert cols[3] == base_entry["ooc_message"]  # OOC column
                assert cols[4] == base_entry["channel"]  # Channel column
                break
        else:
            pytest.fail("No data row found in game_log.md")

    def test_ooc_absent_renders_empty_cell(self, client: TestClient) -> None:
        """Legacy entries without ooc_message must render with an empty OOC cell."""
        req = {
            "entries": [
                {
                    "ch": "a",
                    "channel": "say",
                    # ooc_message intentionally omitted — backward-compat check.
                    "ic_text": "She peers about.",
                    "model": "gemma2:2b",
                    "ipc_id": None,
                }
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 0,
        }
        md = self._read_game_log_md(client, req)

        # Empty OOC cell means the column value is blank — two adjacent pipes.
        for line in md.splitlines():
            if line.startswith("| 1 |"):
                cols = [c.strip() for c in line.split("|")]
                assert cols[3] == ""  # OOC column should be empty
                break
        else:
            pytest.fail("No data row found in game_log.md")

    def test_pipe_in_ooc_is_escaped(self, client: TestClient) -> None:
        """Pipe characters in ooc_message must be escaped in the Markdown table."""
        req = {
            "entries": [
                {
                    "ch": "a",
                    "channel": "say",
                    "ooc_message": "A|B",
                    "ic_text": "She speaks.",
                    "model": "gemma2:2b",
                    "ipc_id": None,
                }
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 0,
        }
        md = self._read_game_log_md(client, req)
        assert "A\\|B" in md


# ---------------------------------------------------------------------------
# IPC provenance hashes in metadata.json
# ---------------------------------------------------------------------------


class TestHashesInMetadata:
    """Verify that IPC provenance hashes are correctly written to metadata.json.

    Covers:
    - system_prompt_hash is non-null and uses compute_system_prompt_hash (not
      compute_output_hash, which was the original bug).
    - system_prompt_hash is extracted from entries when no prompt override is given.
    - per_entry_hashes array is present with all four hash fields per entry.
    - Legacy entries without hash fields produce None values (not errors).
    """

    # ── Helpers ──────────────────────────────────────────────────────────── #

    def _read_metadata(self, client: TestClient, req: dict) -> dict:
        """POST /api/save_chat then extract metadata.json from the zip."""
        folder = client.post("/api/save_chat", json=req).json()["folder_name"]
        resp = client.get(f"/api/save/{folder}/export")
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            return json.loads(zf.read("metadata.json"))

    # ── system_prompt_hash — from explicit prompt ─────────────────────────── #

    def test_system_prompt_hash_non_null_when_prompt_provided(
        self, client: TestClient, base_request: dict
    ) -> None:
        """When system_prompt is in the request, system_prompt_hash must not be null."""
        req = {
            **base_request,
            "system_prompt": "Translate the user's OOC message using this profile.",
        }
        meta = self._read_metadata(client, req)
        assert meta["system_prompt_hash"] is not None

    def test_system_prompt_hash_uses_correct_function(
        self, client: TestClient, base_request: dict
    ) -> None:
        """system_prompt_hash must match compute_system_prompt_hash (not compute_output_hash).

        The original bug used compute_output_hash() which applies a different
        normalisation rule set.  The two functions diverge on multi-line
        prompts with per-line leading indentation:

        - compute_system_prompt_hash strips whitespace from *each line*
          individually, so indented continuation lines are dedented.
        - compute_output_hash only strips the overall string and collapses
          multi-space runs, leaving per-line leading spaces intact.

        We craft a prompt that exercises this difference so the test can
        distinguish between the two functions.
        """
        from pipeworks_ipc import compute_output_hash, compute_system_prompt_hash

        # Multi-line prompt with per-line leading indentation.
        # compute_system_prompt_hash produces "Line A\nLine B" (strips each line).
        # compute_output_hash produces "Line A\n Line B" (collapses 2 spaces → 1).
        prompt = "Line A\n  Line B"
        req = {**base_request, "system_prompt": prompt}
        meta = self._read_metadata(client, req)

        correct_hash = compute_system_prompt_hash(prompt)
        incorrect_hash = compute_output_hash(prompt)

        # Sanity: the two functions must actually diverge for this test to be meaningful.
        assert correct_hash != incorrect_hash, (
            "Test prompt does not distinguish the two hash functions — "
            "choose a prompt with per-line indentation."
        )
        # Must match the system_prompt normaliser.
        assert meta["system_prompt_hash"] == correct_hash
        # Must NOT match the output normaliser (confirms the bug is fixed).
        assert meta["system_prompt_hash"] != incorrect_hash

    # ── system_prompt_hash — from entries when no override ──────────────── #

    def test_system_prompt_hash_extracted_from_entries_when_no_prompt(
        self, client: TestClient, base_request: dict, base_entry: dict
    ) -> None:
        """When system_prompt is absent, sp_hash must come from the entry's stored hash."""
        # base_request has no system_prompt key; base_entry has system_prompt_hash.
        meta = self._read_metadata(client, base_request)
        assert meta["system_prompt_hash"] == base_entry["system_prompt_hash"]

    def test_system_prompt_hash_null_when_no_prompt_and_legacy_entries(
        self, client: TestClient
    ) -> None:
        """When there's no system_prompt and entries have no stored hashes, result is null."""
        req = {
            "entries": [
                {
                    "ch": "a",
                    "channel": "say",
                    "ic_text": "She glances about.",
                    "model": "gemma2:2b",
                    # No system_prompt_hash — legacy entry.
                }
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 0,
        }
        meta = self._read_metadata(client, req)
        assert meta["system_prompt_hash"] is None

    # ── per_entry_hashes ─────────────────────────────────────────────────── #

    def test_per_entry_hashes_key_present(self, client: TestClient, base_request: dict) -> None:
        """metadata.json must contain a 'per_entry_hashes' list."""
        meta = self._read_metadata(client, base_request)
        assert "per_entry_hashes" in meta
        assert isinstance(meta["per_entry_hashes"], list)

    def test_per_entry_hashes_length_matches_entry_count(
        self, client: TestClient, base_request: dict
    ) -> None:
        """per_entry_hashes must have one entry per log entry."""
        meta = self._read_metadata(client, base_request)
        assert len(meta["per_entry_hashes"]) == meta["entry_count"]

    def test_per_entry_hashes_contain_all_prompt_and_hash_fields(
        self, client: TestClient, base_request: dict, base_entry: dict
    ) -> None:
        """Each per_entry_hashes row must carry index, ch, prompt text, and all hashes."""
        meta = self._read_metadata(client, base_request)
        row = meta["per_entry_hashes"][0]

        assert row["index"] == 1
        assert row["ch"] == base_entry["ch"]
        assert row["input_hash"] == base_entry["input_hash"]
        assert row["system_prompt_hash"] == base_entry["system_prompt_hash"]
        assert row["system_prompt"] == base_entry["system_prompt"]
        assert row["output_hash"] == base_entry["output_hash"]
        assert row["ipc_id"] == base_entry["ipc_id"]

    def test_system_prompt_history_contains_unique_prompt_entries(
        self, client: TestClient, base_entry: dict
    ) -> None:
        """metadata.json must preserve every distinct prompt used during the chat log."""
        req = {
            "entries": [
                base_entry,
                {
                    **base_entry,
                    "ch": "b",
                    "ooc_message": "he narrows his eyes",
                    "ic_text": "He narrows his eyes.",
                    "system_prompt": "Server prompt B: {{profile_summary}}",
                    "system_prompt_hash": "bbb5aabaa2474094" + "e" * 48,
                    "ipc_id": "b1e28854078de8dd" + "f" * 48,
                    "input_hash": "dd712615b3b3670a" + "1" * 48,
                    "output_hash": "fa0c157dd83dbffb" + "2" * 48,
                },
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 42,
            "system_prompt": "Server prompt B: {{profile_summary}}",
        }
        meta = self._read_metadata(client, req)

        history = meta["system_prompt_history"]
        assert len(history) == 2
        assert {row["system_prompt"] for row in history} == {
            "Server prompt A: {{profile_summary}}",
            "Server prompt B: {{profile_summary}}",
        }
        assert {row["filename"] for row in history} == {
            "system_prompt_001.md",
            "system_prompt_002.md",
        }
        assert sorted(history[0]["entry_indices"] + history[1]["entry_indices"]) == [1, 2]

    def test_per_entry_hashes_multiple_entries_indexed_correctly(self, client: TestClient) -> None:
        """Row indices must be 1-based and consecutive across all entries."""
        req = {
            "entries": [
                {
                    "ch": "a",
                    "channel": "say",
                    "ic_text": "She enters.",
                    "system_prompt": "Prompt one",
                    "model": "gemma2:2b",
                    "ipc_id": "aaa" + "0" * 61,
                    "input_hash": "bbb" + "0" * 61,
                    "system_prompt_hash": "ccc" + "0" * 61,
                    "output_hash": "ddd" + "0" * 61,
                },
                {
                    "ch": "b",
                    "channel": "whisper",
                    "ic_text": "He nods.",
                    "system_prompt": "Prompt two",
                    "model": "gemma2:2b",
                    "ipc_id": "eee" + "0" * 61,
                    "input_hash": "fff" + "0" * 61,
                    "system_prompt_hash": "ggg" + "0" * 61,
                    "output_hash": "hhh" + "0" * 61,
                },
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 0,
        }
        meta = self._read_metadata(client, req)
        rows = meta["per_entry_hashes"]
        assert rows[0]["index"] == 1
        assert rows[0]["ch"] == "a"
        assert rows[1]["index"] == 2
        assert rows[1]["ch"] == "b"

    def test_legacy_entries_without_hashes_produce_none_not_error(self, client: TestClient) -> None:
        """Entries without hash fields must produce None values in per_entry_hashes
        without raising a server error.
        """
        req = {
            "entries": [
                {
                    "ch": "a",
                    "channel": "say",
                    "ic_text": "She looks about.",
                    "model": "gemma2:2b",
                    # No hash fields at all — backward compat check.
                }
            ],
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 0,
        }
        meta = self._read_metadata(client, req)
        row = meta["per_entry_hashes"][0]
        assert row["input_hash"] is None
        assert row["system_prompt_hash"] is None
        assert row["system_prompt"] is None
        assert row["output_hash"] is None
        assert row["ipc_id"] is None

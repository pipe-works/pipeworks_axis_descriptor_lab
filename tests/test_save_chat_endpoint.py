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
    """Minimal single log entry."""
    return {
        "ch": "a",
        "channel": "say",
        "ic_text": "She peers cautiously about the chamber.",
        "model": "gemma2:2b",
        "ipc_id": None,
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
    """char_a_payload.json / char_b_payload.json / system_prompt.md are optional."""

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
        req = {**base_request, "system_prompt": "Translate: {{ooc_message}}"}
        data = client.post("/api/save_chat", json=req).json()
        assert "system_prompt.md" in data["files"]

    def test_system_prompt_md_absent_when_null(
        self, client: TestClient, base_request: dict
    ) -> None:
        data = client.post("/api/save_chat", json=base_request).json()
        assert "system_prompt.md" not in data["files"]


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

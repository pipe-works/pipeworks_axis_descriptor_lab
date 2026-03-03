"""Tests for the Artifact Editor backend routes and prompt draft safety rules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.mud_server_client import MudServerConnectionError, MudServerSessionExpiredError


class TestLocalPromptArtifacts:
    """Local prompt-artifact listing, loading, and draft creation."""

    def test_lists_local_prompt_artifacts(self, client: TestClient) -> None:
        resp = client.get("/api/artifacts/local/chat-prompts?purpose=chat_translation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["purpose"] == "chat_translation"
        assert any(prompt["name"] == "pipeworks_web_ic_prompt" for prompt in data["prompts"])
        assert any(
            row["placeholder"] == "{{profile_summary}}" for row in data["reference"]["placeholders"]
        )

    def test_loads_local_prompt_document(self, client: TestClient) -> None:
        resp = client.get(
            "/api/artifacts/local/chat-prompts/pipeworks_web_ic_prompt?purpose=chat_translation"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pipeworks_web_ic_prompt"
        assert data["purpose"] == "chat_translation"
        assert "TRANSLATION RULES" in data["content"]
        assert data["origin_path"] == "pipeworks_web_ic_prompt.txt"

    def test_creates_local_prompt_draft_in_drafts_directory(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        prompt_root = tmp_path / "prompts"
        (prompt_root / "chat_translation").mkdir(parents=True)
        (prompt_root / "chat_translation" / "pipeworks_web_ic_prompt.txt").write_text(
            "base prompt",
            encoding="utf-8",
        )
        (prompt_root / "character_description").mkdir(parents=True)

        with (
            patch("app.artifact_editor.PROMPTS_DIR", prompt_root),
            patch("app.file_loaders.PROMPTS_DIR", prompt_root),
        ):
            resp = client.post(
                "/api/artifacts/local/chat-prompts/drafts",
                json={
                    "purpose": "chat_translation",
                    "draft_name": "new_prompt_draft",
                    "content": "draft prompt text",
                    "based_on_name": "pipeworks_web_ic_prompt",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "new_prompt_draft"
        assert data["origin_path"] == "drafts/new_prompt_draft.txt"
        assert (prompt_root / "chat_translation" / "drafts" / "new_prompt_draft.txt").read_text(
            encoding="utf-8"
        ) == "draft prompt text\n"

    def test_rejects_draft_name_collision(self, client: TestClient, tmp_path: Path) -> None:
        prompt_root = tmp_path / "prompts"
        (prompt_root / "chat_translation").mkdir(parents=True)
        (prompt_root / "chat_translation" / "pipeworks_web_ic_prompt.txt").write_text(
            "base prompt",
            encoding="utf-8",
        )
        (prompt_root / "character_description").mkdir(parents=True)

        with (
            patch("app.artifact_editor.PROMPTS_DIR", prompt_root),
            patch("app.file_loaders.PROMPTS_DIR", prompt_root),
        ):
            resp = client.post(
                "/api/artifacts/local/chat-prompts/drafts",
                json={
                    "purpose": "chat_translation",
                    "draft_name": "pipeworks_web_ic_prompt",
                    "content": "draft prompt text",
                },
            )

        assert resp.status_code == 409


class TestLocalAxisPayloadArtifacts:
    """Local AxisPayload JSON artifact listing, loading, and draft creation."""

    def test_lists_local_axis_payload_artifacts(self, client: TestClient) -> None:
        resp = client.get("/api/artifacts/local/axis-payloads")
        assert resp.status_code == 200
        data = resp.json()
        assert any(payload["name"] == "proud_operator" for payload in data["payloads"])
        assert any(field["name"] == "axes" for field in data["reference"]["fields"])

    def test_loads_local_axis_payload_document(self, client: TestClient) -> None:
        resp = client.get("/api/artifacts/local/axis-payloads/proud_operator")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "proud_operator"
        assert data["world_id"] == "pipeworks_web"
        assert '"axes"' in data["content"]
        assert data["origin_path"] == "proud_operator.json"

    def test_creates_local_axis_payload_draft_in_drafts_directory(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        examples_root = tmp_path / "examples"
        examples_root.mkdir(parents=True)
        (examples_root / "proud_operator.json").write_text(
            '{"axes":{"demeanor":{"label":"proud","score":0.8}},"policy_hash":"abc","seed":1,"world_id":"pipeworks_web"}',
            encoding="utf-8",
        )

        with patch("app.artifact_editor.EXAMPLES_DIR", examples_root):
            resp = client.post(
                "/api/artifacts/local/axis-payloads/drafts",
                json={
                    "draft_name": "new_axis_payload",
                    "content": '{"axes":{"health":{"label":"weary","score":0.3}},"policy_hash":"xyz","seed":7,"world_id":"pipeworks_web"}',
                    "based_on_name": "proud_operator",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "new_axis_payload"
        assert data["origin_path"] == "drafts/new_axis_payload.json"
        assert data["world_id"] == "pipeworks_web"
        assert (examples_root / "drafts" / "new_axis_payload.json").exists()

    def test_rejects_invalid_axis_payload_json(self, client: TestClient, tmp_path: Path) -> None:
        examples_root = tmp_path / "examples"
        examples_root.mkdir(parents=True)

        with patch("app.artifact_editor.EXAMPLES_DIR", examples_root):
            resp = client.post(
                "/api/artifacts/local/axis-payloads/drafts",
                json={
                    "draft_name": "bad_axis_payload",
                    "content": '{"axes":}',
                },
            )

        assert resp.status_code == 400

    def test_rejects_axis_payload_name_collision(self, client: TestClient, tmp_path: Path) -> None:
        examples_root = tmp_path / "examples"
        examples_root.mkdir(parents=True)
        (examples_root / "proud_operator.json").write_text(
            '{"axes":{"demeanor":{"label":"proud","score":0.8}},"policy_hash":"abc","seed":1,"world_id":"pipeworks_web"}',
            encoding="utf-8",
        )

        with patch("app.artifact_editor.EXAMPLES_DIR", examples_root):
            resp = client.post(
                "/api/artifacts/local/axis-payloads/drafts",
                json={
                    "draft_name": "proud_operator",
                    "content": '{"axes":{"health":{"label":"weary","score":0.3}},"policy_hash":"xyz","seed":7,"world_id":"pipeworks_web"}',
                },
            )

        assert resp.status_code == 409


class TestServerPromptManifest:
    """Server-backed prompt-manifest normalization."""

    def test_returns_server_prompt_manifest(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.world_config.return_value = {
            "world_id": "pipeworks_web",
            "name": "Pipeworks Web",
            "active_axes": ["demeanor", "health"],
        }
        mock.world_prompts.return_value = {
            "world_id": "pipeworks_web",
            "prompts": [
                {
                    "filename": "ic_prompt.txt",
                    "content": "Prompt {{profile_summary}} {{channel}}",
                    "is_active": True,
                }
            ],
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"
        assert data["world_name"] == "Pipeworks Web"
        assert data["active_prompt_name"] == "ic_prompt"
        assert data["prompts"][0]["is_active"] is True
        assert any(
            row["placeholder"] == "{{demeanor_label}}" for row in data["reference"]["placeholders"]
        )

    def test_server_manifest_requires_authentication(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = False

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web")

        assert resp.status_code == 401

    def test_server_manifest_requires_configured_mud_client(self, client: TestClient) -> None:
        with patch("app.routes_artifact_editor.get_mud_client", return_value=None):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web")

        assert resp.status_code == 503

    def test_server_manifest_handles_expired_session(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.get_server_prompt_manifest",
                side_effect=MudServerSessionExpiredError("expired"),
            ),
        ):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web")

        assert resp.status_code == 401
        assert "session expired" in resp.json()["detail"].lower()

    def test_server_manifest_handles_connection_error(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.get_server_prompt_manifest",
                side_effect=MudServerConnectionError("down"),
            ),
        ):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web")

        assert resp.status_code == 502

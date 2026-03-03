"""Tests for the Artifact Editor backend routes and prompt draft safety rules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


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

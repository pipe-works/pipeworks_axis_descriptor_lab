"""Tests for the Artifact Editor backend routes and prompt draft safety rules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
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


class TestLocalLexiconArtifacts:
    """Local deterministic lexicon JSON artifact listing, loading, and draft creation."""

    def test_lists_local_lexicon_artifacts(self, client: TestClient) -> None:
        resp = client.get("/api/artifacts/local/lexicons")
        assert resp.status_code == 200
        data = resp.json()
        assert any(artifact["name"] == "abstraction_v0_1" for artifact in data["lexicons"])
        assert data["reference"]["artifact_kind"] == "catalog"

    def test_loads_local_lexicon_document(self, client: TestClient) -> None:
        resp = client.get("/api/artifacts/local/lexicons/intensity_v0_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "intensity_v0_1"
        assert data["artifact_kind"] == "intensity"
        assert data["version"] == "0.1"
        assert '"scales"' in data["content"]
        assert data["origin_path"] == "intensity_v0_1.json"

    def test_creates_local_lexicon_draft_in_drafts_directory(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        data_root = tmp_path / "data"
        data_root.mkdir(parents=True)
        (data_root / "embodiment_v0_1.json").write_text(
            '{"version":"0.1","abstract":["tension"],"physical":["hand"]}',
            encoding="utf-8",
        )

        with patch("app.artifact_editor.DATA_DIR", data_root):
            resp = client.post(
                "/api/artifacts/local/lexicons/drafts",
                json={
                    "draft_name": "embodiment_alt_v0_1",
                    "content": '{"version":"0.2","abstract":["fear"],"physical":["posture"]}',
                    "based_on_name": "embodiment_v0_1",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "embodiment_alt_v0_1"
        assert data["artifact_kind"] == "embodiment"
        assert data["origin_path"] == "drafts/embodiment_alt_v0_1.json"
        assert data["version"] == "0.2"
        assert (data_root / "drafts" / "embodiment_alt_v0_1.json").exists()

    def test_rejects_invalid_lexicon_json(self, client: TestClient, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        data_root.mkdir(parents=True)

        with patch("app.artifact_editor.DATA_DIR", data_root):
            resp = client.post(
                "/api/artifacts/local/lexicons/drafts",
                json={
                    "draft_name": "bad_lexicon",
                    "content": '{"version": }',
                },
            )

        assert resp.status_code == 400
        assert "invalid json" in resp.json()["detail"].lower()

    def test_rejects_unknown_lexicon_contract(self, client: TestClient, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        data_root.mkdir(parents=True)

        with patch("app.artifact_editor.DATA_DIR", data_root):
            resp = client.post(
                "/api/artifacts/local/lexicons/drafts",
                json={
                    "draft_name": "bad_shape",
                    "content": '{"version":"0.1","words":["a","b"]}',
                },
            )

        assert resp.status_code == 400
        assert "supported lexicon contract" in resp.json()["detail"].lower()

    def test_rejects_lexicon_name_collision(self, client: TestClient, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        data_root.mkdir(parents=True)
        (data_root / "abstraction_v0_1.json").write_text(
            '{"version":"0.1","abstract_terms":["authority"],"concrete_terms":["coat"]}',
            encoding="utf-8",
        )

        with patch("app.artifact_editor.DATA_DIR", data_root):
            resp = client.post(
                "/api/artifacts/local/lexicons/drafts",
                json={
                    "draft_name": "abstraction_v0_1",
                    "content": '{"version":"0.2","abstract_terms":["risk"],"concrete_terms":["boots"]}',
                },
            )

        assert resp.status_code == 409


class TestLocalPolicyBundleArtifacts:
    """Local normalized policy bundle JSON artifact listing, loading, and draft creation."""

    def test_lists_local_policy_bundle_artifacts(self, client: TestClient) -> None:
        resp = client.get("/api/artifacts/local/policy-bundles")
        assert resp.status_code == 200
        data = resp.json()
        assert any(
            bundle["name"] == "pipeworks_web_policy_bundle_v0_1" for bundle in data["bundles"]
        )
        assert any(field["name"] == "chat_rules" for field in data["reference"]["fields"])

    def test_loads_local_policy_bundle_document(self, client: TestClient) -> None:
        resp = client.get("/api/artifacts/local/policy-bundles/pipeworks_web_policy_bundle_v0_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pipeworks_web_policy_bundle_v0_1"
        assert data["world_id"] == "pipeworks_web"
        assert data["version"] == "0.1.0"
        assert '"chat_rules"' in data["content"]
        assert data["origin_path"] == "pipeworks_web_policy_bundle_v0_1.json"

    def test_creates_local_policy_bundle_draft_in_drafts_directory(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        bundle_root = tmp_path / "policy_bundles"
        bundle_root.mkdir(parents=True)
        (bundle_root / "pipeworks_web_policy_bundle_v0_1.json").write_text(
            '{"world_id":"pipeworks_web","version":"0.1.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.01}}}}',
            encoding="utf-8",
        )

        with patch("app.artifact_editor.POLICY_BUNDLES_DIR", bundle_root):
            resp = client.post(
                "/api/artifacts/local/policy-bundles/drafts",
                json={
                    "draft_name": "pipeworks_web_policy_bundle_alt",
                    "content": '{"world_id":"pipeworks_web","version":"0.2.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.02}}}}',
                    "based_on_name": "pipeworks_web_policy_bundle_v0_1",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pipeworks_web_policy_bundle_alt"
        assert data["origin_path"] == "drafts/pipeworks_web_policy_bundle_alt.json"
        assert data["world_id"] == "pipeworks_web"
        assert data["version"] == "0.2.0"
        assert (bundle_root / "drafts" / "pipeworks_web_policy_bundle_alt.json").exists()

    def test_rejects_invalid_policy_bundle_contract(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        bundle_root = tmp_path / "policy_bundles"
        bundle_root.mkdir(parents=True)

        with patch("app.artifact_editor.POLICY_BUNDLES_DIR", bundle_root):
            resp = client.post(
                "/api/artifacts/local/policy-bundles/drafts",
                json={
                    "draft_name": "bad_policy_bundle",
                    "content": '{"world_id":"pipeworks_web","version":"0.1.0","source":"test","axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"scarred","min":0.6,"max":0.79}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.01}}}}',
                },
            )

        assert resp.status_code == 400
        assert "threshold labels must match ordering exactly" in resp.json()["detail"]

    def test_rejects_policy_bundle_name_collision(self, client: TestClient, tmp_path: Path) -> None:
        bundle_root = tmp_path / "policy_bundles"
        bundle_root.mkdir(parents=True)
        (bundle_root / "pipeworks_web_policy_bundle_v0_1.json").write_text(
            '{"world_id":"pipeworks_web","version":"0.1.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.01}}}}',
            encoding="utf-8",
        )

        with patch("app.artifact_editor.POLICY_BUNDLES_DIR", bundle_root):
            resp = client.post(
                "/api/artifacts/local/policy-bundles/drafts",
                json={
                    "draft_name": "pipeworks_web_policy_bundle_v0_1",
                    "content": '{"world_id":"pipeworks_web","version":"0.2.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.02}}}}',
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


class TestServerPromptDraftArtifacts:
    """Server-backed prompt draft normalization and proxy error handling."""

    def test_lists_server_prompt_drafts(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.world_config.return_value = {
            "world_id": "pipeworks_web",
            "name": "Pipeworks Web",
            "active_axes": ["demeanor", "health"],
        }
        mock.world_prompt_drafts.return_value = {
            "world_id": "pipeworks_web",
            "drafts": [
                {
                    "name": "ic_prompt_variant",
                    "origin_path": "policies/drafts/ic_prompt_variant.txt",
                    "world_id": "pipeworks_web",
                }
            ],
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web/drafts")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"
        assert data["prompts"][0]["name"] == "ic_prompt_variant"
        assert data["prompts"][0]["is_draft"] is True
        assert any(
            row["placeholder"] == "{{health_label}}" for row in data["reference"]["placeholders"]
        )

    def test_loads_server_prompt_draft_document(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.world_config.return_value = {
            "world_id": "pipeworks_web",
            "name": "Pipeworks Web",
            "active_axes": ["demeanor"],
        }
        mock.world_prompt_draft.return_value = {
            "name": "ic_prompt_variant",
            "origin_path": "policies/drafts/ic_prompt_variant.txt",
            "world_id": "pipeworks_web",
            "content": "Prompt {{profile_summary}}\nDelivery Mode: {{channel}}\n",
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ic_prompt_variant"
        assert data["is_draft"] is True
        assert data["purpose"] == "chat_translation"
        assert "Delivery Mode: {{channel}}" in data["content"]

    def test_creates_server_prompt_draft(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.create_world_prompt_draft.return_value = {
            "name": "ic_prompt_variant",
            "origin_path": "policies/drafts/ic_prompt_variant.txt",
            "world_id": "pipeworks_web",
            "based_on_name": "ic_prompt",
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts",
                json={
                    "draft_name": "ic_prompt_variant",
                    "content": "Prompt {{profile_summary}}\nDelivery Mode: {{channel}}",
                    "based_on_name": "ic_prompt",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ic_prompt_variant"
        mock.create_world_prompt_draft.assert_called_once_with(
            world_id="pipeworks_web",
            draft_name="ic_prompt_variant",
            content="Prompt {{profile_summary}}\nDelivery Mode: {{channel}}\n",
            based_on_name="ic_prompt",
        )

    def test_promotes_server_prompt_draft(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.promote_world_prompt_draft.return_value = {
            "name": "ic_prompt_variant",
            "world_id": "pipeworks_web",
            "canonical_name": "ic_prompt_v2",
            "canonical_path": "policies/ic_prompt_v2.txt",
            "active_prompt_path": "policies/ic_prompt_v2.txt",
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant/promote",
                json={"target_name": "ic_prompt_v2"},
            )

        assert resp.status_code == 200
        assert resp.json()["canonical_name"] == "ic_prompt_v2"
        mock.promote_world_prompt_draft.assert_called_once_with(
            world_id="pipeworks_web",
            draft_name="ic_prompt_variant",
            target_name="ic_prompt_v2",
        )

    def test_server_prompt_drafts_require_configured_mud_client(self, client: TestClient) -> None:
        with patch("app.routes_artifact_editor.get_mud_client", return_value=None):
            list_resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web/drafts")
            load_resp = client.get(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant"
            )
            create_resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts",
                json={"draft_name": "ic_prompt_variant", "content": "Prompt"},
            )
            promote_resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant/promote",
                json={"target_name": "ic_prompt_v2"},
            )

        assert list_resp.status_code == 503
        assert load_resp.status_code == 503
        assert create_resp.status_code == 503
        assert promote_resp.status_code == 503

    def test_server_prompt_drafts_require_authentication(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = False

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            list_resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web/drafts")
            load_resp = client.get(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant"
            )
            create_resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts",
                json={"draft_name": "ic_prompt_variant", "content": "Prompt"},
            )
            promote_resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant/promote",
                json={"target_name": "ic_prompt_v2"},
            )

        assert list_resp.status_code == 401
        assert load_resp.status_code == 401
        assert create_resp.status_code == 401
        assert promote_resp.status_code == 401

    def test_server_prompt_drafts_handle_expired_session(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.list_server_prompt_artifacts",
                side_effect=MudServerSessionExpiredError("expired"),
            ),
        ):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web/drafts")

        assert resp.status_code == 401
        assert "session expired" in resp.json()["detail"].lower()

    def test_server_prompt_draft_load_handles_connection_error(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.load_server_prompt_draft_artifact",
                side_effect=MudServerConnectionError("down"),
            ),
        ):
            resp = client.get(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant"
            )

        assert resp.status_code == 502

    def test_server_prompt_draft_create_propagates_mud_http_errors(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        request = httpx.Request(
            "POST",
            "http://example.test/api/lab/world-prompts/pipeworks_web/drafts",
        )
        response = httpx.Response(
            status_code=409,
            request=request,
            json={"detail": "A prompt draft named 'ic_prompt_variant' already exists."},
        )

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.create_server_prompt_draft",
                side_effect=httpx.HTTPStatusError("exists", request=request, response=response),
            ),
        ):
            resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts",
                json={"draft_name": "ic_prompt_variant", "content": "Prompt"},
            )

        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_server_prompt_draft_promote_handles_expired_session(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.promote_server_prompt_draft",
                side_effect=MudServerSessionExpiredError("expired"),
            ),
        ):
            resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant/promote",
                json={"target_name": "ic_prompt_v2"},
            )

        assert resp.status_code == 401
        assert "session expired" in resp.json()["detail"].lower()

    def test_server_prompt_draft_promote_propagates_mud_http_errors(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        request = httpx.Request(
            "POST",
            "http://example.test/api/lab/world-prompts/pipeworks_web/drafts/ic_prompt_variant/promote",
        )
        response = httpx.Response(
            status_code=409,
            request=request,
            json={"detail": "A canonical prompt named 'ic_prompt_v2' already exists."},
        )

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.promote_server_prompt_draft",
                side_effect=httpx.HTTPStatusError("exists", request=request, response=response),
            ),
        ):
            resp = client.post(
                "/api/artifacts/server/chat-prompts/pipeworks_web/drafts/ic_prompt_variant/promote",
                json={"target_name": "ic_prompt_v2"},
            )

        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]


class TestServerPolicyBundleArtifact:
    """Server-backed policy bundle normalization and proxy error handling."""

    def test_returns_server_policy_bundle_artifact(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.world_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "version": "0.1.0",
            "source": "mud_server policy package normalized to JSON",
            "policy_hash": "abc123",
            "source_files": [
                "policies/axes.yaml",
                "policies/thresholds.yaml",
                "policies/resolution.yaml",
            ],
            "axes_order": ["demeanor"],
            "axes": {
                "demeanor": {
                    "group": "character",
                    "ordering": ["timid", "proud"],
                    "thresholds": [
                        {"label": "timid", "min": 0.0, "max": 0.49},
                        {"label": "proud", "min": 0.5, "max": 1.0},
                    ],
                }
            },
            "chat_rules": {
                "channel_multipliers": {"say": 1.0, "yell": 1.5, "whisper": 0.5},
                "min_gap_threshold": 0.05,
                "axes": {"demeanor": {"resolver": "dominance_shift", "base_magnitude": 0.03}},
            },
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pipeworks_web_policy_bundle"
        assert data["world_id"] == "pipeworks_web"
        assert data["version"] == "0.1.0"
        assert "policies/axes.yaml" in data["origin_path"]
        assert '"chat_rules"' in data["content"]

    def test_server_policy_bundle_requires_configured_mud_client(self, client: TestClient) -> None:
        with patch("app.routes_artifact_editor.get_mud_client", return_value=None):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web")

        assert resp.status_code == 503

    def test_server_policy_bundle_requires_authentication(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = False

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web")

        assert resp.status_code == 401

    def test_server_prompt_manifest_propagates_mud_http_errors(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        request = httpx.Request("GET", "http://example.test/api/lab/world-prompts/pipeworks_web")
        response = httpx.Response(
            status_code=404,
            request=request,
            json={"detail": "World 'pipeworks_web' not found."},
        )

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.get_server_prompt_manifest",
                side_effect=httpx.HTTPStatusError("missing", request=request, response=response),
            ),
        ):
            resp = client.get("/api/artifacts/server/chat-prompts/pipeworks_web")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "World 'pipeworks_web' not found."

    def test_server_policy_bundle_handles_expired_session(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.get_server_policy_bundle_artifact",
                side_effect=MudServerSessionExpiredError("expired"),
            ),
        ):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web")

        assert resp.status_code == 401
        assert "session expired" in resp.json()["detail"].lower()

    def test_server_policy_bundle_handles_connection_error(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.get_server_policy_bundle_artifact",
                side_effect=MudServerConnectionError("down"),
            ),
        ):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web")

        assert resp.status_code == 502

    def test_server_policy_bundle_propagates_mud_http_errors(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        request = httpx.Request(
            "GET", "http://example.test/api/lab/world-policy-bundle/pipeworks_web"
        )
        response = httpx.Response(
            status_code=404,
            request=request,
            json={"detail": "Axis policy files unavailable for world 'pipeworks_web'."},
        )

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.get_server_policy_bundle_artifact",
                side_effect=httpx.HTTPStatusError("missing", request=request, response=response),
            ),
        ):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web")

        assert resp.status_code == 404
        assert "axis policy files unavailable" in resp.json()["detail"].lower()

    def test_creates_server_policy_bundle_draft(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.create_world_policy_bundle_draft.return_value = {
            "name": "pipeworks_web_bundle_alt",
            "origin_path": "policies/drafts/pipeworks_web_bundle_alt.json",
            "world_id": "pipeworks_web",
            "version": "0.2.0",
            "based_on_name": "pipeworks_web_policy_bundle",
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.post(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts",
                json={
                    "draft_name": "pipeworks_web_bundle_alt",
                    "content": '{"world_id":"pipeworks_web","version":"0.2.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.02}}}}',
                    "based_on_name": "pipeworks_web_policy_bundle",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pipeworks_web_bundle_alt"
        assert data["origin_path"] == "policies/drafts/pipeworks_web_bundle_alt.json"

    def test_lists_server_policy_bundle_drafts(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.world_policy_bundle_drafts.return_value = {
            "world_id": "pipeworks_web",
            "drafts": [
                {
                    "name": "pipeworks_web_bundle_alt",
                    "origin_path": "policies/drafts/pipeworks_web_bundle_alt.json",
                    "world_id": "pipeworks_web",
                    "version": "0.2.0",
                }
            ],
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web/drafts")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"
        assert data["bundles"][0]["name"] == "pipeworks_web_bundle_alt"
        assert data["bundles"][0]["is_draft"] is True

    def test_server_policy_bundle_draft_list_requires_configured_mud_client(
        self, client: TestClient
    ) -> None:
        with patch("app.routes_artifact_editor.get_mud_client", return_value=None):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web/drafts")

        assert resp.status_code == 503

    def test_server_policy_bundle_draft_list_requires_authentication(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = False

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web/drafts")

        assert resp.status_code == 401

    def test_server_policy_bundle_draft_list_handles_expired_session(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.list_server_policy_bundle_artifacts",
                side_effect=MudServerSessionExpiredError("expired"),
            ),
        ):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web/drafts")

        assert resp.status_code == 401

    def test_server_policy_bundle_draft_list_handles_connection_error(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.list_server_policy_bundle_artifacts",
                side_effect=MudServerConnectionError("down"),
            ),
        ):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web/drafts")

        assert resp.status_code == 502

    def test_server_policy_bundle_draft_list_propagates_mud_http_errors(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        request = httpx.Request(
            "GET", "http://example.test/api/lab/world-policy-bundle/pipeworks_web/drafts"
        )
        response = httpx.Response(
            status_code=404,
            request=request,
            json={"detail": "World 'pipeworks_web' not found."},
        )

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.list_server_policy_bundle_artifacts",
                side_effect=httpx.HTTPStatusError("missing", request=request, response=response),
            ),
        ):
            resp = client.get("/api/artifacts/server/policy-bundles/pipeworks_web/drafts")

        assert resp.status_code == 404

    def test_loads_server_policy_bundle_draft_document(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        mock.world_policy_bundle_draft.return_value = {
            "name": "pipeworks_web_bundle_alt",
            "origin_path": "policies/drafts/pipeworks_web_bundle_alt.json",
            "world_id": "pipeworks_web",
            "version": "0.2.0",
            "content": {
                "world_id": "pipeworks_web",
                "version": "0.2.0",
                "source": "test",
                "policy_hash": None,
                "axes_order": ["health"],
                "axes": {
                    "health": {
                        "group": "character",
                        "ordering": ["weary"],
                        "thresholds": [{"label": "weary", "min": 0.4, "max": 0.59}],
                    }
                },
                "chat_rules": {
                    "channel_multipliers": {"say": 1.0, "yell": 1.5, "whisper": 0.5},
                    "min_gap_threshold": 0.05,
                    "axes": {"health": {"resolver": "shared_drain", "base_magnitude": 0.02}},
                },
            },
        }

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts/pipeworks_web_bundle_alt"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pipeworks_web_bundle_alt"
        assert data["is_draft"] is True
        assert data["origin_path"] == "policies/drafts/pipeworks_web_bundle_alt.json"

    def test_server_policy_bundle_draft_load_requires_configured_mud_client(
        self, client: TestClient
    ) -> None:
        with patch("app.routes_artifact_editor.get_mud_client", return_value=None):
            resp = client.get(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts/pipeworks_web_bundle_alt"
            )

        assert resp.status_code == 503

    def test_server_policy_bundle_draft_load_requires_authentication(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = False

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.get(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts/pipeworks_web_bundle_alt"
            )

        assert resp.status_code == 401

    def test_server_policy_bundle_draft_load_handles_expired_session(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.load_server_policy_bundle_draft_artifact",
                side_effect=MudServerSessionExpiredError("expired"),
            ),
        ):
            resp = client.get(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts/pipeworks_web_bundle_alt"
            )

        assert resp.status_code == 401

    def test_server_policy_bundle_draft_load_handles_connection_error(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.load_server_policy_bundle_draft_artifact",
                side_effect=MudServerConnectionError("down"),
            ),
        ):
            resp = client.get(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts/pipeworks_web_bundle_alt"
            )

        assert resp.status_code == 502

    def test_server_policy_bundle_draft_load_propagates_mud_http_errors(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        request = httpx.Request(
            "GET",
            "http://example.test/api/lab/world-policy-bundle/pipeworks_web/drafts/pipeworks_web_bundle_alt",
        )
        response = httpx.Response(
            status_code=404,
            request=request,
            json={"detail": "Policy bundle draft 'pipeworks_web_bundle_alt' not found."},
        )

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.load_server_policy_bundle_draft_artifact",
                side_effect=httpx.HTTPStatusError("missing", request=request, response=response),
            ),
        ):
            resp = client.get(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts/pipeworks_web_bundle_alt"
            )

        assert resp.status_code == 404

    def test_server_policy_bundle_draft_requires_world_match(self, client: TestClient) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.post(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts",
                json={
                    "draft_name": "pipeworks_web_bundle_alt",
                    "content": '{"world_id":"daily_undertaking","version":"0.2.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.02}}}}',
                },
            )

        assert resp.status_code == 400
        assert "must match the selected mud-server world" in resp.json()["detail"]

    def test_server_policy_bundle_draft_propagates_mud_http_errors(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True
        request = httpx.Request(
            "POST",
            "http://example.test/api/lab/world-policy-bundle/pipeworks_web/drafts",
        )
        response = httpx.Response(
            status_code=409,
            request=request,
            json={"detail": "Draft file already exists."},
        )
        mock.create_world_policy_bundle_draft.side_effect = httpx.HTTPStatusError(
            "conflict",
            request=request,
            response=response,
        )

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.post(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts",
                json={
                    "draft_name": "pipeworks_web_bundle_alt",
                    "content": '{"world_id":"pipeworks_web","version":"0.2.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.02}}}}',
                },
            )

        assert resp.status_code == 409
        assert resp.json()["detail"] == "Draft file already exists."

    def test_server_policy_bundle_draft_create_requires_configured_mud_client(
        self, client: TestClient
    ) -> None:
        with patch("app.routes_artifact_editor.get_mud_client", return_value=None):
            resp = client.post(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts",
                json={"draft_name": "pipeworks_web_bundle_alt", "content": "{}"},
            )

        assert resp.status_code == 503

    def test_server_policy_bundle_draft_create_requires_authentication(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = False

        with patch("app.routes_artifact_editor.get_mud_client", return_value=mock):
            resp = client.post(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts",
                json={"draft_name": "pipeworks_web_bundle_alt", "content": "{}"},
            )

        assert resp.status_code == 401

    def test_server_policy_bundle_draft_create_handles_expired_session(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.create_server_policy_bundle_draft",
                side_effect=MudServerSessionExpiredError("expired"),
            ),
        ):
            resp = client.post(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts",
                json={
                    "draft_name": "pipeworks_web_bundle_alt",
                    "content": '{"world_id":"pipeworks_web","version":"0.2.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.02}}}}',
                },
            )

        assert resp.status_code == 401

    def test_server_policy_bundle_draft_create_handles_connection_error(
        self, client: TestClient
    ) -> None:
        mock = MagicMock()
        mock.is_authenticated = True

        with (
            patch("app.routes_artifact_editor.get_mud_client", return_value=mock),
            patch(
                "app.routes_artifact_editor.create_server_policy_bundle_draft",
                side_effect=MudServerConnectionError("down"),
            ),
        ):
            resp = client.post(
                "/api/artifacts/server/policy-bundles/pipeworks_web/drafts",
                json={
                    "draft_name": "pipeworks_web_bundle_alt",
                    "content": '{"world_id":"pipeworks_web","version":"0.2.0","source":"test","policy_hash":null,"axes_order":["health"],"axes":{"health":{"group":"character","ordering":["weary"],"thresholds":[{"label":"weary","min":0.4,"max":0.59}]}},"chat_rules":{"channel_multipliers":{"say":1.0,"yell":1.5,"whisper":0.5},"min_gap_threshold":0.05,"axes":{"health":{"resolver":"shared_drain","base_magnitude":0.02}}}}',
                },
            )

        assert resp.status_code == 502

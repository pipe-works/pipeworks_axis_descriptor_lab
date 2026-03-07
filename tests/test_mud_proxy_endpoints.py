"""
Tests for /api/mud/* proxy endpoints in app/main.py.

All tests mock the MudServerClient at the module level to avoid real
HTTP calls.  Tests verify request/response shapes and error handling.

Test coverage:
  - GET/POST /api/mud/mode: returns and switches runtime mode config.
  - POST /api/mud/login: success, failure, standalone mode.
  - POST /api/mud/logout: clears session.
  - GET /api/mud/session: returns translation_mode.
  - GET /api/mud/worlds: proxied list, auth error, standalone 503.
  - GET /api/mud/world-config/{world_id}: proxied config, auth error.
  - POST /api/mud/select-world: stores world_id.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import app.routes_mud as routes_mud
from app.main import app
from app.mud_server_client import (
    MudServerConnectionError,
    MudServerFeatureUnavailableError,
    MudServerSessionExpiredError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


def _mock_mud_client(*, authenticated: bool = False, world_id: str | None = None) -> MagicMock:
    """Create a mock MudServerClient with configurable state."""
    mock = MagicMock()
    mock.is_authenticated = authenticated
    mock.selected_world_id = world_id
    mock.session_status.return_value = {
        "authenticated": authenticated,
        "role": "admin" if authenticated else None,
        "selected_world_id": world_id,
    }
    return mock


def _mode_config(
    *,
    mode_key: str = "standalone",
    translation_mode: str = "standalone",
    active_server_url: str | None = None,
) -> dict:
    """Build a serialisable runtime mode config dict for route tests."""
    return {
        "mode_key": mode_key,
        "translation_mode": translation_mode,
        "active_server_url": active_server_url,
        "available_modes": [
            {
                "key": "standalone",
                "label": "Offline",
                "translation_mode": "standalone",
                "server_url": None,
            },
            {
                "key": "development",
                "label": "Development server",
                "translation_mode": "server-local",
                "server_url": "http://localhost:8000",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers (routes_mud)
# ---------------------------------------------------------------------------


class TestMudRuntimeOptionHelpers:
    """Coverage for runtime-option helper branches used by pipeline bootstrap."""

    def test_parse_inline_token_list_normalizes_values(self) -> None:
        values = routes_mud._parse_inline_token_list(" [ goblin , 'human' , goblin ] ")
        assert values == ["goblin", "human"]

    def test_extract_species_registry_missing_manifest_returns_empty(self, tmp_path: Path) -> None:
        species = routes_mud._extract_species_from_local_policy_registry(
            "pipeworks_web", world_root=tmp_path
        )
        assert species == []

    def test_extract_species_registry_missing_species_registry_key_returns_empty(
        self, tmp_path: Path
    ) -> None:
        world_policies = tmp_path / "pipeworks_web" / "policies"
        world_policies.mkdir(parents=True)
        (world_policies / "manifest.yaml").write_text(
            "\n".join(
                [
                    "image:",
                    "  registries:",
                    "    clothing: policies/image/registries/clothing_registry.yaml",
                ]
            ),
            encoding="utf-8",
        )

        species = routes_mud._extract_species_from_local_policy_registry(
            "pipeworks_web", world_root=tmp_path
        )
        assert species == []

    def test_extract_species_registry_missing_registry_file_returns_empty(
        self, tmp_path: Path
    ) -> None:
        world_policies = tmp_path / "pipeworks_web" / "policies"
        world_policies.mkdir(parents=True)
        (world_policies / "manifest.yaml").write_text(
            "\n".join(
                [
                    "image:",
                    "  registries:",
                    "    species: policies/image/registries/species_registry.yaml",
                ]
            ),
            encoding="utf-8",
        )

        species = routes_mud._extract_species_from_local_policy_registry(
            "pipeworks_web", world_root=tmp_path
        )
        assert species == []

    def test_extract_species_registry_manifest_read_error_returns_empty(
        self, tmp_path: Path
    ) -> None:
        world_policies = tmp_path / "pipeworks_web" / "policies"
        world_policies.mkdir(parents=True)
        manifest_path = world_policies / "manifest.yaml"
        manifest_path.write_text("image:\n  registries:\n", encoding="utf-8")

        original_read_text = Path.read_text

        def _raise_for_manifest(path_obj: Path, *args, **kwargs) -> str:
            if path_obj == manifest_path:
                raise OSError("manifest read failed")
            return original_read_text(path_obj, *args, **kwargs)

        with patch("pathlib.Path.read_text", autospec=True, side_effect=_raise_for_manifest):
            species = routes_mud._extract_species_from_local_policy_registry(
                "pipeworks_web", world_root=tmp_path
            )
        assert species == []

    def test_extract_species_registry_registry_read_error_returns_empty(
        self, tmp_path: Path
    ) -> None:
        world_policies = tmp_path / "pipeworks_web" / "policies"
        registry_dir = world_policies / "image" / "registries"
        registry_dir.mkdir(parents=True)
        (world_policies / "manifest.yaml").write_text(
            "\n".join(
                [
                    "image:",
                    "  registries:",
                    "    species: policies/image/registries/species_registry.yaml",
                ]
            ),
            encoding="utf-8",
        )
        registry_path = registry_dir / "species_registry.yaml"
        registry_path.write_text(
            "entries:\n- id: goblin_pipeworks_v1\n  compatible_species: [goblin]",
            encoding="utf-8",
        )

        original_read_text = Path.read_text

        def _raise_for_registry(path_obj: Path, *args, **kwargs) -> str:
            if path_obj == registry_path:
                raise OSError("registry read failed")
            return original_read_text(path_obj, *args, **kwargs)

        with patch("pathlib.Path.read_text", autospec=True, side_effect=_raise_for_registry):
            species = routes_mud._extract_species_from_local_policy_registry(
                "pipeworks_web", world_root=tmp_path
            )
        assert species == []

    def test_extract_runtime_options_supports_top_level_runtime_options(
        self, tmp_path: Path
    ) -> None:
        world_config = {
            "runtime_options": {
                "species": ["human", "goblin"],
                "gender": ["female", "male"],
            }
        }
        options = routes_mud._extract_runtime_options("pipeworks_web", world_config)
        assert options.species == ["human", "goblin"]
        assert options.gender == ["female", "male"]


# ---------------------------------------------------------------------------
# GET/POST /api/mud/mode
# ---------------------------------------------------------------------------


class TestMudMode:
    """Runtime mode endpoint tests."""

    def test_get_mode_returns_config(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_mode_config", return_value=_mode_config()):
            resp = test_client.get("/api/mud/mode")

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode_key"] == "standalone"
        assert len(data["available_modes"]) == 2

    def test_post_mode_switches_config(self, test_client: TestClient) -> None:
        with patch(
            "app.routes_mud.set_mud_mode",
            return_value=_mode_config(
                mode_key="development",
                translation_mode="server-local",
                active_server_url="http://localhost:8000",
            ),
        ) as mock_set_mode:
            resp = test_client.post("/api/mud/mode", json={"mode_key": "development"})

        assert resp.status_code == 200
        assert resp.json()["translation_mode"] == "server-local"
        mock_set_mode.assert_called_once_with("development", server_url=None)

    def test_post_mode_forwards_development_server_url(self, test_client: TestClient) -> None:
        with patch(
            "app.routes_mud.set_mud_mode",
            return_value=_mode_config(
                mode_key="development",
                translation_mode="server-local",
                active_server_url="http://devbox:8000",
            ),
        ) as mock_set_mode:
            resp = test_client.post(
                "/api/mud/mode",
                json={"mode_key": "development", "server_url": "http://devbox:8000"},
            )

        assert resp.status_code == 200
        assert resp.json()["active_server_url"] == "http://devbox:8000"
        mock_set_mode.assert_called_once_with("development", server_url="http://devbox:8000")

    def test_post_mode_rejects_unknown_key(self, test_client: TestClient) -> None:
        with patch(
            "app.routes_mud.set_mud_mode", side_effect=ValueError("Unknown mud mode 'bad'.")
        ):
            resp = test_client.post("/api/mud/mode", json={"mode_key": "bad"})

        assert resp.status_code == 400
        assert "Unknown mud mode" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/mud/login
# ---------------------------------------------------------------------------


class TestMudLogin:
    """Proxy login endpoint tests."""

    def test_login_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client()
        mock.login.return_value = {
            "success": True,
            "session_id": "abc-123",
            "role": "admin",
            "message": "Login successful.",
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/login", json={"username": "user", "password": "pass"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is True
        assert data["role"] == "admin"

    def test_login_failure(self, test_client: TestClient) -> None:
        mock = _mock_mud_client()
        mock.login.return_value = {
            "success": False,
            "message": "Invalid credentials.",
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/login", json={"username": "user", "password": "wrong"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False

    def test_login_unauthorised_role_returns_denied(self, test_client: TestClient) -> None:
        mock = _mock_mud_client()
        mock.login.return_value = {
            "success": True,
            "session_id": "abc-123",
            "role": "player",
            "message": "Login successful.",
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/login", json={"username": "user", "password": "pass"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert data["role"] == "player"
        assert "author" in data["message"].lower()
        mock.logout.assert_called_once()

    def test_login_standalone_mode(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.post("/api/mud/login", json={"username": "user", "password": "pass"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert "offline" in data["message"].lower() or "standalone" in data["message"].lower()

    def test_login_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client()
        mock.login.side_effect = MudServerConnectionError("unreachable")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/login", json={"username": "user", "password": "pass"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert "connect" in data["message"].lower()

    def test_login_http_status_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client()
        fake_response = MagicMock()
        fake_response.status_code = 500
        mock.login.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=fake_response
        )

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/login", json={"username": "user", "password": "pass"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert "500" in data["message"]


# ---------------------------------------------------------------------------
# POST /api/mud/logout
# ---------------------------------------------------------------------------


class TestMudLogout:
    """Proxy logout endpoint tests."""

    def test_logout_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/logout")

        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock.logout.assert_called_once()

    def test_logout_standalone(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.post("/api/mud/logout")

        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ---------------------------------------------------------------------------
# GET /api/mud/session
# ---------------------------------------------------------------------------


class TestMudSession:
    """Session status endpoint tests."""

    def test_session_authenticated(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True, world_id="pipeworks_web")

        with (
            patch("app.routes_mud.get_mud_client", return_value=mock),
            patch(
                "app.routes_mud.get_mud_mode_config",
                return_value=_mode_config(
                    mode_key="configured",
                    translation_mode="server-prod",
                    active_server_url="https://api.pipe-works.org",
                ),
            ),
        ):
            resp = test_client.get("/api/mud/session")

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is True
        assert data["translation_mode"] == "server-prod"
        assert data["mode_key"] == "configured"
        assert data["selected_world_id"] == "pipeworks_web"
        assert data["active_server_url"] == "https://api.pipe-works.org"

    def test_session_standalone(self, test_client: TestClient) -> None:
        with (
            patch("app.routes_mud.get_mud_client", return_value=None),
            patch("app.routes_mud.get_mud_mode_config", return_value=_mode_config()),
        ):
            resp = test_client.get("/api/mud/session")

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert data["translation_mode"] == "standalone"
        assert data["mode_key"] == "standalone"
        assert data["active_server_url"] is None


# ---------------------------------------------------------------------------
# GET /api/mud/worlds
# ---------------------------------------------------------------------------


class TestMudWorlds:
    """World list proxy endpoint tests."""

    def test_worlds_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.return_value = [
            {"world_id": "pipeworks_web", "name": "Pipeworks Web", "translation_enabled": True}
        ]

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["worlds"]) == 1
        assert data["worlds"][0]["world_id"] == "pipeworks_web"

    def test_worlds_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 401

    def test_worlds_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 503

    def test_worlds_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.side_effect = MudServerConnectionError("down")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 502

    def test_worlds_forbidden_returns_403(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        fake_response = MagicMock()
        fake_response.status_code = 403
        mock.list_worlds.side_effect = httpx.HTTPStatusError(
            "Forbidden", request=MagicMock(), response=fake_response
        )

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 403
        assert "author" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /api/mud/world-config/{world_id}
# ---------------------------------------------------------------------------


class TestMudWorldConfig:
    """World config proxy endpoint tests."""

    def test_world_config_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_config.return_value = {
            "world_id": "pipeworks_web",
            "name": "Pipeworks Web",
            "model": "gemma2:2b",
            "active_axes": ["health", "age"],
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"

    def test_world_config_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_config.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 401

    def test_world_config_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_config.side_effect = MudServerConnectionError("down")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 502

    def test_world_config_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/mud/world-prompts/{world_id}
# ---------------------------------------------------------------------------


class TestMudWorldPrompts:
    """World prompts proxy endpoint tests."""

    def test_world_prompts_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_prompts.return_value = {
            "world_id": "pipeworks_web",
            "prompts": [
                {"filename": "ic_prompt.txt", "content": "template", "is_active": True},
            ],
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-prompts/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"
        assert len(data["prompts"]) == 1

    def test_world_prompts_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_prompts.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-prompts/pipeworks_web")

        assert resp.status_code == 401

    def test_world_prompts_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_prompts.side_effect = MudServerConnectionError("down")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-prompts/pipeworks_web")

        assert resp.status_code == 502

    def test_world_prompts_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.get("/api/mud/world-prompts/pipeworks_web")

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/mud/world-image-policy-bundle/{world_id}
# ---------------------------------------------------------------------------


class TestMudWorldImagePolicyBundle:
    """Image policy bundle proxy endpoint tests."""

    def test_world_image_policy_bundle_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "abc123",
            "composition_order": ["species_canon_block", "descriptor_layer_output"],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species"],
            "missing_components": [],
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-image-policy-bundle/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"
        assert data["policy_hash"] == "abc123"

    def test_world_image_policy_bundle_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-image-policy-bundle/pipeworks_web")

        assert resp.status_code == 401

    def test_world_image_policy_bundle_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.side_effect = MudServerConnectionError("down")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-image-policy-bundle/pipeworks_web")

        assert resp.status_code == 502

    def test_world_image_policy_bundle_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.get("/api/mud/world-image-policy-bundle/pipeworks_web")

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/mud/pipeline-build/bootstrap/{world_id}
# ---------------------------------------------------------------------------


class TestMudPipelineBuildBootstrap:
    """Pipeline bootstrap aggregation endpoint tests."""

    def test_pipeline_bootstrap_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.return_value = [
            {
                "world_id": "pipeworks_web",
                "name": "Pipeworks Web",
                "description": "Canonical web world.",
                "translation_enabled": True,
            }
        ]
        mock.world_config.return_value = {
            "world_id": "pipeworks_web",
            "name": "Pipeworks Web",
            "image_generation": {
                "runtime_options": {
                    "species": ["goblin", "orc"],
                    "gender": ["male", "female"],
                    "world_context_tags": ["ledgerfall", "docks"],
                    "occupation_signals": ["trader", "scribe"],
                }
            },
        }
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "abc123",
            "composition_order": [
                "species_canon_block",
                "clothing_block",
                "descriptor_layer",
                "tone_profile",
            ],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species", "entity.axes"],
            "missing_components": [],
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/pipeline-build/bootstrap/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"
        assert data["world_summary"]["world_row"]["world_id"] == "pipeworks_web"
        assert data["policy_bundle"]["policy_hash"] == "abc123"
        assert data["policy_source"]["source_kind"] == "mud_server_canonical"
        assert data["policy_source"]["source_label"] == "Mud server canonical"
        assert data["policy_source"]["source_path"] is None
        assert data["policy_source"]["reference"]["world_id"] == "pipeworks_web"
        assert data["policy_source"]["reference"]["policy_bundle_id"] == "pipeworks_web_default"
        assert data["policy_source"]["reference"]["policy_bundle_version"] == 1
        assert data["policy_source"]["reference"]["policy_hash"] == "abc123"
        assert data["runtime_options"]["species"] == ["goblin", "orc"]
        assert data["runtime_options"]["world_context_tags"] == ["ledgerfall", "docks"]
        assert data["required_fields"] == [
            "entity.identity.gender",
            "entity.species",
            "entity.axes",
        ]
        mock.list_worlds.assert_called_once_with()
        mock.world_config.assert_called_once_with("pipeworks_web")
        mock.world_image_policy_bundle.assert_called_once_with("pipeworks_web")

    def test_pipeline_bootstrap_species_falls_back_to_local_registry(
        self, test_client: TestClient, tmp_path
    ) -> None:
        """Bootstrap should derive species options from local policy registry when config omits them."""
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.return_value = [
            {
                "world_id": "pipeworks_web",
                "name": "Pipeworks Web",
                "description": "Canonical web world.",
                "translation_enabled": True,
            }
        ]
        mock.world_config.return_value = {
            "world_id": "pipeworks_web",
            "name": "Pipeworks Web",
            "image_generation": {
                "runtime_options": {
                    "gender": ["male", "female"],
                }
            },
        }
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "abc123",
            "composition_order": [
                "species_canon_block",
                "clothing_block",
                "descriptor_layer",
                "tone_profile",
            ],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species", "entity.axes"],
            "missing_components": [],
        }

        world_root = tmp_path / "pipeworks_web" / "policies"
        (world_root / "image" / "registries").mkdir(parents=True)
        (world_root / "manifest.yaml").write_text(
            "\n".join(
                [
                    "image:",
                    "  registries:",
                    "    species: policies/image/registries/species_registry.yaml",
                ]
            ),
            encoding="utf-8",
        )
        (world_root / "image" / "registries" / "species_registry.yaml").write_text(
            "\n".join(
                [
                    "entries:",
                    "- id: goblin_pipeworks_v1",
                    "  compatible_species: [goblin]",
                    "- id: human_pipeworks_v1",
                    "  compatible_species: [human]",
                ]
            ),
            encoding="utf-8",
        )

        with (
            patch("app.routes_mud.get_mud_client", return_value=mock),
            patch("app.routes_mud.WORLD_ROOT", tmp_path),
        ):
            resp = test_client.get("/api/mud/pipeline-build/bootstrap/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["runtime_options"]["species"] == ["goblin", "human"]
        assert data["runtime_options"]["gender"] == ["male", "female"]

    def test_pipeline_bootstrap_world_not_found(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.return_value = [{"world_id": "other_world", "name": "Other"}]

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/pipeline-build/bootstrap/pipeworks_web")

        assert resp.status_code == 404
        data = resp.json()
        assert data["code"] == "PIPELINE_WORLD_NOT_FOUND"
        assert data["stage"] == "session_world"

    def test_pipeline_bootstrap_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/pipeline-build/bootstrap/pipeworks_web")

        assert resp.status_code == 401
        data = resp.json()
        assert data["code"] == "PIPELINE_AUTH_REQUIRED"
        assert data["stage"] == "session_world"

    def test_pipeline_bootstrap_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.side_effect = MudServerConnectionError("down")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/pipeline-build/bootstrap/pipeworks_web")

        assert resp.status_code == 502
        data = resp.json()
        assert data["code"] == "PIPELINE_UPSTREAM_UNAVAILABLE"
        assert data["stage"] == "session_world"

    def test_pipeline_bootstrap_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.get("/api/mud/pipeline-build/bootstrap/pipeworks_web")

        assert resp.status_code == 503
        data = resp.json()
        assert data["code"] == "PIPELINE_MODE_UNAVAILABLE"
        assert data["stage"] == "session_world"


# ---------------------------------------------------------------------------
# POST /api/mud/pipeline-build/resolve-image-selection
# ---------------------------------------------------------------------------


class TestMudPipelineBuildResolveImageSelection:
    """Pipeline resolve preview endpoint tests."""

    def test_pipeline_resolve_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "bundle_hash",
            "composition_order": [
                "species_canon_block",
                "clothing_block",
                "descriptor_layer",
                "tone_profile",
            ],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species", "entity.axes"],
            "missing_components": [],
        }
        mock.compile_image_prompt.return_value = {
            "world_id": "pipeworks_web",
            "policy_hash": "policy_hash_resolved",
            "axis_hash": "axis_hash_resolved",
            "selected_species_block_id": "goblin_v1",
            "selected_descriptor_layer_id": "portrait_surface_v2",
            "selected_tone_profile_id": "archival_neutral_v1",
            "selected_clothing_slot_ids": {
                "environment": "urban_trader_01",
                "activity": "ledger_work_02",
                "wealth": "modest_03",
            },
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"demeanor": {"label": "proud", "score": 0.81}},
                    "world_context": ["ledgerfall"],
                    "occupation_signals": ["trader"],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_blocks"]["species_canon_block"] == "goblin_v1"
        assert data["selected_blocks"]["clothing_block"]["wealth"] == "modest_03"
        assert data["descriptor_layer"] == "portrait_surface_v2"
        assert data["tone_profile"] == "archival_neutral_v1"
        assert data["policy_hash"] == "policy_hash_resolved"
        assert data["axis_hash"] == "axis_hash_resolved"
        assert isinstance(data["compiler_input_hash"], str)
        assert len(data["compiler_input_hash"]) == 64
        call_kwargs = mock.compile_image_prompt.call_args.kwargs
        assert call_kwargs["world_id"] == "pipeworks_web"
        assert call_kwargs["species"] == "goblin"
        assert call_kwargs["gender"] == "male"
        assert call_kwargs["world_context"] == ["ledgerfall"]
        assert call_kwargs["occupation_signals"] == ["trader"]
        assert "model_id" not in call_kwargs
        assert "aspect_ratio" not in call_kwargs
        assert "seed" not in call_kwargs

    def test_pipeline_resolve_compiler_input_hash_is_stable_for_axis_key_order(
        self, test_client: TestClient
    ) -> None:
        """Equivalent axis payloads with different key order should hash identically."""
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "bundle_hash",
            "composition_order": ["species_canon_block", "clothing_block"],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species", "entity.axes"],
            "missing_components": [],
        }
        mock.compile_image_prompt.return_value = {
            "world_id": "pipeworks_web",
            "policy_hash": "policy_hash_resolved",
            "axis_hash": "axis_hash_resolved",
            "selected_species_block_id": "goblin_v1",
            "selected_descriptor_layer_id": "portrait_surface_v2",
            "selected_tone_profile_id": "archival_neutral_v1",
            "selected_clothing_slot_ids": {"wealth": "modest_03"},
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp_a = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {
                        "demeanor": {"label": "proud", "score": 0.81},
                        "health": {"label": "weary", "score": 0.34},
                    },
                    "world_context": ["ledgerfall"],
                    "occupation_signals": ["trader"],
                },
            )
            resp_b = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {
                        "health": {"label": "weary", "score": 0.34},
                        "demeanor": {"label": "proud", "score": 0.81},
                    },
                    "world_context": ["ledgerfall"],
                    "occupation_signals": ["trader"],
                },
            )

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        hash_a = resp_a.json()["compiler_input_hash"]
        hash_b = resp_b.json()["compiler_input_hash"]
        assert hash_a == hash_b

    def test_pipeline_resolve_compiler_input_hash_changes_when_inputs_change(
        self, test_client: TestClient
    ) -> None:
        """Changing runtime inputs should change compiler_input_hash."""
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "bundle_hash",
            "composition_order": ["species_canon_block", "clothing_block"],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species", "entity.axes"],
            "missing_components": [],
        }
        mock.compile_image_prompt.return_value = {
            "world_id": "pipeworks_web",
            "policy_hash": "policy_hash_resolved",
            "axis_hash": "axis_hash_resolved",
            "selected_species_block_id": "goblin_v1",
            "selected_descriptor_layer_id": "portrait_surface_v2",
            "selected_tone_profile_id": "archival_neutral_v1",
            "selected_clothing_slot_ids": {"wealth": "modest_03"},
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp_a = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"demeanor": {"label": "proud", "score": 0.81}},
                    "world_context": ["ledgerfall"],
                    "occupation_signals": ["trader"],
                },
            )
            resp_b = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"demeanor": {"label": "proud", "score": 0.81}},
                    "world_context": ["ledgerfall"],
                    "occupation_signals": ["scribe"],
                },
            )

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        hash_a = resp_a.json()["compiler_input_hash"]
        hash_b = resp_b.json()["compiler_input_hash"]
        assert hash_a != hash_b

    def test_pipeline_resolve_missing_axis_hash_returns_502(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "bundle_hash",
            "composition_order": ["species_canon_block"],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species", "entity.axes"],
            "missing_components": [],
        }
        mock.compile_image_prompt.return_value = {
            "world_id": "pipeworks_web",
            "policy_hash": "policy_hash_resolved",
            "selected_species_block_id": "goblin_v1",
            "selected_clothing_slot_ids": {},
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"demeanor": {"label": "proud", "score": 0.81}},
                },
            )

        assert resp.status_code == 502
        data = resp.json()
        assert data["code"] == "PIPELINE_UPSTREAM_INVALID"
        assert data["stage"] == "compile_output"

    def test_pipeline_resolve_upstream_http_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_bundle_version": 1,
            "policy_hash": "bundle_hash",
            "composition_order": ["species_canon_block"],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species", "entity.axes"],
            "missing_components": [],
        }
        response = MagicMock()
        response.status_code = 409
        response.json.return_value = {
            "detail": "Missing required runtime inputs for compile: entity.axes"
        }
        response.text = "Missing required runtime inputs for compile: entity.axes"
        mock.compile_image_prompt.side_effect = httpx.HTTPStatusError(
            "Conflict",
            request=MagicMock(),
            response=response,
        )

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {},
                },
            )

        assert resp.status_code == 409
        data = resp.json()
        assert data["code"] == "PIPELINE_UPSTREAM_HTTP_ERROR"
        assert data["stage"] == "compile_output"
        assert "Missing required runtime inputs" in data["detail"]

    def test_pipeline_resolve_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"demeanor": {"label": "proud", "score": 0.81}},
                },
            )

        assert resp.status_code == 503
        data = resp.json()
        assert data["code"] == "PIPELINE_MODE_UNAVAILABLE"
        assert data["stage"] == "session_world"

    def test_pipeline_resolve_auth_expired_returns_structured_401(
        self, test_client: TestClient
    ) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_image_policy_bundle.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/resolve-image-selection",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"demeanor": {"label": "proud", "score": 0.81}},
                },
            )

        assert resp.status_code == 401
        data = resp.json()
        assert data["code"] == "PIPELINE_AUTH_REQUIRED"
        assert data["stage"] == "session_world"


# ---------------------------------------------------------------------------
# POST /api/mud/pipeline-build/generate-condition-axis
# ---------------------------------------------------------------------------


class TestMudPipelineBuildGenerateConditionAxis:
    """Pipeline stage-4 canonical axis generation endpoint tests."""

    def test_generate_condition_axis_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.generate_condition_axis_payload.return_value = {
            "axes": {"demeanor": {"label": "proud", "score": 0.81}},
            "policy_hash": "policy_hash_value",
            "seed": 42,
            "world_id": "pipeworks_web",
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/generate-condition-axis",
                json={
                    "world_id": "pipeworks_web",
                    "seed": 42,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"
        assert data["seed"] == 42
        assert data["axes"]["demeanor"]["label"] == "proud"
        mock.generate_condition_axis_payload.assert_called_once_with(
            world_id="pipeworks_web",
            seed=42,
        )

    def test_generate_condition_axis_auth_expired_returns_structured_401(
        self, test_client: TestClient
    ) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.generate_condition_axis_payload.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/generate-condition-axis",
                json={
                    "world_id": "pipeworks_web",
                    "seed": None,
                },
            )

        assert resp.status_code == 401
        data = resp.json()
        assert data["code"] == "PIPELINE_AUTH_REQUIRED"
        assert data["stage"] == "session_world"

    def test_generate_condition_axis_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.generate_condition_axis_payload.side_effect = MudServerConnectionError("down")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/generate-condition-axis",
                json={
                    "world_id": "pipeworks_web",
                    "seed": None,
                },
            )

        assert resp.status_code == 502
        data = resp.json()
        assert data["code"] == "PIPELINE_UPSTREAM_UNAVAILABLE"
        assert data["stage"] == "axis_input"

    def test_generate_condition_axis_upstream_http_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        response = MagicMock()
        response.status_code = 409
        response.json.return_value = {
            "detail": "Condition-axis generation unavailable for this world."
        }
        response.text = "Condition-axis generation unavailable for this world."
        mock.generate_condition_axis_payload.side_effect = httpx.HTTPStatusError(
            "Conflict",
            request=MagicMock(),
            response=response,
        )

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/generate-condition-axis",
                json={
                    "world_id": "pipeworks_web",
                    "seed": 99,
                },
            )

        assert resp.status_code == 409
        data = resp.json()
        assert data["code"] == "PIPELINE_UPSTREAM_HTTP_ERROR"
        assert data["stage"] == "axis_input"
        assert "Condition-axis generation unavailable" in data["detail"]

    def test_generate_condition_axis_unsupported_by_upstream_returns_501(
        self, test_client: TestClient
    ) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.generate_condition_axis_payload.side_effect = MudServerFeatureUnavailableError(
            "Mud server does not expose condition-axis generation endpoints "
            "(/api/lab/generate-condition-axis, /api/lab/generate-axis-payload)."
        )

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/pipeline-build/generate-condition-axis",
                json={
                    "world_id": "pipeworks_web",
                    "seed": None,
                },
            )

        assert resp.status_code == 501
        data = resp.json()
        assert data["code"] == "PIPELINE_UPSTREAM_UNSUPPORTED"
        assert data["stage"] == "axis_input"
        assert "does not expose canonical condition-axis generation" in data["detail"]

    def test_generate_condition_axis_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.post(
                "/api/mud/pipeline-build/generate-condition-axis",
                json={
                    "world_id": "pipeworks_web",
                    "seed": None,
                },
            )

        assert resp.status_code == 503
        data = resp.json()
        assert data["code"] == "PIPELINE_MODE_UNAVAILABLE"
        assert data["stage"] == "session_world"


# ---------------------------------------------------------------------------
# POST /api/mud/compile-image-prompt
# ---------------------------------------------------------------------------


class TestMudCompileImagePrompt:
    """Canonical image compile proxy endpoint tests."""

    def test_compile_image_prompt_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.compile_image_prompt.return_value = {
            "world_id": "pipeworks_web",
            "policy_hash": "abc",
            "axis_hash": "def",
            "compiled_prompt": "Compiled canonical prompt.",
        }

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/compile-image-prompt",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"wealth": {"label": "modest", "score": 0.3}},
                    "world_context": ["coastal"],
                    "occupation_signals": ["manual_labour"],
                    "model_id": "flux-2-klein-4b",
                    "aspect_ratio": "1:1",
                    "seed": 123,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["compiled_prompt"].startswith("Compiled")
        call_kwargs = mock.compile_image_prompt.call_args.kwargs
        assert call_kwargs["world_id"] == "pipeworks_web"
        assert call_kwargs["species"] == "goblin"
        assert call_kwargs["gender"] == "male"
        assert call_kwargs["axes"]["wealth"]["label"] == "modest"
        assert call_kwargs["world_context"] == ["coastal"]
        assert call_kwargs["occupation_signals"] == ["manual_labour"]
        assert call_kwargs["model_id"] == "flux-2-klein-4b"
        assert call_kwargs["aspect_ratio"] == "1:1"
        assert call_kwargs["seed"] == 123

    def test_compile_image_prompt_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.compile_image_prompt.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/compile-image-prompt",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"wealth": {"label": "modest", "score": 0.3}},
                },
            )

        assert resp.status_code == 401

    def test_compile_image_prompt_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.compile_image_prompt.side_effect = MudServerConnectionError("down")

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/compile-image-prompt",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"wealth": {"label": "modest", "score": 0.3}},
                },
            )

        assert resp.status_code == 502

    def test_compile_image_prompt_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.post(
                "/api/mud/compile-image-prompt",
                json={
                    "world_id": "pipeworks_web",
                    "species": "goblin",
                    "gender": "male",
                    "axes": {"wealth": {"label": "modest", "score": 0.3}},
                },
            )

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/mud/select-world
# ---------------------------------------------------------------------------


class TestMudSelectWorld:
    """World selection endpoint tests."""

    def test_select_world_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)

        with patch("app.routes_mud.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/select-world", json={"world_id": "pipeworks_web"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["world_id"] == "pipeworks_web"
        mock.select_world.assert_called_once_with("pipeworks_web")

    def test_select_world_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.routes_mud.get_mud_client", return_value=None):
            resp = test_client.post("/api/mud/select-world", json={"world_id": "pipeworks_web"})

        assert resp.status_code == 503

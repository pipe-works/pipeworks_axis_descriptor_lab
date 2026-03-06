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

from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mud_server_client import MudServerConnectionError, MudServerSessionExpiredError

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

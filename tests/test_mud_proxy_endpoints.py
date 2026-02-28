"""
Tests for /api/mud/* proxy endpoints in app/main.py.

All tests mock the MudServerClient at the module level to avoid real
HTTP calls.  Tests verify request/response shapes and error handling.

Test coverage:
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
    }
    return mock


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

        with patch("app.main.get_mud_client", return_value=mock):
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

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.post(
                "/api/mud/login", json={"username": "user", "password": "wrong"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False

    def test_login_standalone_mode(self, test_client: TestClient) -> None:
        with patch("app.main.get_mud_client", return_value=None):
            resp = test_client.post("/api/mud/login", json={"username": "user", "password": "pass"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert "standalone" in data["message"].lower() or "Standalone" in data["message"]

    def test_login_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client()
        mock.login.side_effect = MudServerConnectionError("unreachable")

        with patch("app.main.get_mud_client", return_value=mock):
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

        with patch("app.main.get_mud_client", return_value=mock):
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

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/logout")

        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock.logout.assert_called_once()

    def test_logout_standalone(self, test_client: TestClient) -> None:
        with patch("app.main.get_mud_client", return_value=None):
            resp = test_client.post("/api/mud/logout")

        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ---------------------------------------------------------------------------
# GET /api/mud/session
# ---------------------------------------------------------------------------


class TestMudSession:
    """Session status endpoint tests."""

    def test_session_authenticated(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)

        with (
            patch("app.main.get_mud_client", return_value=mock),
            patch("app.main.compute_translation_mode", return_value="server-prod"),
        ):
            resp = test_client.get("/api/mud/session")

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is True
        assert data["translation_mode"] == "server-prod"

    def test_session_standalone(self, test_client: TestClient) -> None:
        with (
            patch("app.main.get_mud_client", return_value=None),
            patch("app.main.compute_translation_mode", return_value="standalone"),
        ):
            resp = test_client.get("/api/mud/session")

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert data["translation_mode"] == "standalone"


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

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["worlds"]) == 1
        assert data["worlds"][0]["world_id"] == "pipeworks_web"

    def test_worlds_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 401

    def test_worlds_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.main.get_mud_client", return_value=None):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 503

    def test_worlds_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.list_worlds.side_effect = MudServerConnectionError("down")

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/worlds")

        assert resp.status_code == 502


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

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == "pipeworks_web"

    def test_world_config_auth_expired(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_config.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 401

    def test_world_config_connection_error(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)
        mock.world_config.side_effect = MudServerConnectionError("down")

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 502

    def test_world_config_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.main.get_mud_client", return_value=None):
            resp = test_client.get("/api/mud/world-config/pipeworks_web")

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/mud/select-world
# ---------------------------------------------------------------------------


class TestMudSelectWorld:
    """World selection endpoint tests."""

    def test_select_world_success(self, test_client: TestClient) -> None:
        mock = _mock_mud_client(authenticated=True)

        with patch("app.main.get_mud_client", return_value=mock):
            resp = test_client.post("/api/mud/select-world", json={"world_id": "pipeworks_web"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["world_id"] == "pipeworks_web"
        mock.select_world.assert_called_once_with("pipeworks_web")

    def test_select_world_standalone_503(self, test_client: TestClient) -> None:
        with patch("app.main.get_mud_client", return_value=None):
            resp = test_client.post("/api/mud/select-world", json={"world_id": "pipeworks_web"})

        assert resp.status_code == 503

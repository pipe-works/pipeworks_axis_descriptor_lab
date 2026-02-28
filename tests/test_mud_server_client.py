"""
Tests for app/mud_server_client.py — MudServerClient unit tests.

All tests mock httpx.Client to avoid real HTTP calls.  The client is
tested in isolation from the FastAPI app.

Test coverage:
  - login: success stores session_id, failure clears session.
  - logout: session cleared, server call failure tolerated.
  - session_status: returns correct auth state.
  - translate: request body shape, session_id included.
  - 401 handling: MudServerSessionExpiredError raised, token cleared.
  - Connection error handling: MudServerConnectionError raised.
  - list_worlds / world_config: authenticated GET with query params.
  - select_world: stores world_id in memory.
  - compute_translation_mode: correct mode strings.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.mud_server_client import (
    MudServerClient,
    MudServerConnectionError,
    MudServerSessionExpiredError,
    compute_translation_mode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> MudServerClient:
    """Fresh MudServerClient pointing at a fake server."""
    return MudServerClient("http://fake-server:8000", timeout=5.0)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    """Login stores or clears session based on server response."""

    def test_login_success_stores_session(self, client: MudServerClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "session_id": "abc-123",
            "role": "admin",
            "message": "Login successful.",
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=MagicMock())
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            data = client.login("user", "pass")

        assert client.is_authenticated
        assert data["success"] is True
        assert data["session_id"] == "abc-123"

    def test_login_failure_clears_session(self, client: MudServerClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": False,
            "message": "Invalid credentials.",
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=MagicMock())
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            client.login("user", "wrong")

        assert not client.is_authenticated

    def test_login_connect_error_raises(self, client: MudServerClient) -> None:
        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.post.side_effect = httpx.ConnectError("refused")
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(MudServerConnectionError):
                client.login("user", "pass")


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


class TestLogout:
    """Logout always clears local session."""

    def test_logout_clears_session(self, client: MudServerClient) -> None:
        # Manually set authenticated state
        client._session_id = "abc-123"
        client._role = "admin"

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_resp = MagicMock()
            mock_ctx.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            client.logout()

        assert not client.is_authenticated

    def test_logout_tolerates_server_failure(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.post.side_effect = Exception("server down")
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            client.logout()  # should not raise

        assert not client.is_authenticated


# ---------------------------------------------------------------------------
# Session status
# ---------------------------------------------------------------------------


class TestSessionStatus:
    """session_status returns correct auth state."""

    def test_unauthenticated(self, client: MudServerClient) -> None:
        status = client.session_status()
        assert status["authenticated"] is False
        assert status["role"] is None

    def test_authenticated(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"
        client._role = "superuser"
        status = client.session_status()
        assert status["authenticated"] is True
        assert status["role"] == "superuser"


# ---------------------------------------------------------------------------
# list_worlds / world_config happy path
# ---------------------------------------------------------------------------


class TestListWorlds:
    """list_worlds returns the server's world list."""

    def test_list_worlds_not_authenticated_raises(self, client: MudServerClient) -> None:
        with pytest.raises(MudServerSessionExpiredError):
            client.list_worlds()

    def test_list_worlds_returns_list(self, client: MudServerClient) -> None:
        """Server wraps the list as {"worlds": [...]}."""
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "worlds": [
                {"world_id": "pipeworks_web", "name": "Pipeworks Web", "translation_enabled": True}
            ]
        }

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.get.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            result = client.list_worlds()

        assert isinstance(result, list)
        assert result[0]["world_id"] == "pipeworks_web"

    def test_list_worlds_bare_list_fallback(self, client: MudServerClient) -> None:
        """Bare list (without wrapper) is also accepted for robustness."""
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"world_id": "w1", "name": "World 1", "translation_enabled": True}
        ]

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.get.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            result = client.list_worlds()

        assert isinstance(result, list)
        assert result[0]["world_id"] == "w1"


class TestWorldConfig:
    """world_config returns the server's world configuration."""

    def test_world_config_returns_dict(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "world_id": "pipeworks_web",
            "model": "gemma2:2b",
            "active_axes": ["health", "age"],
        }

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.get.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            result = client.world_config("pipeworks_web")

        assert isinstance(result, dict)
        assert result["world_id"] == "pipeworks_web"


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------


class TestTranslate:
    """translate() sends correct request body and handles responses."""

    def test_translate_request_body_shape(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ic_text": "He grunts.",
            "status": "success",
            "profile_summary": "test",
            "rendered_prompt": "test prompt",
            "model": "gemma2:2b",
            "world_config": {},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            result = client.translate(
                world_id="pipeworks_web",
                axes={"health": {"label": "weary", "score": 0.3}},
                channel="say",
                ooc_message="I look around.",
                seed=42,
                temperature=0.7,
            )

        # Verify the POST body
        call_args = mock_ctx.post.call_args
        body = call_args[1]["json"]
        assert body["session_id"] == "abc-123"
        assert body["world_id"] == "pipeworks_web"
        assert body["ooc_message"] == "I look around."
        assert body["seed"] == 42
        assert result["status"] == "success"

    def test_translate_not_authenticated_raises(self, client: MudServerClient) -> None:
        with pytest.raises(MudServerSessionExpiredError):
            client.translate(
                world_id="test",
                axes={},
                channel="say",
                ooc_message="test",
            )


# ---------------------------------------------------------------------------
# 401 handling
# ---------------------------------------------------------------------------


class TestAuthExpiry:
    """401 from server clears token and raises MudServerSessionExpiredError."""

    def test_get_401_clears_session(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"
        client._role = "admin"

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.get.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(MudServerSessionExpiredError):
                client.list_worlds()

        assert not client.is_authenticated

    def test_post_401_clears_session(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(MudServerSessionExpiredError):
                client.translate(
                    world_id="test",
                    axes={},
                    channel="say",
                    ooc_message="test",
                )

        assert not client.is_authenticated


# ---------------------------------------------------------------------------
# Connection errors
# ---------------------------------------------------------------------------


class TestConnectionErrors:
    """Connection errors raise MudServerConnectionError."""

    def test_get_connect_error(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.get.side_effect = httpx.ConnectError("refused")
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(MudServerConnectionError):
                client.list_worlds()

    def test_post_connect_error(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            mock_ctx.post.side_effect = httpx.ConnectError("refused")
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(MudServerConnectionError):
                client.translate(
                    world_id="test",
                    axes={},
                    channel="say",
                    ooc_message="test",
                )


# ---------------------------------------------------------------------------
# World selection
# ---------------------------------------------------------------------------


class TestWorldSelection:
    """select_world stores world_id in memory."""

    def test_select_world(self, client: MudServerClient) -> None:
        assert client.selected_world_id is None
        client.select_world("pipeworks_web")
        assert client.selected_world_id == "pipeworks_web"

    def test_logout_clears_world(self, client: MudServerClient) -> None:
        client._session_id = "abc"
        client.select_world("pipeworks_web")

        with patch("app.mud_server_client.httpx.Client") as MockClient:
            mock_ctx = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            client.logout()

        assert client.selected_world_id is None


# ---------------------------------------------------------------------------
# compute_translation_mode
# ---------------------------------------------------------------------------


class TestComputeTranslationMode:
    """compute_translation_mode returns correct mode strings."""

    def test_standalone_when_unset(self) -> None:
        with patch("app.mud_server_client._MUD_SERVER_URL", None):
            assert compute_translation_mode() == "standalone"

    def test_server_local_for_localhost(self) -> None:
        with patch("app.mud_server_client._MUD_SERVER_URL", "http://localhost:8000"):
            assert compute_translation_mode() == "server-local"

    def test_server_local_for_127(self) -> None:
        with patch("app.mud_server_client._MUD_SERVER_URL", "http://127.0.0.1:8000"):
            assert compute_translation_mode() == "server-local"

    def test_server_prod_for_domain(self) -> None:
        with patch("app.mud_server_client._MUD_SERVER_URL", "https://api.pipe-works.org"):
            assert compute_translation_mode() == "server-prod"

"""
Tests for app/mud_server_client.py — MudServerClient unit tests.

All tests mock the persistent ``httpx.Client`` instance (``_client``) to
avoid real HTTP calls.  The client is tested in isolation from the FastAPI
app.

Test coverage:
  - login: success stores session_id, failure clears session.
  - logout: session cleared, server call failure tolerated.
  - session_status: returns correct auth state.
  - translate: request body shape, session_id included.
  - 401 handling: MudServerSessionExpiredError raised, token cleared.
  - Connection error handling: MudServerConnectionError raised.
  - Timeout handling: TimeoutException raised → MudServerConnectionError.
  - list_worlds / world_config: authenticated GET with query params.
  - select_world: stores world_id in memory.
  - compute_translation_mode: correct mode strings.
  - close: httpx.Client.close called.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

import app.mud_server_client as mud_client_module
from app.mud_server_client import (
    MudServerClient,
    MudServerConnectionError,
    MudServerFeatureUnavailableError,
    MudServerSessionExpiredError,
    get_mud_client,
    get_mud_mode_config,
    list_mud_mode_options,
    set_mud_mode,
    compute_translation_mode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> MudServerClient:
    """Fresh MudServerClient with a mocked httpx.Client."""
    with patch("app.mud_server_client.httpx.Client"):
        c = MudServerClient("http://fake-server:8000", timeout=5.0)
    return c


@pytest.fixture(autouse=True)
def reset_runtime_mode_state() -> None:
    """Reset shared mud runtime state so tests do not leak mode changes."""
    mud_client_module.close_all_mud_clients()
    mud_client_module._ACTIVE_MODE_KEY = "standalone"
    mud_client_module._RUNTIME_DEV_SERVER_URL = mud_client_module._DEV_MUD_SERVER_URL
    yield
    mud_client_module.close_all_mud_clients()
    mud_client_module._RUNTIME_DEV_SERVER_URL = mud_client_module._DEV_MUD_SERVER_URL
    mud_client_module._ACTIVE_MODE_KEY = mud_client_module._default_mode_key()


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
        client._client.post.return_value = mock_resp

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
        client._client.post.return_value = mock_resp

        client.login("user", "wrong")

        assert not client.is_authenticated

    def test_login_connect_error_raises(self, client: MudServerClient) -> None:
        client._client.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(MudServerConnectionError):
            client.login("user", "pass")

    def test_login_timeout_raises_connection_error(self, client: MudServerClient) -> None:
        """TimeoutException during login → MudServerConnectionError."""
        client._client.post.side_effect = httpx.ReadTimeout("timed out")

        with pytest.raises(MudServerConnectionError):
            client.login("user", "pass")


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


class TestLogout:
    """Logout always clears local session."""

    def test_logout_clears_session(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"
        client._role = "admin"
        mock_resp = MagicMock()
        client._client.post.return_value = mock_resp

        client.logout()

        assert not client.is_authenticated

    def test_logout_tolerates_server_failure(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"
        client._client.post.side_effect = Exception("server down")

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
        client._client.get.return_value = mock_resp

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
        client._client.get.return_value = mock_resp

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
        client._client.get.return_value = mock_resp

        result = client.world_config("pipeworks_web")

        assert isinstance(result, dict)
        assert result["world_id"] == "pipeworks_web"


class TestWorldPolicyBundle:
    """world_policy_bundle returns the server's normalized policy bundle."""

    def test_world_policy_bundle_returns_dict(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "world_id": "pipeworks_web",
            "version": "0.1.0",
            "axes_order": ["demeanor"],
            "axes": {"demeanor": {"ordering": ["timid", "proud"]}},
            "chat_rules": {"axes": {"demeanor": {"resolver": "dominance_shift"}}},
        }
        client._client.get.return_value = mock_resp

        result = client.world_policy_bundle("pipeworks_web")

        assert isinstance(result, dict)
        assert result["world_id"] == "pipeworks_web"
        assert result["axes_order"] == ["demeanor"]

    def test_create_world_policy_bundle_draft_posts_expected_body(
        self, client: MudServerClient
    ) -> None:
        """create_world_policy_bundle_draft sends a create-only draft body."""

        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "pipeworks_web_bundle_alt",
            "origin_path": "policies/drafts/pipeworks_web_bundle_alt.json",
            "world_id": "pipeworks_web",
            "version": "0.2.0",
            "based_on_name": "pipeworks_web_policy_bundle",
        }
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.create_world_policy_bundle_draft(
            world_id="pipeworks_web",
            draft_name="pipeworks_web_bundle_alt",
            content={
                "world_id": "pipeworks_web",
                "version": "0.2.0",
                "source": "test",
                "policy_hash": None,
                "axes_order": ["demeanor"],
                "axes": {"demeanor": {"ordering": ["timid", "proud"]}},
                "chat_rules": {"axes": {"demeanor": {"resolver": "dominance_shift"}}},
            },
            based_on_name="pipeworks_web_policy_bundle",
        )

        call_args = client._client.post.call_args
        body = call_args[1]["json"]
        assert body["session_id"] == "abc-123"
        assert body["draft_name"] == "pipeworks_web_bundle_alt"
        assert body["based_on_name"] == "pipeworks_web_policy_bundle"
        assert body["content"]["world_id"] == "pipeworks_web"
        assert result["origin_path"] == "policies/drafts/pipeworks_web_bundle_alt.json"

    def test_world_policy_bundle_drafts_returns_dict(self, client: MudServerClient) -> None:
        """world_policy_bundle_drafts returns the mud server's draft listing payload."""

        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
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
        client._client.get.return_value = mock_resp

        result = client.world_policy_bundle_drafts("pipeworks_web")

        assert result["drafts"][0]["name"] == "pipeworks_web_bundle_alt"

    def test_world_policy_bundle_draft_returns_dict(self, client: MudServerClient) -> None:
        """world_policy_bundle_draft returns one mud-server draft payload."""

        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "pipeworks_web_bundle_alt",
            "origin_path": "policies/drafts/pipeworks_web_bundle_alt.json",
            "world_id": "pipeworks_web",
            "version": "0.2.0",
            "content": {
                "world_id": "pipeworks_web",
                "version": "0.2.0",
                "source": "test",
                "policy_hash": None,
                "axes_order": ["demeanor"],
                "axes": {"demeanor": {"ordering": ["timid", "proud"]}},
                "chat_rules": {"axes": {"demeanor": {"resolver": "dominance_shift"}}},
            },
        }
        client._client.get.return_value = mock_resp

        result = client.world_policy_bundle_draft("pipeworks_web", "pipeworks_web_bundle_alt")

        assert result["name"] == "pipeworks_web_bundle_alt"

    def test_promote_world_policy_bundle_draft_posts_expected_body(
        self, client: MudServerClient
    ) -> None:
        """promote_world_policy_bundle_draft sends the expected promotion payload."""

        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "pipeworks_web_bundle_alt",
            "world_id": "pipeworks_web",
            "canonical_name": "pipeworks_web_policy_bundle",
            "source_files": [
                "policies/axes.yaml",
                "policies/thresholds.yaml",
                "policies/resolution.yaml",
            ],
            "version": "0.2.0",
            "policy_hash": "abc123",
        }
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.promote_world_policy_bundle_draft(
            world_id="pipeworks_web",
            draft_name="pipeworks_web_bundle_alt",
        )

        assert result["canonical_name"] == "pipeworks_web_policy_bundle"
        call_args = client._client.post.call_args
        assert call_args[0][0].endswith(
            "/api/lab/world-policy-bundle/pipeworks_web/drafts/pipeworks_web_bundle_alt/promote"
        )
        assert call_args[1]["json"] == {"session_id": "abc-123"}


class TestWorldImagePolicyBundle:
    """world_image_policy_bundle returns the server's image policy bundle."""

    def test_world_image_policy_bundle_returns_dict(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "world_id": "pipeworks_web",
            "policy_schema": "pipeworks_policy_v1",
            "policy_bundle_id": "pipeworks_web_default",
            "policy_hash": "abc123",
            "composition_order": ["species_canon_block", "descriptor_layer_output"],
            "required_runtime_inputs": ["entity.identity.gender", "entity.species"],
            "missing_components": [],
        }
        client._client.get.return_value = mock_resp

        result = client.world_image_policy_bundle("pipeworks_web")

        assert isinstance(result, dict)
        assert result["world_id"] == "pipeworks_web"
        assert result["policy_hash"] == "abc123"


class TestCompileImagePrompt:
    """compile_image_prompt forwards canonical compile requests."""

    def test_compile_image_prompt_posts_expected_body(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "world_id": "pipeworks_web",
            "policy_hash": "abc",
            "axis_hash": "def",
            "compiled_prompt": "Compiled canonical prompt.",
        }
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.compile_image_prompt(
            world_id="pipeworks_web",
            species="goblin",
            gender="male",
            axes={"wealth": {"label": "modest", "score": 0.3}},
            world_context=["coastal"],
            occupation_signals=["manual_labour"],
            model_id="flux-2-klein-4b",
            aspect_ratio="1:1",
            seed=123,
        )

        assert result["compiled_prompt"].startswith("Compiled")
        call_args = client._client.post.call_args
        assert call_args[0][0].endswith("/api/lab/compile-image-prompt")
        body = call_args[1]["json"]
        assert body["session_id"] == "abc-123"
        assert body["world_id"] == "pipeworks_web"
        assert body["species"] == "goblin"
        assert body["gender"] == "male"
        assert body["axes"]["wealth"]["label"] == "modest"
        assert body["world_context"] == ["coastal"]
        assert body["occupation_signals"] == ["manual_labour"]
        assert body["model_id"] == "flux-2-klein-4b"
        assert body["aspect_ratio"] == "1:1"
        assert body["seed"] == 123

    def test_compile_image_prompt_not_authenticated_raises(self, client: MudServerClient) -> None:
        with pytest.raises(MudServerSessionExpiredError):
            client.compile_image_prompt(
                world_id="pipeworks_web",
                species="goblin",
                gender="male",
                axes={"wealth": {"label": "modest", "score": 0.3}},
            )


class TestGenerateConditionAxisPayload:
    """generate_condition_axis_payload forwards canonical axis generation requests."""

    def test_generate_condition_axis_payload_posts_expected_body(
        self, client: MudServerClient
    ) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "axes": {"demeanor": {"label": "proud", "score": 0.81}},
            "policy_hash": "policy_hash_value",
            "seed": 42,
            "world_id": "pipeworks_web",
        }
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.generate_condition_axis_payload(
            world_id="pipeworks_web",
            seed=42,
            species="goblin",
            gender="male",
        )

        assert result["world_id"] == "pipeworks_web"
        call_args = client._client.post.call_args
        assert call_args[0][0].endswith("/api/pipeline/condition-axis/generate")
        assert call_args[1]["params"]["session_id"] == "abc-123"
        body = call_args[1]["json"]
        assert body["world_id"] == "pipeworks_web"
        assert body["seed"] == 42
        assert body["inputs"]["entity"]["species"] == "goblin"
        assert body["inputs"]["entity"]["identity"]["gender"] == "male"

    def test_generate_condition_axis_payload_404_raises_unavailable(
        self, client: MudServerClient
    ) -> None:
        client._session_id = "abc-123"

        resp = MagicMock()
        resp.status_code = 404
        resp.text = "Not Found"
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=resp,
        )
        client._client.post.return_value = resp

        with pytest.raises(MudServerFeatureUnavailableError, match="canonical condition-axis"):
            client.generate_condition_axis_payload(
                world_id="pipeworks_web",
                seed=None,
                species="goblin",
                gender="male",
            )

        call_args = client._client.post.call_args
        assert call_args[0][0].endswith("/api/pipeline/condition-axis/generate")
        assert call_args[1]["params"]["session_id"] == "abc-123"

    def test_generate_condition_axis_payload_not_authenticated_raises(
        self, client: MudServerClient
    ) -> None:
        with pytest.raises(MudServerSessionExpiredError):
            client.generate_condition_axis_payload(
                world_id="pipeworks_web",
                seed=None,
                species="goblin",
                gender="male",
            )

    def test_generate_condition_axis_payload_connect_error_raises_connection_error(
        self, client: MudServerClient
    ) -> None:
        client._session_id = "abc-123"
        client._client.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(MudServerConnectionError, match="Cannot connect"):
            client.generate_condition_axis_payload(
                world_id="pipeworks_web",
                seed=7,
                species="goblin",
                gender="male",
            )

    def test_generate_condition_axis_payload_401_clears_session_and_raises(
        self, client: MudServerClient
    ) -> None:
        client._session_id = "abc-123"
        client._role = "admin"
        expired = MagicMock()
        expired.status_code = 401
        client._client.post.return_value = expired

        with pytest.raises(MudServerSessionExpiredError, match="Session expired"):
            client.generate_condition_axis_payload(
                world_id="pipeworks_web",
                seed=7,
                species="goblin",
                gender="male",
            )

        assert client._session_id is None
        assert client._role is None

    def test_generate_condition_axis_payload_non_404_http_error_raises(
        self, client: MudServerClient
    ) -> None:
        client._session_id = "abc-123"
        error_resp = MagicMock()
        error_resp.status_code = 409
        error_resp.text = "conflict"
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Conflict",
            request=MagicMock(),
            response=error_resp,
        )
        client._client.post.return_value = error_resp

        with pytest.raises(httpx.HTTPStatusError):
            client.generate_condition_axis_payload(
                world_id="pipeworks_web",
                seed=7,
                species="goblin",
                gender="male",
            )

    def test_generate_condition_axis_payload_invalid_json_raises_type_error(
        self, client: MudServerClient
    ) -> None:
        client._session_id = "abc-123"
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()
        ok_response.json.side_effect = ValueError("invalid json")
        client._client.post.return_value = ok_response

        with pytest.raises(TypeError, match="Invalid JSON response"):
            client.generate_condition_axis_payload(
                world_id="pipeworks_web",
                seed=7,
                species="goblin",
                gender="male",
            )


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
        client._client.post.return_value = mock_resp

        result = client.translate(
            world_id="pipeworks_web",
            axes={"health": {"label": "weary", "score": 0.3}},
            channel="say",
            ooc_message="I look around.",
            seed=42,
            temperature=0.7,
        )

        # Verify the POST body
        call_args = client._client.post.call_args
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
        client._client.get.return_value = mock_resp

        with pytest.raises(MudServerSessionExpiredError):
            client.list_worlds()

        assert not client.is_authenticated

    def test_post_401_clears_session(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        client._client.post.return_value = mock_resp

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
        client._client.get.side_effect = httpx.ConnectError("refused")

        with pytest.raises(MudServerConnectionError):
            client.list_worlds()

    def test_post_connect_error(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"
        client._client.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(MudServerConnectionError):
            client.translate(
                world_id="test",
                axes={},
                channel="say",
                ooc_message="test",
            )


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Timeout exceptions are caught and raised as MudServerConnectionError."""

    def test_get_read_timeout(self, client: MudServerClient) -> None:
        """ReadTimeout on GET → MudServerConnectionError."""
        client._session_id = "abc-123"
        client._client.get.side_effect = httpx.ReadTimeout("timed out")

        with pytest.raises(MudServerConnectionError):
            client.list_worlds()

    def test_post_read_timeout(self, client: MudServerClient) -> None:
        """ReadTimeout on POST → MudServerConnectionError."""
        client._session_id = "abc-123"
        client._client.post.side_effect = httpx.ReadTimeout("timed out")

        with pytest.raises(MudServerConnectionError):
            client.translate(
                world_id="test",
                axes={},
                channel="say",
                ooc_message="test",
            )

    def test_post_connect_timeout(self, client: MudServerClient) -> None:
        """ConnectTimeout on POST → MudServerConnectionError."""
        client._session_id = "abc-123"
        client._client.post.side_effect = httpx.ConnectTimeout("connect timed out")

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


class TestWorldPrompts:
    """world_prompts returns the server's prompt template files."""

    def test_world_prompts_returns_dict(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "world_id": "pipeworks_web",
            "prompts": [
                {"filename": "ic_prompt.txt", "content": "template text", "is_active": True},
            ],
        }
        client._client.get.return_value = mock_resp

        result = client.world_prompts("pipeworks_web")

        assert isinstance(result, dict)
        assert result["world_id"] == "pipeworks_web"
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["is_active"] is True

    def test_world_prompts_not_authenticated_raises(self, client: MudServerClient) -> None:
        with pytest.raises(MudServerSessionExpiredError):
            client.world_prompts("pipeworks_web")

    def test_world_prompts_falls_back_to_policy_api_when_legacy_route_missing(
        self, client: MudServerClient
    ) -> None:
        """Legacy 404 should resolve the active prompt from canonical policy endpoints."""

        client._session_id = "abc-123"

        legacy_req = httpx.Request(
            "GET",
            "http://fake-server:8000/api/lab/world-prompts/pipeworks_web",
        )
        legacy_resp = MagicMock()
        legacy_resp.status_code = 404
        legacy_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "missing",
            request=legacy_req,
            response=legacy_resp,
        )

        activations_resp = MagicMock()
        activations_resp.status_code = 200
        activations_resp.raise_for_status = MagicMock()
        activations_resp.json.return_value = {
            "items": [
                {
                    "policy_id": "prompt:translation.prompts.ic:default",
                    "variant": "v1",
                }
            ]
        }

        policy_resp = MagicMock()
        policy_resp.status_code = 200
        policy_resp.raise_for_status = MagicMock()
        policy_resp.json.return_value = {
            "policy_key": "default",
            "content": {"text": "Canonical prompt {{profile_summary}}"},
        }

        client._client.get.side_effect = [legacy_resp, activations_resp, policy_resp]

        result = client.world_prompts("pipeworks_web")

        assert result["world_id"] == "pipeworks_web"
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["filename"] == "default.txt"
        assert result["prompts"][0]["is_active"] is True
        assert "Canonical prompt" in result["prompts"][0]["content"]

    def test_world_prompts_fallback_returns_empty_when_policy_lookup_unavailable(
        self, client: MudServerClient
    ) -> None:
        """When both old and fallback routes are unavailable, return empty prompts."""

        client._session_id = "abc-123"

        legacy_req = httpx.Request(
            "GET",
            "http://fake-server:8000/api/lab/world-prompts/pipeworks_web",
        )
        legacy_resp = MagicMock()
        legacy_resp.status_code = 404
        legacy_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "missing",
            request=legacy_req,
            response=legacy_resp,
        )

        activation_req = httpx.Request(
            "GET",
            "http://fake-server:8000/api/policy-activations",
        )
        activation_resp = MagicMock()
        activation_resp.status_code = 404
        activation_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "missing",
            request=activation_req,
            response=activation_resp,
        )

        client._client.get.side_effect = [legacy_resp, activation_resp]

        result = client.world_prompts("pipeworks_web")

        assert result == {"world_id": "pipeworks_web", "prompts": []}


class TestWorldPromptDrafts:
    """Prompt-draft mud-server helpers round-trip the expected payloads."""

    def test_create_world_prompt_draft_posts_payload(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "ic_prompt_variant",
            "origin_path": "policies/drafts/ic_prompt_variant.txt",
            "world_id": "pipeworks_web",
            "based_on_name": "ic_prompt",
        }
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.create_world_prompt_draft(
            world_id="pipeworks_web",
            draft_name="ic_prompt_variant",
            content="Prompt {{profile_summary}}\n",
            based_on_name="ic_prompt",
        )

        assert result["name"] == "ic_prompt_variant"
        body = client._client.post.call_args[1]["json"]
        assert body["draft_name"] == "ic_prompt_variant"
        assert body["content"] == "Prompt {{profile_summary}}\n"
        assert body["based_on_name"] == "ic_prompt"

    def test_world_prompt_drafts_returns_dict(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "world_id": "pipeworks_web",
            "drafts": [
                {
                    "name": "ic_prompt_variant",
                    "origin_path": "policies/drafts/ic_prompt_variant.txt",
                    "world_id": "pipeworks_web",
                }
            ],
        }
        client._client.get.return_value = mock_resp

        result = client.world_prompt_drafts("pipeworks_web")

        assert result["world_id"] == "pipeworks_web"
        assert result["drafts"][0]["name"] == "ic_prompt_variant"

    def test_world_prompt_draft_returns_dict(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "ic_prompt_variant",
            "origin_path": "policies/drafts/ic_prompt_variant.txt",
            "world_id": "pipeworks_web",
            "content": "Prompt {{profile_summary}}\n",
        }
        client._client.get.return_value = mock_resp

        result = client.world_prompt_draft("pipeworks_web", "ic_prompt_variant")

        assert result["name"] == "ic_prompt_variant"
        assert result["content"].startswith("Prompt")

    def test_world_prompt_drafts_not_authenticated_raises(self, client: MudServerClient) -> None:
        with pytest.raises(MudServerSessionExpiredError):
            client.world_prompt_drafts("pipeworks_web")

    def test_promote_world_prompt_draft_posts_payload(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "ic_prompt_variant",
            "world_id": "pipeworks_web",
            "canonical_name": "ic_prompt_v2",
            "canonical_path": "policies/ic_prompt_v2.txt",
            "active_prompt_path": "policies/ic_prompt_v2.txt",
        }
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.promote_world_prompt_draft(
            world_id="pipeworks_web",
            draft_name="ic_prompt_variant",
            target_name="ic_prompt_v2",
        )

        assert result["canonical_name"] == "ic_prompt_v2"
        body = client._client.post.call_args[1]["json"]
        assert body["target_name"] == "ic_prompt_v2"


class TestTranslatePromptOverride:
    """translate() with prompt_template_override."""

    def test_translate_includes_override_in_body(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ic_text": "ok", "status": "success"}
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        client.translate(
            world_id="pipeworks_web",
            axes={},
            channel="say",
            ooc_message="test",
            prompt_template_override="Custom prompt: {{profile_summary}}",
        )

        body = client._client.post.call_args[1]["json"]
        assert body["prompt_template_override"] == "Custom prompt: {{profile_summary}}"

    def test_translate_omits_override_when_none(self, client: MudServerClient) -> None:
        client._session_id = "abc-123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ic_text": "ok", "status": "success"}
        mock_resp.raise_for_status = MagicMock()
        client._client.post.return_value = mock_resp

        client.translate(
            world_id="pipeworks_web",
            axes={},
            channel="say",
            ooc_message="test",
        )

        body = client._client.post.call_args[1]["json"]
        assert "prompt_template_override" not in body


class TestWorldSelection:
    """select_world stores world_id in memory."""

    def test_select_world(self, client: MudServerClient) -> None:
        assert client.selected_world_id is None
        client.select_world("pipeworks_web")
        assert client.selected_world_id == "pipeworks_web"

    def test_logout_clears_world(self, client: MudServerClient) -> None:
        client._session_id = "abc"
        client.select_world("pipeworks_web")
        client._client.post.return_value = MagicMock()

        client.logout()

        assert client.selected_world_id is None


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    """close() closes the underlying httpx.Client."""

    def test_close_calls_client_close(self, client: MudServerClient) -> None:
        client.close()
        client._client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Runtime mode management
# ---------------------------------------------------------------------------


class TestRuntimeModeConfig:
    """Runtime mode helpers expose selectable modes and cache clients by URL."""

    def test_mode_options_include_offline_and_development(self) -> None:
        with (
            patch.object(mud_client_module, "_ENV_MUD_SERVER_URL", "https://api.pipe-works.org"),
            patch.object(mud_client_module, "_DEV_MUD_SERVER_URL", "http://localhost:8000"),
        ):
            options = list_mud_mode_options()

        keys = [option["key"] for option in options]
        assert keys == ["standalone", "development", "configured"]
        assert options[0]["translation_mode"] == "standalone"
        assert options[1]["server_url"] == "http://localhost:8000"
        assert options[2]["translation_mode"] == "server-prod"

    def test_get_mode_config_returns_active_server_url(self) -> None:
        with (
            patch.object(mud_client_module, "_ENV_MUD_SERVER_URL", "https://api.pipe-works.org"),
            patch.object(mud_client_module, "_DEV_MUD_SERVER_URL", "http://localhost:8000"),
            patch.object(mud_client_module, "_ACTIVE_MODE_KEY", "configured"),
        ):
            config = get_mud_mode_config()

        assert config["mode_key"] == "configured"
        assert config["translation_mode"] == "server-prod"
        assert config["active_server_url"] == "https://api.pipe-works.org"

    def test_default_mode_is_standalone_even_when_server_urls_exist(self) -> None:
        """Startup should remain local/offline to avoid immediate login prompts."""
        with (
            patch.object(mud_client_module, "_ENV_MUD_SERVER_URL", "https://api.pipe-works.org"),
            patch.object(mud_client_module, "_RUNTIME_DEV_SERVER_URL", "http://localhost:8000"),
        ):
            assert mud_client_module._default_mode_key() == "standalone"

    def test_set_mode_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown mud mode"):
            set_mud_mode("does-not-exist")

    def test_set_mode_replaces_development_server_url(self) -> None:
        with patch.object(mud_client_module, "_DEV_MUD_SERVER_URL", "http://localhost:8000"):
            config = set_mud_mode("development", server_url="http://devbox:8000/")

        assert config["active_server_url"] == "http://devbox:8000"
        assert mud_client_module._RUNTIME_DEV_SERVER_URL == "http://devbox:8000"

    def test_set_mode_rejects_empty_development_server_url(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            set_mud_mode("development", server_url="   ")

    def test_get_mud_client_returns_none_in_offline_mode(self) -> None:
        with patch.object(mud_client_module, "_ACTIVE_MODE_KEY", "standalone"):
            assert get_mud_client() is None

    def test_get_mud_client_caches_client_per_url(self) -> None:
        fake_client = MagicMock()
        with (
            patch.object(mud_client_module, "_DEV_MUD_SERVER_URL", "http://localhost:8000"),
            patch.object(mud_client_module, "_ACTIVE_MODE_KEY", "development"),
            patch("app.mud_server_client.MudServerClient", return_value=fake_client) as mock_cls,
        ):
            first = get_mud_client()
            second = get_mud_client()

        assert first is fake_client
        assert second is fake_client
        mock_cls.assert_called_once_with(
            "http://localhost:8000", timeout=mud_client_module._MUD_SERVER_TIMEOUT
        )

    def test_get_mud_client_uses_runtime_development_url_override(self) -> None:
        fake_client = MagicMock()
        with (
            patch.object(mud_client_module, "_ACTIVE_MODE_KEY", "development"),
            patch.object(mud_client_module, "_RUNTIME_DEV_SERVER_URL", "http://devbox:8000"),
            patch("app.mud_server_client.MudServerClient", return_value=fake_client) as mock_cls,
        ):
            result = get_mud_client()

        assert result is fake_client
        mock_cls.assert_called_once_with(
            "http://devbox:8000", timeout=mud_client_module._MUD_SERVER_TIMEOUT
        )


# ---------------------------------------------------------------------------
# compute_translation_mode
# ---------------------------------------------------------------------------


class TestComputeTranslationMode:
    """compute_translation_mode returns the active runtime mode string."""

    def test_standalone_when_unset(self) -> None:
        with patch.object(mud_client_module, "_ACTIVE_MODE_KEY", "standalone"):
            assert compute_translation_mode() == "standalone"

    def test_server_local_for_development_mode(self) -> None:
        with (
            patch.object(mud_client_module, "_DEV_MUD_SERVER_URL", "http://localhost:8000"),
            patch.object(mud_client_module, "_ACTIVE_MODE_KEY", "development"),
        ):
            assert compute_translation_mode() == "server-local"

    def test_server_prod_for_configured_mode(self) -> None:
        with (
            patch.object(mud_client_module, "_ENV_MUD_SERVER_URL", "https://api.pipe-works.org"),
            patch.object(mud_client_module, "_ACTIVE_MODE_KEY", "configured"),
        ):
            assert compute_translation_mode() == "server-prod"

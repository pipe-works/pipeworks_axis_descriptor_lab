"""
Tests for POST /api/translate_chat — OOC→IC translation endpoint.

All tests mock ``ChatRenderer.render`` at the ``app.chat_renderer`` module
level to avoid real HTTP calls.  The inner ``_translate_one`` function is
not directly testable (it is defined inside the route handler), so all
assertions go through the HTTP interface using FastAPI's TestClient.

Test coverage:
  - Happy path: single character and both characters.
  - Character B optional: sent as null when omitted.
  - API error path: renderer returns None → fallback.api_error.
  - Validation failure: renderer returns PASSTHROUGH → fallback.validation_failed.
  - Active axes filtering: disabled axes absent from rendered profile.
  - Inline system_prompt overrides file-based prompt.
  - prompt_name loads from disk.
  - IPC hashes present on success; output_hash and ipc_id absent on failure.
  - Pydantic validation: missing required fields → 422.
  - Strict mode and lenient mode flags forwarded correctly.
  - Server-mode translation via MudServerClient proxy.
  - Fallback to standalone when client not authenticated.
  - MudServerSessionExpiredError during translate → 401.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mud_server_client import MudServerConnectionError, MudServerSessionExpiredError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _standalone_mode():
    """Default to standalone mode (no mud server) for all tests in this module.

    Server-mode tests override this with their own
    ``patch("app.main.get_mud_client", return_value=mock)`` context manager.
    """
    with patch("app.main.get_mud_client", return_value=None):
        yield


@pytest.fixture()
def client() -> TestClient:
    """FastAPI test client (same as conftest but explicit for clarity)."""
    return TestClient(app)


@pytest.fixture()
def axes_a() -> dict:
    """Minimal axes dict for Character A suitable for a translate request."""
    return {
        "health": {"label": "weary", "score": 0.3},
        "age": {"label": "old", "score": 0.75},
    }


@pytest.fixture()
def axes_b() -> dict:
    """Minimal axes dict for Character B."""
    return {
        "health": {"label": "vigorous", "score": 0.8},
        "age": {"label": "young", "score": 0.25},
    }


@pytest.fixture()
def base_request(axes_a: dict) -> dict:
    """Minimal valid ChatTranslationRequest body for Character A only."""
    return {
        "character_a": {
            "axes": axes_a,
            "ooc_message": "I look around the room.",
            "channel": "say",
        },
        "model": "gemma2:2b",
        "temperature": 0.7,
        "max_tokens": 128,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_renderer(return_value: str | None):
    """Patch ChatRenderer.render to return a fixed value.

    Patches at the module level where the class is defined, which is where
    the ``from app.chat_renderer import ChatRenderer`` inside the route
    handler picks it up.
    """
    return patch("app.chat_renderer.ChatRenderer.render", return_value=return_value)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestTranslateChatSuccess:
    """Successful translation scenarios."""

    def test_single_character_returns_200(self, client: TestClient, base_request: dict) -> None:
        with _patch_renderer("She peers cautiously about the chamber."):
            resp = client.post("/api/translate_chat", json=base_request)
        assert resp.status_code == 200

    def test_single_character_status_is_success(
        self, client: TestClient, base_request: dict
    ) -> None:
        with _patch_renderer("She peers cautiously about the chamber."):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert data["character_a"]["status"] == "success"

    def test_single_character_ic_text_returned(
        self, client: TestClient, base_request: dict
    ) -> None:
        expected = "She peers cautiously about the chamber."
        with _patch_renderer(expected):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert data["character_a"]["ic_text"] == expected

    def test_character_b_none_when_not_provided(
        self, client: TestClient, base_request: dict
    ) -> None:
        """When character_b is absent from the request, the response b is null."""
        with _patch_renderer("ok"):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert data["character_b"] is None

    def test_both_characters_translated(
        self, client: TestClient, base_request: dict, axes_b: dict
    ) -> None:
        req = {
            **base_request,
            "character_b": {
                "axes": axes_b,
                "ooc_message": "I grin at the stranger.",
                "channel": "say",
            },
        }
        with _patch_renderer("He grins wolfishly at the newcomer."):
            data = client.post("/api/translate_chat", json=req).json()
        assert data["character_a"]["status"] == "success"
        assert data["character_b"]["status"] == "success"
        assert data["character_b"]["ic_text"] is not None

    def test_ipc_hashes_present_on_success(self, client: TestClient, base_request: dict) -> None:
        """All four IPC fields must be non-null on a successful translation."""
        with _patch_renderer("A weathered figure nods."):
            data = client.post("/api/translate_chat", json=base_request).json()
        result = data["character_a"]
        assert result["input_hash"] is not None
        assert result["system_prompt_hash"] is not None
        assert result["output_hash"] is not None
        assert result["ipc_id"] is not None

    def test_ipc_hashes_are_hex_strings(self, client: TestClient, base_request: dict) -> None:
        """IPC hash fields are hex-encoded SHA-256 strings (64 chars)."""
        with _patch_renderer("She nods gravely."):
            data = client.post("/api/translate_chat", json=base_request).json()
        result = data["character_a"]
        for field in ("input_hash", "system_prompt_hash", "output_hash"):
            assert len(result[field]) == 64
            int(result[field], 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestTranslateChatFailures:
    """API error and validation failure paths."""

    def test_api_error_status_when_renderer_returns_none(
        self, client: TestClient, base_request: dict
    ) -> None:
        """When the renderer returns None (Ollama unreachable), status is api_error."""
        with _patch_renderer(None):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert data["character_a"]["status"] == "fallback.api_error"

    def test_api_error_ic_text_is_none(self, client: TestClient, base_request: dict) -> None:
        with _patch_renderer(None):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert data["character_a"]["ic_text"] is None

    def test_api_error_output_hash_is_none(self, client: TestClient, base_request: dict) -> None:
        """output_hash and ipc_id are absent when there is no IC output."""
        with _patch_renderer(None):
            data = client.post("/api/translate_chat", json=base_request).json()
        result = data["character_a"]
        assert result["output_hash"] is None
        assert result["ipc_id"] is None

    def test_api_error_input_hash_still_present(
        self, client: TestClient, base_request: dict
    ) -> None:
        """input_hash and system_prompt_hash are computed before the Ollama call."""
        with _patch_renderer(None):
            data = client.post("/api/translate_chat", json=base_request).json()
        result = data["character_a"]
        assert result["input_hash"] is not None
        assert result["system_prompt_hash"] is not None

    def test_validation_failed_when_passthrough_returned(
        self, client: TestClient, base_request: dict
    ) -> None:
        """PASSTHROUGH output is rejected by OutputValidator → validation_failed."""
        req = {**base_request, "strict_mode": True}
        with _patch_renderer("PASSTHROUGH"):
            data = client.post("/api/translate_chat", json=req).json()
        assert data["character_a"]["status"] == "fallback.validation_failed"
        assert data["character_a"]["ic_text"] is None

    def test_validation_failed_multi_line_strict(
        self, client: TestClient, base_request: dict
    ) -> None:
        """Multi-line output in strict mode → validation_failed."""
        req = {**base_request, "strict_mode": True}
        with _patch_renderer("Line one.\nLine two."):
            data = client.post("/api/translate_chat", json=req).json()
        assert data["character_a"]["status"] == "fallback.validation_failed"

    def test_lenient_mode_recovers_multi_line(self, client: TestClient, base_request: dict) -> None:
        """Multi-line output in lenient mode → first line extracted → success."""
        req = {**base_request, "strict_mode": False}
        with _patch_renderer("First line of dialogue.\nSome explanation."):
            data = client.post("/api/translate_chat", json=req).json()
        assert data["character_a"]["status"] == "success"
        assert data["character_a"]["ic_text"] == "First line of dialogue."


# ---------------------------------------------------------------------------
# Active axes filtering
# ---------------------------------------------------------------------------


class TestActiveAxesFiltering:
    """active_axes controls which axes appear in the rendered profile."""

    def test_all_axes_active_when_active_axes_is_null(
        self, client: TestClient, axes_a: dict
    ) -> None:
        """When active_axes is not provided, all axes are included."""
        req = {
            "character_a": {
                "axes": axes_a,
                "ooc_message": "test",
                "channel": "say",
                # active_axes omitted → defaults to None → all axes active
            },
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 42,
        }
        with _patch_renderer("ok") as mock_render:
            client.post("/api/translate_chat", json=req)
        # Verify render was called (i.e. no early rejection)
        mock_render.assert_called_once()

    def test_single_active_axis_only_that_axis_in_profile(self, client: TestClient) -> None:
        """With active_axes=['health'], the 'age' axis is excluded from the
        rendered profile summary injected into the system prompt."""
        req = {
            "character_a": {
                "axes": {
                    "health": {"label": "weary", "score": 0.3},
                    "age": {"label": "old", "score": 0.75},
                },
                "ooc_message": "test",
                "channel": "say",
                "active_axes": ["health"],  # explicitly only 'health'
            },
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 42,
        }

        captured_prompts: list[str] = []

        def capture_render(self_inner, system_prompt: str, user_message: str):
            captured_prompts.append(system_prompt)
            return "test output"

        with patch("app.chat_renderer.ChatRenderer.render", capture_render):
            resp = client.post("/api/translate_chat", json=req)

        assert resp.status_code == 200
        assert len(captured_prompts) == 1
        sp = captured_prompts[0]
        # profile_summary lines have the format "  axis_name: label (score: N.NNN)"
        # 'health' is active → its profile line must appear in the system prompt
        assert "health: weary" in sp
        # 'age' is inactive → its profile line must NOT appear in the profile summary.
        # We check for the score-bearing form because the bare word "age" may appear
        # anywhere in the static template text.
        assert "age: old (score:" not in sp

    def test_empty_active_axes_produces_no_profile(self, client: TestClient) -> None:
        """active_axes=[] disables all axes; profile_summary says no axes active."""
        req = {
            "character_a": {
                "axes": {
                    "health": {"label": "weary", "score": 0.3},
                },
                "ooc_message": "test",
                "channel": "say",
                "active_axes": [],
            },
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 42,
        }

        captured_prompts: list[str] = []

        def capture_render(self_inner, system_prompt: str, user_message: str):
            captured_prompts.append(system_prompt)
            return "test output"

        with patch("app.chat_renderer.ChatRenderer.render", capture_render):
            client.post("/api/translate_chat", json=req)

        assert len(captured_prompts) == 1
        assert "(no axes active)" in captured_prompts[0]


# ---------------------------------------------------------------------------
# System prompt handling
# ---------------------------------------------------------------------------


class TestSystemPromptHandling:
    """Inline system_prompt, prompt_name, and default fallback."""

    def test_inline_system_prompt_used(self, client: TestClient, base_request: dict) -> None:
        """When system_prompt is provided inline, it is used as the template."""
        req = {
            **base_request,
            "system_prompt": "Translate the user's OOC message using this profile.",
        }

        captured: list[tuple[str, str]] = []

        def capture_render(self_inner, system_prompt: str, user_message: str):
            captured.append((system_prompt, user_message))
            return "ok"

        with patch("app.chat_renderer.ChatRenderer.render", capture_render):
            client.post("/api/translate_chat", json=req)

        assert len(captured) == 1
        # Inline prompt text is passed through unchanged; the OOC input stays in
        # the user turn rather than being injected into the system prompt.
        assert captured[0][0] == req["system_prompt"]
        assert captured[0][1] == base_request["character_a"]["ooc_message"]

    def test_prompt_name_404_raises_error(self, client: TestClient, base_request: dict) -> None:
        """A non-existent prompt_name returns a 404 from the endpoint."""
        req = {**base_request, "prompt_name": "does_not_exist_at_all"}
        with _patch_renderer("ok"):
            resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 404

    def test_prompt_name_ic_v01_loads(self, client: TestClient, base_request: dict) -> None:
        """prompt_name='ic_v01_undertaking' loads and substitutes placeholders."""
        req = {**base_request, "prompt_name": "ic_v01_undertaking"}
        with _patch_renderer("ok") as mock_render:
            resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 200
        mock_render.assert_called_once()


# ---------------------------------------------------------------------------
# Pydantic validation (422 responses)
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """Invalid request bodies return 422 Unprocessable Entity."""

    def test_missing_model_returns_422(self, client: TestClient, axes_a: dict) -> None:
        req = {
            "character_a": {
                "axes": axes_a,
                "ooc_message": "test",
            },
            # model is required
            "seed": 42,
        }
        resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 422

    def test_missing_seed_returns_422(self, client: TestClient, axes_a: dict) -> None:
        req = {
            "character_a": {"axes": axes_a, "ooc_message": "test"},
            "model": "gemma2:2b",
            # seed is required
        }
        resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 422

    def test_missing_character_a_returns_422(self, client: TestClient) -> None:
        req = {"model": "gemma2:2b", "seed": 42}
        resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 422

    def test_empty_ooc_message_returns_422(self, client: TestClient, axes_a: dict) -> None:
        """ooc_message has min_length=1; empty string → 422."""
        req = {
            "character_a": {"axes": axes_a, "ooc_message": ""},
            "model": "gemma2:2b",
            "seed": 42,
        }
        resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 422

    def test_temperature_out_of_range_returns_422(
        self, client: TestClient, base_request: dict
    ) -> None:
        """temperature has ge=0, le=2."""
        req = {**base_request, "temperature": 3.5}
        resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 422

    def test_max_tokens_below_minimum_returns_422(
        self, client: TestClient, base_request: dict
    ) -> None:
        """max_tokens has ge=10."""
        req = {**base_request, "max_tokens": 5}
        resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Seed forwarding
# ---------------------------------------------------------------------------


class TestSeedForwarding:
    """The seed in the request body is forwarded to ChatRenderer."""

    def test_seed_forwarded_to_renderer(self, client: TestClient, base_request: dict) -> None:
        """ChatRenderer is constructed with the seed from the request."""
        req = {**base_request, "seed": 99999}
        renderer_seeds: list[int | None] = []

        original_init = __import__(
            "app.chat_renderer", fromlist=["ChatRenderer"]
        ).ChatRenderer.__init__

        def capture_init(self_inner, *, seed, **kwargs):
            renderer_seeds.append(seed)
            original_init(self_inner, seed=seed, **kwargs)

        with patch("app.chat_renderer.ChatRenderer.__init__", capture_init):
            with _patch_renderer("ok"):
                client.post("/api/translate_chat", json=req)

        assert renderer_seeds == [99999]


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestResponseStructure:
    """Verify the top-level structure and field names of the response."""

    def test_response_has_character_a_key(self, client: TestClient, base_request: dict) -> None:
        with _patch_renderer("ok"):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert "character_a" in data

    def test_response_has_character_b_key(self, client: TestClient, base_request: dict) -> None:
        with _patch_renderer("ok"):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert "character_b" in data  # present but null when B not requested

    def test_result_has_required_fields(self, client: TestClient, base_request: dict) -> None:
        """Each ChatTranslationResult must have all defined fields."""
        with _patch_renderer("ok"):
            data = client.post("/api/translate_chat", json=base_request).json()
        result = data["character_a"]
        for field in (
            "ic_text",
            "status",
            "input_hash",
            "system_prompt_hash",
            "output_hash",
            "ipc_id",
        ):
            assert field in result, f"Missing field: {field}"

    def test_status_is_one_of_three_values(self, client: TestClient, base_request: dict) -> None:
        valid_statuses = {"success", "fallback.api_error", "fallback.validation_failed"}
        with _patch_renderer("ok"):
            data = client.post("/api/translate_chat", json=base_request).json()
        assert data["character_a"]["status"] in valid_statuses


# ---------------------------------------------------------------------------
# Optional character_a (live mode support)
# ---------------------------------------------------------------------------


class TestOptionalCharacterA:
    """character_a is now optional when character_b is provided."""

    def test_character_a_optional_with_b_only(self, client: TestClient, axes_b: dict) -> None:
        """POST with character_a=None returns character_b result and character_a=None."""
        req = {
            "character_a": None,
            "character_b": {
                "axes": axes_b,
                "ooc_message": "I scan the horizon.",
                "channel": "say",
            },
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 99,
        }
        with _patch_renderer("Scanning the horizon intently."):
            resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["character_a"] is None
        assert data["character_b"] is not None
        assert data["character_b"]["status"] == "success"
        assert data["character_b"]["ic_text"] == "Scanning the horizon intently."

    def test_neither_character_raises_422(self, client: TestClient) -> None:
        """POST with both character_a and character_b as None returns 422."""
        req = {
            "character_a": None,
            "character_b": None,
            "model": "gemma2:2b",
            "temperature": 0.7,
            "max_tokens": 128,
            "seed": 0,
        }
        resp = client.post("/api/translate_chat", json=req)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Server-mode translation (via MudServerClient)
# ---------------------------------------------------------------------------


def _mock_mud_client(*, authenticated: bool = True, world_id: str = "pipeworks_web") -> MagicMock:
    """Create a mock MudServerClient for server-mode tests."""
    mock = MagicMock()
    mock.is_authenticated = authenticated
    mock.selected_world_id = world_id
    return mock


class TestServerModeTranslation:
    """Translation delegated to mud server when client is authenticated."""

    def test_server_mode_returns_success(self, client: TestClient, base_request: dict) -> None:
        """When mud client is authenticated, translation goes through server."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": "She peers cautiously about the chamber.",
            "status": "success",
            "profile_summary": "health: weary (score: 0.3)\nage: old (score: 0.75)",
            "rendered_prompt": "You are a narrative engine. Profile:\n  health: weary",
            "model": "gemma2:2b",
            "world_config": {"world_id": "pipeworks_web"},
        }

        with patch("app.main.get_mud_client", return_value=mock):
            resp = client.post("/api/translate_chat", json=base_request)

        assert resp.status_code == 200
        data = resp.json()
        assert data["character_a"]["status"] == "success"
        assert data["character_a"]["ic_text"] == "She peers cautiously about the chamber."

    def test_server_mode_ipc_hashes_present(self, client: TestClient, base_request: dict) -> None:
        """IPC hashes are recomputed from server's rendered_prompt."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": "A weathered figure nods.",
            "status": "success",
            "profile_summary": "test",
            "rendered_prompt": "System prompt for IPC hashing.",
            "model": "gemma2:2b",
            "world_config": {},
        }

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        result = data["character_a"]
        assert result["input_hash"] is not None
        assert result["system_prompt_hash"] is not None
        assert result["output_hash"] is not None
        assert result["ipc_id"] is not None
        # Verify they are valid 64-char hex strings
        for field in ("input_hash", "system_prompt_hash", "output_hash"):
            assert len(result[field]) == 64
            int(result[field], 16)

    def test_server_mode_api_error_from_server(
        self, client: TestClient, base_request: dict
    ) -> None:
        """Server returns api_error status → propagated to response."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": None,
            "status": "fallback.api_error",
            "profile_summary": "",
            "rendered_prompt": "",
            "model": "gemma2:2b",
            "world_config": {},
        }

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["status"] == "fallback.api_error"
        assert data["character_a"]["ic_text"] is None

    def test_server_mode_connection_error_fallback(
        self, client: TestClient, base_request: dict
    ) -> None:
        """MudServerConnectionError → fallback.api_error in result."""
        mock = _mock_mud_client()
        mock.translate.side_effect = MudServerConnectionError("unreachable")

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["status"] == "fallback.api_error"

    def test_server_mode_session_expired_returns_401(
        self, client: TestClient, base_request: dict
    ) -> None:
        """MudServerSessionExpiredError during translate → 401 HTTP response."""
        mock = _mock_mud_client()
        mock.translate.side_effect = MudServerSessionExpiredError("expired")

        with patch("app.main.get_mud_client", return_value=mock):
            resp = client.post("/api/translate_chat", json=base_request)

        assert resp.status_code == 401


class TestServerModePromptOverride:
    """Server mode forwards system_prompt as prompt_template_override."""

    def test_system_prompt_forwarded_as_override(
        self, client: TestClient, base_request: dict
    ) -> None:
        """When system_prompt is set, it is passed as prompt_template_override."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": "She nods.",
            "status": "success",
            "profile_summary": "test",
            "rendered_prompt": "Custom prompt rendered",
            "model": "gemma2:2b",
            "world_config": {},
        }
        request = {**base_request, "system_prompt": "Custom prompt: {{profile_summary}}"}

        with patch("app.main.get_mud_client", return_value=mock):
            resp = client.post("/api/translate_chat", json=request)

        assert resp.status_code == 200
        call_kwargs = mock.translate.call_args[1]
        assert call_kwargs["prompt_template_override"] == "Custom prompt: {{profile_summary}}"

    def test_no_system_prompt_sends_none_override(
        self, client: TestClient, base_request: dict
    ) -> None:
        """When system_prompt is absent, prompt_template_override is None."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": "She nods.",
            "status": "success",
            "profile_summary": "test",
            "rendered_prompt": "Default prompt",
            "model": "gemma2:2b",
            "world_config": {},
        }

        with patch("app.main.get_mud_client", return_value=mock):
            resp = client.post("/api/translate_chat", json=base_request)

        assert resp.status_code == 200
        call_kwargs = mock.translate.call_args[1]
        assert call_kwargs["prompt_template_override"] is None


class TestServerModeGuards:
    """Server mode rejects requests when auth or world selection is missing."""

    def test_unauthenticated_returns_401(self, client: TestClient, base_request: dict) -> None:
        """Unauthenticated client in server mode → 401, not silent fallback."""
        mock = _mock_mud_client(authenticated=False)

        with patch("app.main.get_mud_client", return_value=mock):
            resp = client.post("/api/translate_chat", json=base_request)

        assert resp.status_code == 401
        mock.translate.assert_not_called()

    def test_no_world_selected_returns_400(self, client: TestClient, base_request: dict) -> None:
        """Authenticated but no world selected anywhere → 400."""
        mock = _mock_mud_client(authenticated=True, world_id=None)

        with patch("app.main.get_mud_client", return_value=mock):
            resp = client.post("/api/translate_chat", json=base_request)

        assert resp.status_code == 400
        mock.translate.assert_not_called()

    def test_world_id_in_request_overrides_missing_selection(
        self, client: TestClient, base_request: dict
    ) -> None:
        """world_id in request body rescues a lost in-memory selection."""
        mock = _mock_mud_client(authenticated=True, world_id=None)
        mock.translate.return_value = {
            "ic_text": "She nods.",
            "status": "success",
            "profile_summary": "",
            "rendered_prompt": "prompt",
            "model": "gemma2:2b",
            "world_config": {},
        }
        req = {**base_request, "world_id": "pipeworks_web"}

        with patch("app.main.get_mud_client", return_value=mock):
            resp = client.post("/api/translate_chat", json=req)

        assert resp.status_code == 200
        assert resp.json()["character_a"]["status"] == "success"
        mock.translate.assert_called_once()

    def test_none_client_uses_standalone(self, client: TestClient, base_request: dict) -> None:
        """When get_mud_client returns None (standalone mode), Ollama pipeline runs."""
        with (
            patch("app.main.get_mud_client", return_value=None),
            _patch_renderer("She nods."),
        ):
            resp = client.post("/api/translate_chat", json=base_request)

        assert resp.status_code == 200
        assert resp.json()["character_a"]["status"] == "success"


# ---------------------------------------------------------------------------
# error_detail field (Issue 1)
# ---------------------------------------------------------------------------


class TestErrorDetail:
    """error_detail is populated on server-mode error paths."""

    def test_connection_error_has_error_detail(
        self, client: TestClient, base_request: dict
    ) -> None:
        """MudServerConnectionError → error_detail explains the cause."""
        mock = _mock_mud_client()
        mock.translate.side_effect = MudServerConnectionError("unreachable")

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["error_detail"] == "Cannot connect to mud server."

    def test_generic_exception_has_error_detail(
        self, client: TestClient, base_request: dict
    ) -> None:
        """Unexpected exception → error_detail includes the exception class."""
        mock = _mock_mud_client()
        mock.translate.side_effect = RuntimeError("boom")

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["error_detail"] == "Server error: RuntimeError"

    def test_server_returns_failure_has_error_detail(
        self, client: TestClient, base_request: dict
    ) -> None:
        """Server returns non-success status → error_detail about model loading."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": None,
            "status": "fallback.api_error",
            "rendered_prompt": "",
            "model": "gemma2:2b",
        }

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["error_detail"] == (
            "Remote translation failed — server returned status 'fallback.api_error'."
            " The model may still be loading in Ollama."
        )

    def test_standalone_success_has_no_error_detail(
        self, client: TestClient, base_request: dict
    ) -> None:
        """Standalone success path → error_detail is None."""
        with _patch_renderer("She nods."):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["error_detail"] is None


# ---------------------------------------------------------------------------
# model field in result (Issue 4)
# ---------------------------------------------------------------------------


class TestModelField:
    """model field is populated in ChatTranslationResult."""

    def test_standalone_success_returns_model(self, client: TestClient, base_request: dict) -> None:
        """Standalone success → model echoes request model."""
        with _patch_renderer("She nods."):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["model"] == "gemma2:2b"

    def test_standalone_api_error_returns_model(
        self, client: TestClient, base_request: dict
    ) -> None:
        """Standalone api_error → model still echoes request model."""
        with _patch_renderer(None):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["model"] == "gemma2:2b"

    def test_server_mode_returns_server_model(self, client: TestClient, base_request: dict) -> None:
        """Server mode → model comes from server response, not request."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": "She nods.",
            "status": "success",
            "rendered_prompt": "prompt",
            "model": "llama3:8b",
        }

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        assert data["character_a"]["model"] == "llama3:8b"

    def test_server_mode_ipc_uses_server_model(
        self, client: TestClient, base_request: dict
    ) -> None:
        """IPC ID is computed with the server-returned model, not req.model."""
        mock = _mock_mud_client()
        mock.translate.return_value = {
            "ic_text": "She nods.",
            "status": "success",
            "rendered_prompt": "prompt",
            "model": "llama3:8b",  # different from request's gemma2:2b
        }

        with patch("app.main.get_mud_client", return_value=mock):
            data = client.post("/api/translate_chat", json=base_request).json()

        # The IPC ID should contain the server model, not the request model.
        # IPC ID format: input:prompt:model:temp:tokens:seed
        ipc_id = data["character_a"]["ipc_id"]
        assert ipc_id is not None
        # The model component is the 3rd segment (0-indexed: 2)
        # Model is hashed, so we verify indirectly: changing the model
        # should change the IPC ID.
        mock2 = _mock_mud_client()
        mock2.translate.return_value = {
            "ic_text": "She nods.",
            "status": "success",
            "rendered_prompt": "prompt",
            "model": "gemma2:2b",  # same as request model
        }

        with patch("app.main.get_mud_client", return_value=mock2):
            data2 = client.post("/api/translate_chat", json=base_request).json()

        ipc_id2 = data2["character_a"]["ipc_id"]
        # Different model → different IPC ID
        assert ipc_id != ipc_id2

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
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
            "system_prompt": "Translate: {{ooc_message}}",
        }

        captured: list[str] = []

        def capture_render(self_inner, system_prompt: str, user_message: str):
            captured.append(system_prompt)
            return "ok"

        with patch("app.chat_renderer.ChatRenderer.render", capture_render):
            client.post("/api/translate_chat", json=req)

        assert len(captured) == 1
        # ooc_message placeholder should be substituted with the actual message
        assert "I look around the room." in captured[0]

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

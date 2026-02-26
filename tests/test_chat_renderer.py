"""
Tests for app/chat_renderer.py — unified synchronous Ollama HTTP client.

All tests mock ``httpx.Client`` so no real HTTP connections are made.

Key behaviours verified:
  render():
    - Happy path: message.content extracted and stripped.
    - Empty / whitespace-only content returns None.
    - All three network failure paths (timeout, connect error, generic) return None.
    - Seed omitted from options when seed=None.
    - Seed included in options when seed is an integer.
    - Request body structure: model, stream=False, messages array, options.

  generate():
    - Happy path: (text, usage) returned; text is stripped.
    - Missing 'message' key → ValueError.
    - Missing usage fields → None values in usage dict.
    - Non-2xx response → httpx.HTTPStatusError propagates.
    - Seed forwarded in options when provided.
    - seed=0 (falsey) still forwarded.
    - stream: False in body.
    - Messages array: system first, user second.

  list_models():
    - Returns sorted names.
    - Empty models list handled.
    - Connection error → [] + warning logged.
    - Models without 'name' key skipped.
    - Timeout → [] + warning logged.
    - HTTP error → [] + warning logged.
    - Custom host used in URL.
    - Trailing slash on custom host stripped.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.chat_renderer import ChatRenderer

# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_response(message_content: str, status_code: int = 200) -> httpx.Response:
    """Build a mock Ollama /api/chat response with the given message content."""
    return httpx.Response(
        status_code=status_code,
        json={
            "model": "gemma2:2b",
            "message": {"role": "assistant", "content": message_content},
            "done": True,
        },
        request=httpx.Request("POST", "http://test/api/chat"),
    )


def _make_renderer(**kwargs) -> ChatRenderer:
    """Construct a ChatRenderer with sensible defaults; override via kwargs."""
    defaults = dict(
        host="http://localhost:11434",
        model="gemma2:2b",
        temperature=0.7,
        seed=42,
        max_tokens=128,
    )
    defaults.update(kwargs)
    return ChatRenderer(**defaults)


def _patch_client(mock_response: httpx.Response):
    """Context manager: patches httpx.Client and configures it to return mock_response."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = lambda s: s
    mock_ctx.__exit__ = lambda s, *a: None
    mock_ctx.post.return_value = mock_response
    return patch("app.chat_renderer.httpx.Client", return_value=mock_ctx)


# ── render() — Happy path ─────────────────────────────────────────────────────


class TestChatRendererHappyPath:
    """Successful /api/chat responses are correctly parsed."""

    def test_returns_message_content(self) -> None:
        renderer = _make_renderer()
        with _patch_client(_make_response("Hello, traveller.")):
            result = renderer.render("system prompt", "say hi")
        assert result == "Hello, traveller."

    def test_content_is_stripped(self) -> None:
        """Leading and trailing whitespace in message.content is stripped."""
        renderer = _make_renderer()
        with _patch_client(_make_response("  Hello there.  ")):
            result = renderer.render("sys", "hi")
        assert result == "Hello there."

    def test_empty_content_returns_none(self) -> None:
        """Empty string content (after stripping) returns None, not empty string."""
        renderer = _make_renderer()
        with _patch_client(_make_response("")):
            result = renderer.render("sys", "hi")
        assert result is None

    def test_whitespace_only_content_returns_none(self) -> None:
        """Whitespace-only content collapses to empty string → None."""
        renderer = _make_renderer()
        with _patch_client(_make_response("   \t  ")):
            result = renderer.render("sys", "hi")
        assert result is None

    def test_missing_message_key_returns_none(self) -> None:
        """Ollama response without 'message' key → empty dict → empty content → None."""
        mock_resp = httpx.Response(
            status_code=200,
            json={"done": True},  # no 'message' key
            request=httpx.Request("POST", "http://test/api/chat"),
        )
        renderer = _make_renderer()
        with _patch_client(mock_resp):
            result = renderer.render("sys", "hi")
        assert result is None


# ── render() — Network failure paths ─────────────────────────────────────────


class TestNetworkFailures:
    """All network failures return None; warnings are logged."""

    def test_timeout_returns_none(self) -> None:
        renderer = _make_renderer()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.side_effect = httpx.TimeoutException("read timed out")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            result = renderer.render("sys", "msg")
        assert result is None

    def test_timeout_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        renderer = _make_renderer()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.side_effect = httpx.TimeoutException("read timed out")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            with caplog.at_level(logging.WARNING, logger="app.chat_renderer"):
                renderer.render("sys", "msg")
        assert any("timed out" in r.message for r in caplog.records)

    def test_connect_error_returns_none(self) -> None:
        renderer = _make_renderer()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.side_effect = httpx.ConnectError("connection refused")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            result = renderer.render("sys", "msg")
        assert result is None

    def test_connect_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        renderer = _make_renderer()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.side_effect = httpx.ConnectError("connection refused")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            with caplog.at_level(logging.WARNING, logger="app.chat_renderer"):
                renderer.render("sys", "msg")
        assert any("cannot connect" in r.message for r in caplog.records)

    def test_http_status_error_returns_none(self) -> None:
        """Non-2xx response triggers raise_for_status → captured by generic handler."""
        mock_resp = httpx.Response(
            status_code=404,
            text="model not found",
            request=httpx.Request("POST", "http://test/api/chat"),
        )
        renderer = _make_renderer()
        with _patch_client(mock_resp):
            result = renderer.render("sys", "msg")
        assert result is None

    def test_unexpected_exception_returns_none(self) -> None:
        """Unexpected exceptions (e.g. JSON decode error) return None, not raise."""
        renderer = _make_renderer()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.side_effect = ValueError("unexpected json error")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            result = renderer.render("sys", "msg")
        assert result is None

    def test_unexpected_exception_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        renderer = _make_renderer()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.side_effect = RuntimeError("boom")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            with caplog.at_level(logging.ERROR, logger="app.chat_renderer"):
                renderer.render("sys", "msg")
        assert any(r.levelno == logging.ERROR for r in caplog.records)


# ── render() — Seed handling ──────────────────────────────────────────────────


class TestSeedHandling:
    """Seed is forwarded to Ollama options.seed when provided; omitted when None."""

    def _get_posted_body(self, renderer: ChatRenderer) -> dict:
        """Render a request and return the posted JSON body."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.return_value = _make_response("ok")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            renderer.render("system", "user")
        call_args = mock_ctx.post.call_args
        return call_args.kwargs.get("json") or call_args[1].get("json")

    def test_seed_included_when_provided(self) -> None:
        renderer = _make_renderer(seed=12345)
        body = self._get_posted_body(renderer)
        assert body["options"]["seed"] == 12345

    def test_seed_omitted_when_none(self) -> None:
        """When seed=None, the 'seed' key must not appear in options at all."""
        renderer = _make_renderer(seed=None)
        body = self._get_posted_body(renderer)
        assert "seed" not in body["options"]

    def test_seed_zero_is_included(self) -> None:
        """seed=0 is a valid integer seed and must be forwarded."""
        renderer = _make_renderer(seed=0)
        body = self._get_posted_body(renderer)
        assert body["options"]["seed"] == 0


# ── render() — Request body structure ────────────────────────────────────────


class TestRequestBodyStructure:
    """The JSON body posted to Ollama must match the /api/chat specification."""

    def _get_posted_body(
        self,
        system_prompt: str = "You are a test.",
        user_message: str = "Hello.",
        **renderer_kwargs,
    ) -> dict:
        renderer = _make_renderer(**renderer_kwargs)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.return_value = _make_response("ok")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            renderer.render(system_prompt, user_message)
        call_args = mock_ctx.post.call_args
        return call_args.kwargs.get("json") or call_args[1].get("json")

    def test_stream_is_false(self) -> None:
        """stream must be False to receive a single JSON response."""
        body = self._get_posted_body()
        assert body["stream"] is False

    def test_messages_array_has_two_entries(self) -> None:
        body = self._get_posted_body()
        assert len(body["messages"]) == 2

    def test_system_role_is_first(self) -> None:
        body = self._get_posted_body(system_prompt="sp")
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "sp"

    def test_user_role_is_second(self) -> None:
        body = self._get_posted_body(user_message="hello")
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "hello"

    def test_model_in_body(self) -> None:
        body = self._get_posted_body(model="llama3.2:1b")
        assert body["model"] == "llama3.2:1b"

    def test_temperature_in_options(self) -> None:
        body = self._get_posted_body(temperature=0.3)
        assert body["options"]["temperature"] == pytest.approx(0.3)

    def test_max_tokens_as_num_predict(self) -> None:
        body = self._get_posted_body(max_tokens=64)
        assert body["options"]["num_predict"] == 64

    def test_post_url_appends_api_chat(self) -> None:
        """The POST is sent to host + /api/chat."""
        renderer = _make_renderer(host="http://myhost:11434")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.return_value = _make_response("ok")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            renderer.render("sp", "msg")
        call_args = mock_ctx.post.call_args
        posted_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
        assert posted_url == "http://myhost:11434/api/chat"

    def test_trailing_slash_on_host_stripped(self) -> None:
        """A trailing slash on the host is normalised before appending /api/chat."""
        renderer = _make_renderer(host="http://myhost:11434/")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.return_value = _make_response("ok")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            renderer.render("sp", "msg")
        call_args = mock_ctx.post.call_args
        posted_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
        assert posted_url == "http://myhost:11434/api/chat"


# ── render() — Timeout configuration ─────────────────────────────────────────


class TestTimeoutConfiguration:
    """httpx.Timeout is configured with the provided read timeout + 10s connect."""

    def test_default_timeout_is_120s(self) -> None:
        renderer = ChatRenderer(host="http://localhost:11434", model="m")
        assert renderer._timeout.read == pytest.approx(120.0)
        assert renderer._timeout.connect == pytest.approx(10.0)

    def test_custom_timeout_applied(self) -> None:
        renderer = ChatRenderer(
            host="http://localhost:11434",
            model="m",
            timeout_seconds=60.0,
        )
        assert renderer._timeout.read == pytest.approx(60.0)
        assert renderer._timeout.connect == pytest.approx(10.0)


# ── generate() ────────────────────────────────────────────────────────────────


def _make_generate_response(
    message_content: str,
    prompt_eval_count: int | None = 100,
    eval_count: int | None = 25,
    status_code: int = 200,
) -> httpx.Response:
    body: dict = {
        "model": "gemma2:2b",
        "message": {"role": "assistant", "content": message_content},
        "done": True,
    }
    if prompt_eval_count is not None:
        body["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        body["eval_count"] = eval_count
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", "http://test/api/chat"),
    )


class TestGenerate:
    """generate() raises on failure and returns (text, usage) on success."""

    def _get_posted_body(self, renderer: ChatRenderer) -> dict:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.return_value = _make_generate_response("ok")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            renderer.generate("system", "user")
        call_args = mock_ctx.post.call_args
        return call_args.kwargs.get("json") or call_args[1].get("json")

    def test_returns_text_and_usage(self) -> None:
        renderer = _make_renderer()
        mock_resp = _make_generate_response("  A weathered figure.  ", 100, 25)
        with _patch_client(mock_resp):
            text, usage = renderer.generate("system prompt", '{"axes": {}}')
        assert text == "A weathered figure."
        assert usage["prompt_eval_count"] == 100
        assert usage["eval_count"] == 25

    def test_text_is_stripped(self) -> None:
        renderer = _make_renderer()
        mock_resp = _make_generate_response("  Hello.  ")
        with _patch_client(mock_resp):
            text, _ = renderer.generate("sys", "hi")
        assert text == "Hello."

    def test_missing_message_key_raises_value_error(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"done": True},  # no 'message' key
            request=httpx.Request("POST", "http://test/api/chat"),
        )
        renderer = _make_renderer()
        with _patch_client(mock_resp):
            with pytest.raises(ValueError, match="missing the 'message' key"):
                renderer.generate("sys", "hi")

    def test_missing_usage_fields_returns_none_values(self) -> None:
        """Usage fields absent from response → None in the dict."""
        mock_resp = httpx.Response(
            status_code=200,
            json={"model": "m", "message": {"role": "assistant", "content": "text"}, "done": True},
            request=httpx.Request("POST", "http://test/api/chat"),
        )
        renderer = _make_renderer()
        with _patch_client(mock_resp):
            _, usage = renderer.generate("sys", "hi")
        assert usage["prompt_eval_count"] is None
        assert usage["eval_count"] is None

    def test_http_error_propagates(self) -> None:
        mock_resp = httpx.Response(
            status_code=404,
            text="model not found",
            request=httpx.Request("POST", "http://test/api/chat"),
        )
        renderer = _make_renderer()
        with _patch_client(mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                renderer.generate("sys", "msg")

    def test_timeout_propagates(self) -> None:
        renderer = _make_renderer()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.post.side_effect = httpx.TimeoutException("read timed out")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            with pytest.raises(httpx.TimeoutException):
                renderer.generate("sys", "msg")

    def test_seed_forwarded_in_options(self) -> None:
        renderer = _make_renderer(seed=12345)
        body = self._get_posted_body(renderer)
        assert body["options"]["seed"] == 12345

    def test_seed_zero_forwarded(self) -> None:
        """seed=0 is falsey but must still be forwarded."""
        renderer = _make_renderer(seed=0)
        body = self._get_posted_body(renderer)
        assert body["options"]["seed"] == 0

    def test_seed_none_omitted(self) -> None:
        renderer = _make_renderer(seed=None)
        body = self._get_posted_body(renderer)
        assert "seed" not in body["options"]

    def test_stream_is_false(self) -> None:
        renderer = _make_renderer()
        body = self._get_posted_body(renderer)
        assert body["stream"] is False

    def test_system_message_first(self) -> None:
        renderer = _make_renderer()
        body = self._get_posted_body(renderer)
        assert body["messages"][0]["role"] == "system"

    def test_user_message_second(self) -> None:
        renderer = _make_renderer()
        body = self._get_posted_body(renderer)
        assert body["messages"][1]["role"] == "user"


# ── list_models() ─────────────────────────────────────────────────────────────


def _patch_get_client(mock_response: httpx.Response):
    """Context manager: patches httpx.Client for GET calls."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = lambda s: s
    mock_ctx.__exit__ = lambda s, *a: None
    mock_ctx.get.return_value = mock_response
    return patch("app.chat_renderer.httpx.Client", return_value=mock_ctx)


class TestListModels:
    """ChatRenderer.list_models() queries /api/tags and handles all failure cases."""

    def test_returns_sorted_names(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": [{"name": "llama3:8b"}, {"name": "gemma2:2b"}]},
            request=httpx.Request("GET", "http://test/api/tags"),
        )
        with _patch_get_client(mock_resp):
            result = ChatRenderer.list_models()
        assert result == ["gemma2:2b", "llama3:8b"]

    def test_empty_models_list(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": []},
            request=httpx.Request("GET", "http://test/api/tags"),
        )
        with _patch_get_client(mock_resp):
            assert ChatRenderer.list_models() == []

    def test_connection_error_returns_empty_list(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.get.side_effect = httpx.ConnectError("refused")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            assert ChatRenderer.list_models() == []

    def test_connection_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.get.side_effect = httpx.ConnectError("connection refused")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            with caplog.at_level(logging.WARNING, logger="app.chat_renderer"):
                result = ChatRenderer.list_models()
        assert result == []
        assert len(caplog.records) == 1
        assert "ConnectError" in caplog.records[0].message
        assert "connection refused" in caplog.records[0].message

    def test_models_without_name_key_skipped(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": [{"name": "valid:1b"}, {"size": 123}]},
            request=httpx.Request("GET", "http://test/api/tags"),
        )
        with _patch_get_client(mock_resp):
            assert ChatRenderer.list_models() == ["valid:1b"]

    def test_timeout_returns_empty_list(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.get.side_effect = httpx.TimeoutException("read timed out")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            assert ChatRenderer.list_models() == []

    def test_timeout_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.get.side_effect = httpx.TimeoutException("read timed out")
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            with caplog.at_level(logging.WARNING, logger="app.chat_renderer"):
                result = ChatRenderer.list_models()
        assert result == []
        assert len(caplog.records) == 1
        assert "TimeoutException" in caplog.records[0].message
        assert "read timed out" in caplog.records[0].message

    def test_http_error_returns_empty_list(self) -> None:
        mock_resp = httpx.Response(
            status_code=500,
            text="internal server error",
            request=httpx.Request("GET", "http://test/api/tags"),
        )
        with _patch_get_client(mock_resp):
            assert ChatRenderer.list_models() == []

    def test_http_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_resp = httpx.Response(
            status_code=500,
            text="internal server error",
            request=httpx.Request("GET", "http://test/api/tags"),
        )
        with _patch_get_client(mock_resp):
            with caplog.at_level(logging.WARNING, logger="app.chat_renderer"):
                result = ChatRenderer.list_models()
        assert result == []
        assert len(caplog.records) == 1
        assert "HTTPStatusError" in caplog.records[0].message

    def test_custom_host_used_in_url(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": [{"name": "gemma2:2b"}]},
            request=httpx.Request("GET", "http://custom:11434/api/tags"),
        )
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.get.return_value = mock_resp
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            result = ChatRenderer.list_models(host="http://custom:11434")
        call_args = mock_ctx.get.call_args
        get_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
        assert get_url == "http://custom:11434/api/tags"
        assert result == ["gemma2:2b"]

    def test_custom_host_trailing_slash_stripped(self) -> None:
        """Trailing slash on custom host is normalised."""
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": [{"name": "llama3:8b"}]},
            request=httpx.Request("GET", "http://myhost:11434/api/tags"),
        )
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = lambda s, *a: None
        mock_ctx.get.return_value = mock_resp
        with patch("app.chat_renderer.httpx.Client", return_value=mock_ctx):
            ChatRenderer.list_models(host="http://myhost:11434/")
        call_args = mock_ctx.get.call_args
        get_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
        assert get_url == "http://myhost:11434/api/tags"

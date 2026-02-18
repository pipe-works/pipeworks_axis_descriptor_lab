"""Tests for app/ollama_client.py – Ollama HTTP wrapper (mocked)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import httpx
import pytest

from app.ollama_client import list_local_models, ollama_generate

# ── ollama_generate ──────────────────────────────────────────────────────────


class TestOllamaGenerate:
    def _mock_response(self, json_data: dict, status_code: int = 200) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            json=json_data,
            request=httpx.Request("POST", "http://test/api/generate"),
        )

    def test_successful_generation(self) -> None:
        mock_resp = self._mock_response(
            {
                "response": "  A weathered figure stands.  ",
                "prompt_eval_count": 100,
                "eval_count": 25,
            }
        )

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            text, usage = ollama_generate(
                model="gemma2:2b",
                system_prompt="You are a test.",
                user_json_str='{"axes": {}}',
                temperature=0.2,
                max_tokens=120,
            )

        assert text == "A weathered figure stands."
        assert usage["prompt_eval_count"] == 100
        assert usage["eval_count"] == 25

    def test_missing_response_key_raises(self) -> None:
        mock_resp = self._mock_response({"model": "test", "done": True})

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            with pytest.raises(ValueError, match="missing the 'response' key"):
                ollama_generate(
                    model="test",
                    system_prompt="sp",
                    user_json_str="{}",
                    temperature=0.1,
                    max_tokens=50,
                )

    def test_missing_usage_fields_returns_none(self) -> None:
        mock_resp = self._mock_response({"response": "text"})

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            _, usage = ollama_generate(
                model="m",
                system_prompt="sp",
                user_json_str="{}",
                temperature=0.1,
                max_tokens=50,
            )

        assert usage["prompt_eval_count"] is None
        assert usage["eval_count"] is None

    def test_http_error_propagates(self) -> None:
        mock_resp = httpx.Response(
            status_code=404,
            text="model not found",
            request=httpx.Request("POST", "http://test/api/generate"),
        )

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            with pytest.raises(httpx.HTTPStatusError):
                ollama_generate(
                    model="missing",
                    system_prompt="sp",
                    user_json_str="{}",
                    temperature=0.1,
                    max_tokens=50,
                )


# ── list_local_models ────────────────────────────────────────────────────────


class TestListLocalModels:
    def test_returns_sorted_names(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": [{"name": "llama3:8b"}, {"name": "gemma2:2b"}]},
            request=httpx.Request("GET", "http://test/api/tags"),
        )

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.return_value = mock_resp

            result = list_local_models()

        assert result == ["gemma2:2b", "llama3:8b"]

    def test_empty_models_list(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": []},
            request=httpx.Request("GET", "http://test/api/tags"),
        )

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.return_value = mock_resp

            assert list_local_models() == []

    def test_connection_error_returns_empty_list(self) -> None:
        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.side_effect = httpx.ConnectError("refused")

            assert list_local_models() == []

    def test_models_without_name_key_skipped(self) -> None:
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": [{"name": "valid:1b"}, {"size": 123}]},
            request=httpx.Request("GET", "http://test/api/tags"),
        )

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.return_value = mock_resp

            assert list_local_models() == ["valid:1b"]

    def test_connection_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.side_effect = httpx.ConnectError("connection refused")

            with caplog.at_level(logging.WARNING, logger="app.ollama_client"):
                result = list_local_models()

        assert result == []
        assert len(caplog.records) == 1
        assert "ConnectError" in caplog.records[0].message
        assert "connection refused" in caplog.records[0].message

    def test_timeout_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.side_effect = httpx.TimeoutException("read timed out")

            with caplog.at_level(logging.WARNING, logger="app.ollama_client"):
                result = list_local_models()

        assert result == []
        assert len(caplog.records) == 1
        assert "TimeoutException" in caplog.records[0].message
        assert "read timed out" in caplog.records[0].message

    def test_http_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_resp = httpx.Response(
            status_code=500,
            text="internal server error",
            request=httpx.Request("GET", "http://test/api/tags"),
        )

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.return_value = mock_resp

            with caplog.at_level(logging.WARNING, logger="app.ollama_client"):
                result = list_local_models()

        assert result == []
        assert len(caplog.records) == 1
        assert "HTTPStatusError" in caplog.records[0].message

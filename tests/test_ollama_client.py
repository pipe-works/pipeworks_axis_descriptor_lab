"""
Tests for app/ollama_client.py – Ollama HTTP wrapper (mocked).

Every test that calls ``ollama_generate`` must supply the ``seed`` keyword
argument, which is forwarded to Ollama's ``options.seed`` for deterministic
token sampling.  Tests verify both the return value behaviour and the
structure of the outgoing HTTP request body.
"""

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
        """Happy path: valid Ollama response returns stripped text and usage."""
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
                seed=42,
            )

        assert text == "A weathered figure stands."
        assert usage["prompt_eval_count"] == 100
        assert usage["eval_count"] == 25

    def test_missing_response_key_raises(self) -> None:
        """Ollama response without a 'response' key raises ValueError."""
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
                    seed=42,
                )

    def test_missing_usage_fields_returns_none(self) -> None:
        """Ollama response without usage fields returns None for those keys."""
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
                seed=42,
            )

        assert usage["prompt_eval_count"] is None
        assert usage["eval_count"] is None

    def test_http_error_propagates(self) -> None:
        """Non-2xx Ollama response raises HTTPStatusError."""
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
                    seed=42,
                )

    def test_seed_included_in_request_body(self) -> None:
        """Verify the seed is forwarded in the Ollama options.seed field.

        This is the critical test for the seed fix: without ``options.seed``
        in the HTTP body, Ollama uses a random seed each call, making output
        non-deterministic even when all other IPC inputs are identical.
        """
        mock_resp = self._mock_response({"response": "deterministic output"})

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            ollama_generate(
                model="gemma2:2b",
                system_prompt="test prompt",
                user_json_str='{"axes": {}}',
                temperature=0.7,
                max_tokens=100,
                seed=12345,
            )

            # Inspect the JSON body that was POSTed to Ollama.
            call_args = mock_client_cls.return_value.post.call_args
            posted_body = call_args.kwargs.get("json") or call_args[1].get("json")

            assert "options" in posted_body
            assert posted_body["options"]["seed"] == 12345
            assert posted_body["options"]["temperature"] == 0.7
            assert posted_body["options"]["num_predict"] == 100

    def test_different_seeds_produce_different_request_bodies(self) -> None:
        """Two calls with different seeds must send different options.seed values.

        This doesn't test Ollama behaviour (that's Ollama's responsibility),
        but it confirms our wrapper correctly propagates distinct seeds.
        """
        mock_resp = self._mock_response({"response": "text"})
        posted_seeds: list[int] = []

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            for seed_val in [100, 200]:
                ollama_generate(
                    model="m",
                    system_prompt="sp",
                    user_json_str="{}",
                    temperature=0.1,
                    max_tokens=50,
                    seed=seed_val,
                )

            # Collect the seed from each POST call.
            for call in mock_client_cls.return_value.post.call_args_list:
                body = call.kwargs.get("json") or call[1].get("json")
                posted_seeds.append(body["options"]["seed"])

        assert posted_seeds == [100, 200]

    def test_custom_host_used_in_url(self) -> None:
        """When ``host`` is provided, the POST URL uses that host instead of the default."""
        mock_resp = self._mock_response({"response": "custom host"})

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            ollama_generate(
                model="gemma2:2b",
                system_prompt="sp",
                user_json_str="{}",
                temperature=0.2,
                max_tokens=50,
                seed=42,
                host="http://192.168.1.50:11434",
            )

            # Inspect the URL that was POSTed to.
            call_args = mock_client_cls.return_value.post.call_args
            posted_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
            assert posted_url == "http://192.168.1.50:11434/api/generate"

    def test_custom_host_trailing_slash_stripped(self) -> None:
        """A trailing slash on the host URL is stripped before building the endpoint URL."""
        mock_resp = self._mock_response({"response": "ok"})

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            ollama_generate(
                model="m",
                system_prompt="sp",
                user_json_str="{}",
                temperature=0.1,
                max_tokens=50,
                seed=42,
                host="http://myhost:11434/",
            )

            call_args = mock_client_cls.return_value.post.call_args
            posted_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
            assert posted_url == "http://myhost:11434/api/generate"

    def test_none_host_uses_default(self) -> None:
        """When ``host`` is None (default), the module-level ``_OLLAMA_HOST`` is used."""
        mock_resp = self._mock_response({"response": "default"})

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.post.return_value = mock_resp

            ollama_generate(
                model="m",
                system_prompt="sp",
                user_json_str="{}",
                temperature=0.1,
                max_tokens=50,
                seed=42,
                host=None,
            )

            call_args = mock_client_cls.return_value.post.call_args
            posted_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
            # Should use the default host (from env or localhost:11434).
            assert "/api/generate" in posted_url


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

    def test_custom_host_used_in_url(self) -> None:
        """When ``host`` is provided, the GET URL uses that host."""
        mock_resp = httpx.Response(
            status_code=200,
            json={"models": [{"name": "gemma2:2b"}]},
            request=httpx.Request("GET", "http://custom:11434/api/tags"),
        )

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.return_value = mock_resp

            result = list_local_models(host="http://custom:11434")

            call_args = mock_client_cls.return_value.get.call_args
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

        with patch("app.ollama_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = lambda s, *a: None
            mock_client_cls.return_value.get.return_value = mock_resp

            list_local_models(host="http://myhost:11434/")

            call_args = mock_client_cls.return_value.get.call_args
            get_url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
            assert get_url == "http://myhost:11434/api/tags"

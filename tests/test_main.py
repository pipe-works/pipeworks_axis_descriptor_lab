"""Tests for app/main.py – FastAPI routes and helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import _load_default_prompt, _load_example, _payload_hash
from app.schema import AxisPayload, AxisValue

# ── Helpers ──────────────────────────────────────────────────────────────────


class TestPayloadHash:
    def test_deterministic(self, sample_payload: AxisPayload) -> None:
        h1 = _payload_hash(sample_payload)
        h2 = _payload_hash(sample_payload)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_payloads_differ(self) -> None:
        p1 = AxisPayload(
            axes={"a": AxisValue(label="x", score=0.1)},
            policy_hash="h1",
            seed=1,
            world_id="w",
        )
        p2 = AxisPayload(
            axes={"a": AxisValue(label="x", score=0.2)},
            policy_hash="h1",
            seed=1,
            world_id="w",
        )
        assert _payload_hash(p1) != _payload_hash(p2)

    def test_order_independent(self) -> None:
        """Axes dict ordering must not affect hash."""
        axes_a = {
            "z": AxisValue(label="z", score=0.1),
            "a": AxisValue(label="a", score=0.9),
        }
        axes_b = {
            "a": AxisValue(label="a", score=0.9),
            "z": AxisValue(label="z", score=0.1),
        }
        p1 = AxisPayload(axes=axes_a, policy_hash="h", seed=1, world_id="w")
        p2 = AxisPayload(axes=axes_b, policy_hash="h", seed=1, world_id="w")
        assert _payload_hash(p1) == _payload_hash(p2)


class TestLoadDefaultPrompt:
    def test_loads_prompt(self) -> None:
        prompt = _load_default_prompt()
        assert "authoritative" in prompt.lower() or "ornamental" in prompt.lower()
        assert len(prompt) > 50

    def test_missing_prompt_raises(self, tmp_path: Path) -> None:
        with patch("app.main._PROMPTS_DIR", tmp_path):
            with pytest.raises(Exception):
                _load_default_prompt()


class TestLoadExample:
    def test_loads_example_a(self) -> None:
        data = _load_example("example_a")
        assert "axes" in data
        assert "seed" in data

    def test_missing_example_raises_404(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _load_example("nonexistent_example")
        assert exc_info.value.status_code == 404

    def test_invalid_json_raises_500(self, tmp_path: Path) -> None:
        from fastapi import HTTPException

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{", encoding="utf-8")
        with patch("app.main._EXAMPLES_DIR", tmp_path):
            with pytest.raises(HTTPException) as exc_info:
                _load_example("bad")
            assert exc_info.value.status_code == 500


# ── API Routes ───────────────────────────────────────────────────────────────


class TestIndexRoute:
    def test_returns_html(self, client: TestClient) -> None:
        with patch("app.main.list_local_models", return_value=["gemma2:2b"]):
            resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestListExamples:
    def test_returns_list(self, client: TestClient) -> None:
        resp = client.get("/api/examples")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert "example_a" in data
        assert "example_b" in data

    def test_sorted(self, client: TestClient) -> None:
        data = client.get("/api/examples").json()
        assert data == sorted(data)


class TestGetExample:
    def test_returns_example(self, client: TestClient) -> None:
        resp = client.get("/api/examples/example_a")
        assert resp.status_code == 200
        data = resp.json()
        assert "axes" in data
        assert data["world_id"] == "pipeworks_web"

    def test_missing_example_404(self, client: TestClient) -> None:
        resp = client.get("/api/examples/does_not_exist")
        assert resp.status_code == 404


class TestGetModels:
    def test_returns_models(self, client: TestClient) -> None:
        with patch("app.main.list_local_models", return_value=["gemma2:2b", "llama3:8b"]):
            resp = client.get("/api/models")
        assert resp.status_code == 200
        assert resp.json() == ["gemma2:2b", "llama3:8b"]

    def test_empty_when_ollama_down(self, client: TestClient) -> None:
        with patch("app.main.list_local_models", return_value=[]):
            resp = client.get("/api/models")
        assert resp.json() == []


class TestGenerateEndpoint:
    def _req_body(self, payload_dict: dict) -> dict:
        return {
            "payload": payload_dict,
            "model": "gemma2:2b",
            "temperature": 0.2,
            "max_tokens": 120,
        }

    def test_successful_generate(self, client: TestClient, sample_payload_dict: dict) -> None:
        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.return_value = (
                "A weathered figure.",
                {"prompt_eval_count": 50, "eval_count": 10},
            )
            resp = client.post("/api/generate", json=self._req_body(sample_payload_dict))

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "A weathered figure."
        assert data["model"] == "gemma2:2b"
        assert data["temperature"] == 0.2
        assert data["usage"]["eval_count"] == 10

    def test_custom_system_prompt(self, client: TestClient, sample_payload_dict: dict) -> None:
        body = self._req_body(sample_payload_dict)
        body["system_prompt"] = "Custom prompt"

        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.return_value = ("text", {})
            resp = client.post("/api/generate", json=body)
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["system_prompt"] == "Custom prompt"

        assert resp.status_code == 200

    def test_ollama_http_error_returns_502(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        import httpx

        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.side_effect = httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(404, text="model not found"),
            )
            resp = client.post("/api/generate", json=self._req_body(sample_payload_dict))

        assert resp.status_code == 502

    def test_ollama_timeout_returns_504(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        import httpx

        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.side_effect = httpx.ReadTimeout("timeout")
            resp = client.post("/api/generate", json=self._req_body(sample_payload_dict))

        assert resp.status_code == 504

    def test_ollama_value_error_returns_502(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.side_effect = ValueError("missing response key")
            resp = client.post("/api/generate", json=self._req_body(sample_payload_dict))

        assert resp.status_code == 502

    def test_unexpected_error_returns_500(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.side_effect = RuntimeError("something broke")
            resp = client.post("/api/generate", json=self._req_body(sample_payload_dict))

        assert resp.status_code == 500


class TestLogEndpoint:
    def test_creates_log_entry(self, client: TestClient, sample_payload_dict: dict) -> None:
        resp = client.post(
            "/api/log",
            params={
                "output": "some text",
                "model": "gemma2:2b",
                "temperature": "0.2",
                "max_tokens": "120",
            },
            json=sample_payload_dict,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["output"] == "some text"
        assert data["model"] == "gemma2:2b"
        assert "input_hash" in data
        assert "timestamp" in data
        assert len(data["input_hash"]) == 64


class TestRelabelEndpoint:
    def test_relabels_known_axes(self, client: TestClient) -> None:
        payload = {
            "axes": {
                "age": {"label": "placeholder", "score": 0.1},
                "health": {"label": "placeholder", "score": 0.9},
            },
            "policy_hash": "hash",
            "seed": 1,
            "world_id": "w",
        }
        resp = client.post("/api/relabel", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["axes"]["age"]["label"] == "young"
        assert data["axes"]["health"]["label"] == "failing"

    def test_preserves_unknown_axes(self, client: TestClient) -> None:
        payload = {
            "axes": {
                "custom_axis": {"label": "original", "score": 0.5},
            },
            "policy_hash": "hash",
            "seed": 1,
            "world_id": "w",
        }
        resp = client.post("/api/relabel", json=payload)
        data = resp.json()
        assert data["axes"]["custom_axis"]["label"] == "original"

    def test_preserves_scores(self, client: TestClient) -> None:
        payload = {
            "axes": {"wealth": {"label": "x", "score": 0.3}},
            "policy_hash": "h",
            "seed": 1,
            "world_id": "w",
        }
        resp = client.post("/api/relabel", json=payload)
        data = resp.json()
        assert data["axes"]["wealth"]["score"] == 0.3
        assert data["axes"]["wealth"]["label"] == "threadbare"

    def test_all_policy_axes(self, client: TestClient) -> None:
        """Verify every axis in the policy table produces a valid relabel."""
        axes_to_test = {
            "age": 0.6,
            "demeanor": 0.3,
            "dependency": 0.5,
            "facial_signal": 0.2,
            "health": 0.4,
            "legitimacy": 0.55,
            "moral_load": 0.8,
            "physique": 0.5,
            "risk_exposure": 0.9,
            "visibility": 0.1,
            "wealth": 0.6,
        }
        payload = {
            "axes": {k: {"label": "test", "score": v} for k, v in axes_to_test.items()},
            "policy_hash": "h",
            "seed": 1,
            "world_id": "w",
        }
        resp = client.post("/api/relabel", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Every axis should have been relabeled to something other than "test"
        for axis_name in axes_to_test:
            assert data["axes"][axis_name]["label"] != "test"

    def test_preserves_non_axis_fields(self, client: TestClient) -> None:
        payload = {
            "axes": {"age": {"label": "x", "score": 0.1}},
            "policy_hash": "keep_this",
            "seed": 999,
            "world_id": "my_world",
        }
        resp = client.post("/api/relabel", json=payload)
        data = resp.json()
        assert data["policy_hash"] == "keep_this"
        assert data["seed"] == 999
        assert data["world_id"] == "my_world"

    def test_boundary_scores(self, client: TestClient) -> None:
        """Test scores at exact policy boundaries."""
        # age: 0.25 → boundary between young and middle-aged
        payload = {
            "axes": {"age": {"label": "x", "score": 0.25}},
            "policy_hash": "h",
            "seed": 1,
            "world_id": "w",
        }
        resp = client.post("/api/relabel", json=payload)
        data = resp.json()
        # score 0.25 >= 0.25 threshold, so falls into next bucket (middle-aged)
        assert data["axes"]["age"]["label"] == "middle-aged"

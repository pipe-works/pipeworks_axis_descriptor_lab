"""Tests for app/main.py – FastAPI routes and helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import (
    _build_baseline_md,
    _build_output_md,
    _build_system_prompt_md,
    _load_default_prompt,
    _load_example,
    _payload_hash,
    _save_folder_name,
)
from app.schema import AxisPayload, AxisValue, SaveRequest

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

    def test_generate_with_large_seed(self, client: TestClient) -> None:
        """Frontend resolveSeed() can produce seeds up to 2^32-1."""
        payload = {
            "axes": {"health": {"label": "weary", "score": 0.5}},
            "policy_hash": "h",
            "seed": 4294967295,
            "world_id": "w",
        }
        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.return_value = ("text", {})
            resp = client.post("/api/generate", json=self._req_body(payload))

        assert resp.status_code == 200

    def test_generate_with_zero_seed(self, client: TestClient) -> None:
        """Seed 0 is a valid deterministic seed (not random)."""
        payload = {
            "axes": {"health": {"label": "weary", "score": 0.5}},
            "policy_hash": "h",
            "seed": 0,
            "world_id": "w",
        }
        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.return_value = ("text", {})
            resp = client.post("/api/generate", json=self._req_body(payload))

        assert resp.status_code == 200


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


# ── Save helpers ─────────────────────────────────────────────────────────────


class TestSaveFolderName:
    """Tests for the _save_folder_name() helper."""

    def test_format_matches_expected_pattern(self) -> None:
        """Folder name must be YYYYMMDD_HHMMSS_<8 hex chars>."""
        import re

        now = datetime(2026, 2, 18, 14, 30, 22, tzinfo=timezone.utc)
        hash_str = "d845cdcf" + "a" * 56  # 64-char hex string
        name = _save_folder_name(now, hash_str)

        assert name == "20260218_143022_d845cdcf"
        assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", name)

    def test_uses_first_eight_chars_of_hash(self) -> None:
        """Only the first 8 characters of the hash should appear."""
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        name = _save_folder_name(now, "abcdef01" + "0" * 56)
        assert name.endswith("_abcdef01")

    def test_different_hashes_produce_different_names(self) -> None:
        """Same timestamp but different hashes must produce different names."""
        now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        name_a = _save_folder_name(now, "aaaa" * 16)
        name_b = _save_folder_name(now, "bbbb" * 16)
        assert name_a != name_b


class TestBuildOutputMd:
    """Tests for the _build_output_md() Markdown builder."""

    def test_contains_text_and_provenance(self) -> None:
        """Output MD must include the generated text and provenance comments."""
        req = SaveRequest(
            payload=AxisPayload(
                axes={"health": AxisValue(label="weary", score=0.5)},
                policy_hash="abc",
                seed=42,
                world_id="w",
            ),
            model="gemma2:2b",
            temperature=0.2,
            max_tokens=120,
            system_prompt="You are a test prompt.",
            output="A weathered figure.",
        )
        now = datetime(2026, 2, 18, 14, 0, 0, tzinfo=timezone.utc)
        md = _build_output_md("A weathered figure.", req, now, "d845" + "0" * 60)

        assert "# Output" in md
        assert "A weathered figure." in md
        assert "gemma2:2b" in md
        assert "2026-02-18" in md
        assert "d845" in md


class TestBuildBaselineMd:
    """Tests for the _build_baseline_md() Markdown builder."""

    def test_contains_text_and_folder_ref(self) -> None:
        """Baseline MD must include the text and reference the save folder."""
        md = _build_baseline_md("Old description text.", "20260218_140000_abcd1234")

        assert "# Baseline (A)" in md
        assert "Old description text." in md
        assert "20260218_140000_abcd1234" in md


class TestBuildSystemPromptMd:
    """Tests for the _build_system_prompt_md() Markdown builder."""

    def test_contains_prompt_in_code_block(self) -> None:
        """System prompt MD must wrap the text in a fenced code block."""
        md = _build_system_prompt_md("You are a descriptive layer.", "20260218_test")

        assert "# System Prompt" in md
        assert "```text" in md
        assert "You are a descriptive layer." in md
        assert "20260218_test" in md


# ── GET /api/system-prompt ───────────────────────────────────────────────────


class TestGetSystemPromptEndpoint:
    """Tests for the GET /api/system-prompt endpoint."""

    def test_returns_prompt_text(self, client: TestClient) -> None:
        """Endpoint must return the default system prompt as text."""
        resp = client.get("/api/system-prompt")
        assert resp.status_code == 200
        # The prompt should contain meaningful content (not empty)
        assert len(resp.text) > 50

    def test_returns_plain_text_content_type(self, client: TestClient) -> None:
        """Response content-type must be text/plain."""
        resp = client.get("/api/system-prompt")
        assert "text/plain" in resp.headers["content-type"]


# ── POST /api/save ───────────────────────────────────────────────────────────


class TestSaveEndpoint:
    """Tests for the POST /api/save endpoint.

    All tests use ``tmp_path`` + ``patch("app.main._DATA_DIR", tmp_path)``
    to isolate file I/O and avoid polluting the real ``data/`` directory.
    """

    def test_creates_folder_and_core_files(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Happy path: saves metadata.json, payload.json, system_prompt.md,
        and output.md when output is provided."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        assert resp.status_code == 200
        data = resp.json()

        # Response fields
        assert "folder_name" in data
        assert "input_hash" in data
        assert len(data["input_hash"]) == 64
        assert "timestamp" in data
        assert "files" in data

        # Verify the folder and expected files exist on disk
        save_dir = tmp_path / data["folder_name"]
        assert save_dir.is_dir()
        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "payload.json").exists()
        assert (save_dir / "system_prompt.md").exists()
        assert (save_dir / "output.md").exists()

    def test_output_md_omitted_when_no_output(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """output.md must not be created when output is None."""
        body = {**save_request_body, "output": None}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        assert not (save_dir / "output.md").exists()
        assert "output.md" not in data["files"]

    def test_baseline_md_omitted_when_no_baseline(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """baseline.md must not be created when baseline is None."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        assert not (save_dir / "baseline.md").exists()
        assert "baseline.md" not in data["files"]

    def test_baseline_md_written_when_provided(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """baseline.md must be created and contain the text when provided."""
        body = {**save_request_body, "baseline": "The old description."}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        assert (save_dir / "baseline.md").exists()
        assert "baseline.md" in data["files"]
        content = (save_dir / "baseline.md").read_text(encoding="utf-8")
        assert "The old description." in content

    def test_metadata_json_contains_expected_fields(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """metadata.json must include all provenance fields."""
        import json as _json

        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = _json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        assert metadata["model"] == "gemma2:2b"
        assert metadata["temperature"] == 0.2
        assert metadata["max_tokens"] == 120
        assert metadata["seed"] == 42
        assert metadata["world_id"] == "test_world"
        assert metadata["policy_hash"] == "abc123"
        assert metadata["axis_count"] == 2  # health + age from fixture
        assert len(metadata["input_hash"]) == 64
        assert "timestamp" in metadata
        assert "folder_name" in metadata

    def test_payload_json_round_trips_cleanly(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """payload.json must contain the full payload with all axes."""
        import json as _json

        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        payload = _json.loads((save_dir / "payload.json").read_text(encoding="utf-8"))

        assert "axes" in payload
        assert "health" in payload["axes"]
        assert "age" in payload["axes"]
        assert payload["seed"] == 42
        assert payload["world_id"] == "test_world"

    def test_folder_name_format(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Folder name must match YYYYMMDD_HHMMSS_<8 hex chars> format."""
        import re

        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        folder_name = resp.json()["folder_name"]
        assert re.match(
            r"^\d{8}_\d{6}_[0-9a-f]{8}$", folder_name
        ), f"Unexpected folder name format: {folder_name}"

    def test_system_prompt_md_contains_prompt_text(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """system_prompt.md must contain the prompt in a fenced code block."""
        body = {
            **save_request_body,
            "system_prompt": "Custom test prompt text.",
        }
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        sp_text = (save_dir / "system_prompt.md").read_text(encoding="utf-8")
        assert "Custom test prompt text." in sp_text
        assert "```text" in sp_text

    def test_files_list_is_sorted(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """files list in response must be sorted alphabetically."""
        body = {**save_request_body, "baseline": "A baseline."}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        files = resp.json()["files"]
        assert files == sorted(files)

    def test_all_five_files_when_both_output_and_baseline(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """When both output and baseline are set, all 5 files must exist."""
        body = {**save_request_body, "baseline": "Baseline text."}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        assert data["files"] == [
            "baseline.md",
            "metadata.json",
            "output.md",
            "payload.json",
            "system_prompt.md",
        ]

    def test_invalid_payload_returns_422(self, client: TestClient, tmp_path: Path) -> None:
        """Malformed request body must return 422 Unprocessable Entity."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json={"payload": "not a payload"})
        assert resp.status_code == 422

    def test_empty_system_prompt_returns_422(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """An empty system_prompt string must be rejected (min_length=1)."""
        body = {**save_request_body, "system_prompt": ""}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)
        assert resp.status_code == 422

    def test_output_md_contains_generated_text(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """output.md must contain the actual generated text."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        content = (save_dir / "output.md").read_text(encoding="utf-8")
        assert "A weathered figure stands near the threshold." in content

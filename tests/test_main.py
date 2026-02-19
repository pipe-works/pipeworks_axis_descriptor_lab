"""Tests for app/main.py – FastAPI routes and helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

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


class TestListPrompts:
    """Tests for GET /api/prompts – list available prompt names."""

    def test_returns_list(self, client: TestClient) -> None:
        """Must return a list containing at least the default prompt."""
        resp = client.get("/api/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert "system_prompt_v01" in data

    def test_sorted(self, client: TestClient) -> None:
        """Prompt names must be returned in sorted order."""
        data = client.get("/api/prompts").json()
        assert data == sorted(data)

    def test_includes_variant_prompts(self, client: TestClient) -> None:
        """All prompt .txt files in app/prompts/ must appear in the list."""
        data = client.get("/api/prompts").json()
        assert len(data) >= 4  # v01 + v02_terse + v03_environmental + v04_contrast
        assert "system_prompt_v02_terse" in data
        assert "system_prompt_v03_environmental" in data
        assert "system_prompt_v04_contrast" in data


class TestGetPrompt:
    """Tests for GET /api/prompts/{name} – retrieve a single prompt's text."""

    def test_returns_prompt_text(self, client: TestClient) -> None:
        """Loading the default prompt must return non-empty text."""
        resp = client.get("/api/prompts/system_prompt_v01")
        assert resp.status_code == 200
        assert len(resp.text) > 50
        assert "ornamental" in resp.text.lower()

    def test_returns_plain_text_content_type(self, client: TestClient) -> None:
        """Response content-type must be text/plain (not JSON)."""
        resp = client.get("/api/prompts/system_prompt_v01")
        assert "text/plain" in resp.headers["content-type"]

    def test_missing_prompt_404(self, client: TestClient) -> None:
        """A non-existent prompt name must return 404."""
        resp = client.get("/api/prompts/does_not_exist")
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

    def test_custom_host_forwarded(self, client: TestClient) -> None:
        """The host query param is forwarded to list_local_models."""
        with patch("app.main.list_local_models", return_value=["gemma2:2b"]) as mock_list:
            resp = client.get("/api/models?host=http://remote:11434")
        assert resp.status_code == 200
        mock_list.assert_called_once_with(host="http://remote:11434")

    def test_no_host_param_passes_none(self, client: TestClient) -> None:
        """When host query param is omitted, None is passed to list_local_models."""
        with patch("app.main.list_local_models", return_value=[]) as mock_list:
            client.get("/api/models")
        mock_list.assert_called_once_with(host=None)


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

        # IPC hash fields must be present and well-formed (64-char hex)
        for key in ("input_hash", "system_prompt_hash", "output_hash", "ipc_id"):
            assert key in data, f"Missing IPC field: {key}"
            assert isinstance(data[key], str), f"{key} is not a string"
            assert len(data[key]) == 64, f"{key} is not 64 chars"
            int(data[key], 16)  # must be valid hex

    def test_generate_hashes_are_deterministic(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Two identical requests must produce identical IPC hashes."""
        body = self._req_body(sample_payload_dict)
        results = []
        for _ in range(2):
            with patch("app.main.ollama_generate") as mock_gen:
                mock_gen.return_value = ("Same output.", {})
                resp = client.post("/api/generate", json=body)
            results.append(resp.json())

        for key in ("input_hash", "system_prompt_hash", "output_hash", "ipc_id"):
            assert results[0][key] == results[1][key], f"{key} differs across identical requests"

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
        # Verify the large seed is forwarded to Ollama for deterministic sampling.
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["seed"] == 4294967295

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
        # Verify seed 0 is explicitly forwarded (not treated as "no seed").
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["seed"] == 0

    def test_seed_forwarded_to_ollama(self, client: TestClient, sample_payload_dict: dict) -> None:
        """The payload's seed must be passed to ollama_generate() as options.seed.

        This is the critical integration test for the seed fix: the seed was
        previously only used in the IPC hash but never forwarded to Ollama for
        deterministic token sampling.  Without this, identical IPC inputs
        could still produce different outputs.
        """
        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.return_value = ("deterministic text", {})
            resp = client.post("/api/generate", json=self._req_body(sample_payload_dict))

        assert resp.status_code == 200
        # The seed from the payload must appear in the ollama_generate kwargs.
        call_kwargs = mock_gen.call_args.kwargs
        assert "seed" in call_kwargs, "seed not forwarded to ollama_generate()"
        assert call_kwargs["seed"] == sample_payload_dict["seed"]

    def test_ollama_host_forwarded_to_ollama(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """When ollama_host is provided, it is forwarded to ollama_generate()."""
        body = self._req_body(sample_payload_dict)
        body["ollama_host"] = "http://remote:11434"

        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.return_value = ("text", {})
            resp = client.post("/api/generate", json=body)

        assert resp.status_code == 200
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["host"] == "http://remote:11434"

    def test_ollama_host_defaults_to_none(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """When ollama_host is omitted, host=None is passed to ollama_generate()."""
        with patch("app.main.ollama_generate") as mock_gen:
            mock_gen.return_value = ("text", {})
            resp = client.post("/api/generate", json=self._req_body(sample_payload_dict))

        assert resp.status_code == 200
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["host"] is None


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

        # output_hash is always present (output is required)
        assert data["output_hash"] is not None
        assert len(data["output_hash"]) == 64

    def test_log_with_system_prompt_includes_all_hashes(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """When system_prompt is provided, all three IPC hash fields must be set."""
        resp = client.post(
            "/api/log",
            params={
                "output": "some text",
                "model": "gemma2:2b",
                "temperature": "0.2",
                "max_tokens": "120",
                "system_prompt": "You are a descriptive layer.",
            },
            json=sample_payload_dict,
        )
        assert resp.status_code == 200
        data = resp.json()

        for key in ("system_prompt_hash", "output_hash", "ipc_id"):
            assert data[key] is not None, f"{key} should not be null"
            assert len(data[key]) == 64, f"{key} should be 64-char hex"
            int(data[key], 16)

    def test_log_without_system_prompt_has_null_prompt_hash(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Without system_prompt, system_prompt_hash and ipc_id must be null."""
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

        # output_hash is always computed
        assert data["output_hash"] is not None

        # Without system_prompt, these cannot be computed
        assert data["system_prompt_hash"] is None
        assert data["ipc_id"] is None


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
        """metadata.json must include all provenance fields including IPC hashes."""
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

        # IPC hash fields in metadata.json
        assert len(metadata["system_prompt_hash"]) == 64
        assert len(metadata["output_hash"]) == 64
        assert len(metadata["ipc_id"]) == 64

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

    def test_all_six_files_when_both_output_and_baseline(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """When both output and baseline are set, all 6 files must exist
        including delta.json from the signal isolation pipeline."""
        body = {**save_request_body, "baseline": "Baseline text."}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        assert data["files"] == [
            "baseline.md",
            "delta.json",
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

    def test_oserror_returns_500(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """An OSError during file I/O must surface as HTTP 500."""
        with (
            patch("app.main._DATA_DIR", tmp_path),
            patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
        ):
            resp = client.post("/api/save", json=save_request_body)

        assert resp.status_code == 500
        assert "disk full" in resp.json()["detail"]

    def test_save_response_contains_hash_fields(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """SaveResponse must include system_prompt_hash, output_hash, and ipc_id."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        assert resp.status_code == 200
        data = resp.json()

        for key in ("system_prompt_hash", "output_hash", "ipc_id"):
            assert key in data, f"Missing field: {key}"
            assert isinstance(data[key], str), f"{key} should be a string"
            assert len(data[key]) == 64, f"{key} should be 64-char hex"
            int(data[key], 16)

    def test_save_without_output_has_null_output_hash(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """When output is None, output_hash and ipc_id must be null."""
        body = {**save_request_body, "output": None}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        assert resp.status_code == 200
        data = resp.json()

        # system_prompt_hash is always computed (prompt is always provided)
        assert data["system_prompt_hash"] is not None
        assert len(data["system_prompt_hash"]) == 64

        # Without output, the chain is incomplete
        assert data["output_hash"] is None
        assert data["ipc_id"] is None

    def test_delta_json_written_when_both_output_and_baseline(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """delta.json must be created when both output and baseline are present."""
        body = {
            **save_request_body,
            "output": "A dark figure lurks beyond the crumbling gate.",
            "baseline": "The weathered figure stands near the threshold.",
        }
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]

        # File must exist and be listed in the response
        assert (save_dir / "delta.json").exists()
        assert "delta.json" in data["files"]

        # Parse and verify structure
        delta = json.loads((save_dir / "delta.json").read_text(encoding="utf-8"))
        assert isinstance(delta["removed"], list)
        assert isinstance(delta["added"], list)
        assert delta["removed_count"] == len(delta["removed"])
        assert delta["added_count"] == len(delta["added"])

        # The two texts differ, so there should be content in the delta
        assert delta["removed_count"] > 0 or delta["added_count"] > 0

    def test_delta_json_lists_are_sorted(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """removed and added lists in delta.json must be alphabetically sorted."""
        body = {
            **save_request_body,
            "output": "The zebra and antelope walk slowly near the river.",
            "baseline": "The monkey and tiger swim quickly across the bridge.",
        }
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        save_dir = tmp_path / resp.json()["folder_name"]
        delta = json.loads((save_dir / "delta.json").read_text(encoding="utf-8"))
        assert delta["removed"] == sorted(delta["removed"])
        assert delta["added"] == sorted(delta["added"])

    def test_delta_json_omitted_when_no_baseline(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """delta.json must not be created when baseline is None."""
        body = {**save_request_body, "baseline": None}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        assert not (save_dir / "delta.json").exists()
        assert "delta.json" not in data["files"]

    def test_delta_json_omitted_when_no_output(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """delta.json must not be created when output is None."""
        body = {**save_request_body, "output": None, "baseline": "Some baseline."}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        assert not (save_dir / "delta.json").exists()
        assert "delta.json" not in data["files"]

    def test_delta_json_does_not_affect_ipc_hashes(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """IPC hashes must be identical whether or not a baseline is present,
        since delta.json is a derived analysis that does not participate in
        the provenance chain."""
        # Save without baseline (no delta.json)
        body_no_baseline = {**save_request_body, "baseline": None}
        with patch("app.main._DATA_DIR", tmp_path):
            r1 = client.post("/api/save", json=body_no_baseline)

        # Save with baseline (delta.json written)
        body_with_baseline = {**save_request_body, "baseline": "Baseline text."}
        with patch("app.main._DATA_DIR", tmp_path):
            r2 = client.post("/api/save", json=body_with_baseline)

        d1 = r1.json()
        d2 = r2.json()

        # All IPC hashes must be identical — the baseline does not affect them
        assert d1["input_hash"] == d2["input_hash"]
        assert d1["system_prompt_hash"] == d2["system_prompt_hash"]
        assert d1["output_hash"] == d2["output_hash"]
        assert d1["ipc_id"] == d2["ipc_id"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/analyze-delta
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeDeltaEndpoint:
    """Tests for the POST /api/analyze-delta endpoint (Signal Isolation Layer)."""

    def test_successful_delta(self, client: TestClient) -> None:
        """Happy path: two different texts produce non-empty removed/added lists."""
        resp = client.post(
            "/api/analyze-delta",
            json={
                "baseline_text": "The dark figure stands near the threshold.",
                "current_text": "A bright goblin lurks beyond the crumbling gate.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "removed" in data
        assert "added" in data
        assert isinstance(data["removed"], list)
        assert isinstance(data["added"], list)
        # Both should have content since the texts are different
        assert len(data["removed"]) > 0
        assert len(data["added"]) > 0

    def test_identical_texts_empty_delta(self, client: TestClient) -> None:
        """Identical texts must produce empty removed and added lists."""
        text = "A weathered figure stands near the threshold."
        resp = client.post(
            "/api/analyze-delta",
            json={"baseline_text": text, "current_text": text},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] == []
        assert data["added"] == []

    def test_results_alphabetically_sorted(self, client: TestClient) -> None:
        """Both removed and added lists must be alphabetically sorted."""
        resp = client.post(
            "/api/analyze-delta",
            json={
                "baseline_text": "The zebra and antelope walk slowly near the river.",
                "current_text": "The monkey and tiger swim quickly across the bridge.",
            },
        )
        data = resp.json()
        assert data["removed"] == sorted(data["removed"])
        assert data["added"] == sorted(data["added"])

    def test_empty_baseline_returns_422(self, client: TestClient) -> None:
        """Empty baseline_text must be rejected by validation."""
        resp = client.post(
            "/api/analyze-delta",
            json={"baseline_text": "", "current_text": "Some text."},
        )
        assert resp.status_code == 422

    def test_empty_current_returns_422(self, client: TestClient) -> None:
        """Empty current_text must be rejected by validation."""
        resp = client.post(
            "/api/analyze-delta",
            json={"baseline_text": "Some text.", "current_text": ""},
        )
        assert resp.status_code == 422

    def test_missing_fields_returns_422(self, client: TestClient) -> None:
        """Missing required fields must return 422."""
        resp = client.post(
            "/api/analyze-delta",
            json={"baseline_text": "Only baseline."},
        )
        assert resp.status_code == 422

    def test_deterministic_across_calls(self, client: TestClient) -> None:
        """Two identical requests must produce identical responses."""
        body = {
            "baseline_text": "The dark figure stands near the threshold.",
            "current_text": "A bright goblin lurks beyond the gate.",
        }
        r1 = client.post("/api/analyze-delta", json=body).json()
        r2 = client.post("/api/analyze-delta", json=body).json()
        assert r1 == r2


class TestTransformationMapEndpoint:
    """Tests for the POST /api/transformation-map endpoint."""

    def test_successful_replacement(self, client: TestClient) -> None:
        """Two different texts should produce at least one replacement row."""
        resp = client.post(
            "/api/transformation-map",
            json={
                "baseline_text": "The old goblin stands near the gate.",
                "current_text": "The young goblin waits by the door.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rows" in data
        assert isinstance(data["rows"], list)
        assert len(data["rows"]) >= 1
        for row in data["rows"]:
            assert "removed" in row
            assert "added" in row

    def test_identical_texts_empty_rows(self, client: TestClient) -> None:
        """Identical texts should produce no rows."""
        text = "A weathered figure stands near the threshold."
        resp = client.post(
            "/api/transformation-map",
            json={"baseline_text": text, "current_text": text},
        )
        assert resp.status_code == 200
        assert resp.json()["rows"] == []

    def test_include_all_parameter(self, client: TestClient) -> None:
        """include_all=True should include insert/delete rows."""
        resp = client.post(
            "/api/transformation-map",
            json={
                "baseline_text": "The goblin stands.",
                "current_text": "The goblin stands. A new sentence appears.",
                "include_all": True,
            },
        )
        assert resp.status_code == 200
        rows = resp.json()["rows"]
        found = any(row["removed"] == "" and "new sentence" in row["added"] for row in rows)
        assert found, f"Expected insert row in {rows}"

    def test_empty_baseline_returns_422(self, client: TestClient) -> None:
        """Empty baseline_text must be rejected by validation."""
        resp = client.post(
            "/api/transformation-map",
            json={"baseline_text": "", "current_text": "Some text."},
        )
        assert resp.status_code == 422

    def test_response_includes_indicators_field(self, client: TestClient) -> None:
        """Every row in the response must include an 'indicators' list."""
        resp = client.post(
            "/api/transformation-map",
            json={
                "baseline_text": "The old goblin stands near the gate.",
                "current_text": "The young goblin waits by the door.",
            },
        )
        assert resp.status_code == 200
        for row in resp.json()["rows"]:
            assert "indicators" in row
            assert isinstance(row["indicators"], list)

    def test_indicator_config_accepted(self, client: TestClient) -> None:
        """Passing indicator_config should not error."""
        resp = client.post(
            "/api/transformation-map",
            json={
                "baseline_text": "The old goblin stands near the gate.",
                "current_text": "The young goblin waits by the door.",
                "indicator_config": {
                    "compression_ratio": 1.5,
                    "expansion_ratio": 1.5,
                    "min_tokens": 1,
                    "modality_density_threshold": 0.25,
                },
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["rows"]) >= 1

    def test_indicator_config_enabled_filter(self, client: TestClient) -> None:
        """When enabled is set, only those indicators should appear."""
        resp = client.post(
            "/api/transformation-map",
            json={
                "baseline_text": "The old goblin stands near the gate.",
                "current_text": "The young goblin waits by the door.",
                "indicator_config": {"enabled": ["compression"]},
            },
        )
        assert resp.status_code == 200
        for row in resp.json()["rows"]:
            for ind in row["indicators"]:
                assert ind == "compression"

    def test_identical_texts_indicators_empty(self, client: TestClient) -> None:
        """Identical texts produce no rows, hence no indicators."""
        text = "A weathered figure stands near the threshold."
        resp = client.post(
            "/api/transformation-map",
            json={"baseline_text": text, "current_text": text},
        )
        assert resp.status_code == 200
        assert resp.json()["rows"] == []


class TestSaveManifest:
    """Tests verifying the manifest section in metadata.json.

    The manifest provides per-file SHA-256 checksums, roles, and byte sizes
    so that save packages are self-describing and scientifically verifiable.
    metadata.json is written LAST so the manifest can include checksums of
    all other files; its own entry carries ``sha256: null`` because it
    cannot hash itself.
    """

    def test_metadata_contains_manifest_key(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """metadata.json must include a 'manifest' section after saving."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        assert "manifest" in metadata
        assert "manifest_version" in metadata["manifest"]
        assert "files" in metadata["manifest"]

    def test_manifest_version_is_one(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Manifest version must be 1."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        assert metadata["manifest"]["manifest_version"] == 1

    def test_manifest_lists_all_written_files(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Every file in the response's files list must appear in the manifest."""
        body = {**save_request_body, "baseline": "Baseline text."}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        manifest_files = metadata["manifest"]["files"]
        for filename in data["files"]:
            assert filename in manifest_files, f"'{filename}' missing from manifest"

    def test_manifest_checksums_are_valid(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Non-null SHA-256 checksums in the manifest must match file contents."""
        import hashlib

        body = {**save_request_body, "baseline": "Baseline text."}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        for filename, entry in metadata["manifest"]["files"].items():
            if entry["sha256"] is None:
                # metadata.json cannot hash itself — skip
                assert filename == "metadata.json"
                continue
            actual_hash = hashlib.sha256((save_dir / filename).read_bytes()).hexdigest()
            assert actual_hash == entry["sha256"], (
                f"Checksum mismatch for {filename}: expected {entry['sha256'][:16]}…, "
                f"got {actual_hash[:16]}…"
            )

    def test_manifest_metadata_json_has_null_sha256(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """metadata.json's manifest entry must have sha256=null (cannot hash itself)."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        assert metadata["manifest"]["files"]["metadata.json"]["sha256"] is None
        assert metadata["manifest"]["files"]["metadata.json"]["role"] == "provenance"

    def test_manifest_roles_are_correct(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Each file's role in the manifest must match the expected _FILE_ROLES mapping."""
        expected_roles = {
            "metadata.json": "provenance",
            "payload.json": "payload",
            "system_prompt.md": "system_prompt",
            "output.md": "output",
        }
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        for filename, expected_role in expected_roles.items():
            if filename in metadata["manifest"]["files"]:
                assert metadata["manifest"]["files"][filename]["role"] == expected_role

    def test_manifest_size_bytes_match_actual(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """size_bytes in manifest entries must match actual file sizes on disk."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=save_request_body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]
        metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))

        for filename, entry in metadata["manifest"]["files"].items():
            if filename == "metadata.json":
                # metadata.json uses size_bytes=0 as sentinel
                assert entry["size_bytes"] == 0
                continue
            actual_size = (save_dir / filename).stat().st_size
            assert actual_size == entry["size_bytes"], (
                f"Size mismatch for {filename}: manifest says {entry['size_bytes']}, "
                f"actual is {actual_size}"
            )


# ── GET /api/save/{folder_name}/export ──────────────────────────────────────


class TestExportEndpoint:
    """Tests for the GET /api/save/{folder_name}/export zip download endpoint.

    Each test saves a package first (via POST /api/save), then exports it
    as a zip to verify the export pipeline end-to-end.
    """

    def test_happy_path_save_then_export(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Save a package, export as zip, verify it's a valid zip with expected files."""
        import zipfile as _zipfile

        with patch("app.main._DATA_DIR", tmp_path):
            save_resp = client.post("/api/save", json=save_request_body)
            folder_name = save_resp.json()["folder_name"]

            export_resp = client.get(f"/api/save/{folder_name}/export")

        assert export_resp.status_code == 200
        assert export_resp.headers["content-type"] == "application/zip"
        assert folder_name in export_resp.headers["content-disposition"]

        # Parse the zip and verify expected files
        zf = _zipfile.ZipFile(io.BytesIO(export_resp.content))
        names = zf.namelist()
        assert "metadata.json" in names
        assert "payload.json" in names
        assert "system_prompt.md" in names
        assert "output.md" in names

    def test_zip_content_round_trips(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """File contents inside the exported zip must match the originals on disk."""
        import zipfile as _zipfile

        with patch("app.main._DATA_DIR", tmp_path):
            save_resp = client.post("/api/save", json=save_request_body)
            folder_name = save_resp.json()["folder_name"]

            export_resp = client.get(f"/api/save/{folder_name}/export")

        save_dir = tmp_path / folder_name
        zf = _zipfile.ZipFile(io.BytesIO(export_resp.content))
        for name in zf.namelist():
            assert zf.read(name) == (save_dir / name).read_bytes()

    def test_missing_folder_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        """Exporting a non-existent folder must return 404."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.get("/api/save/20260219_120000_deadbeef/export")
        assert resp.status_code == 404

    def test_invalid_folder_name_returns_400(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        """A folder name that doesn't match the expected pattern must return 400."""
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.get("/api/save/not_a_valid_folder_name/export")
        assert resp.status_code == 400

    def test_correct_content_disposition_header(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """The Content-Disposition header must include the folder name as the filename."""
        with patch("app.main._DATA_DIR", tmp_path):
            save_resp = client.post("/api/save", json=save_request_body)
            folder_name = save_resp.json()["folder_name"]

            export_resp = client.get(f"/api/save/{folder_name}/export")

        expected = f'attachment; filename="{folder_name}.zip"'
        assert export_resp.headers["content-disposition"] == expected


# ── POST /api/import ────────────────────────────────────────────────────────


class TestImportEndpoint:
    """Tests for the POST /api/import zip upload endpoint.

    Most tests perform a save → export → import round-trip to verify the
    complete pipeline.  The endpoint accepts multipart file uploads and
    returns an ImportResponse with all state needed for frontend restoration.
    """

    def test_round_trip_save_export_import(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """Full round-trip: save → export → import → verify restored state."""
        with patch("app.main._DATA_DIR", tmp_path):
            # 1. Save
            save_resp = client.post("/api/save", json=save_request_body)
            folder_name = save_resp.json()["folder_name"]

            # 2. Export
            export_resp = client.get(f"/api/save/{folder_name}/export")

            # 3. Import
            import_resp = client.post(
                "/api/import",
                files={"file": (f"{folder_name}.zip", export_resp.content, "application/zip")},
            )

        assert import_resp.status_code == 200
        data = import_resp.json()

        # Verify restored state matches the save request
        assert data["folder_name"] == folder_name
        assert data["model"] == "gemma2:2b"
        assert data["temperature"] == 0.2
        assert data["max_tokens"] == 120
        assert data["manifest_valid"] is True
        assert data["payload"]["seed"] == 42
        assert data["payload"]["world_id"] == "test_world"
        assert "health" in data["payload"]["axes"]

        # System prompt extracted from fenced code block
        assert "deterministic system" in data["system_prompt"]

        # Output extracted from markdown body
        assert "weathered figure" in data["output"]

    def test_import_preserves_baseline(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """When baseline was saved, import must restore it."""
        body = {**save_request_body, "baseline": "The old goblin shuffles forward."}
        with patch("app.main._DATA_DIR", tmp_path):
            save_resp = client.post("/api/save", json=body)
            folder_name = save_resp.json()["folder_name"]
            export_resp = client.get(f"/api/save/{folder_name}/export")

            import_resp = client.post(
                "/api/import",
                files={"file": (f"{folder_name}.zip", export_resp.content, "application/zip")},
            )

        data = import_resp.json()
        assert data["baseline"] is not None
        assert "old goblin" in data["baseline"]

    def test_import_without_manifest_warns(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        """Importing a zip without a manifest should succeed with a warning."""
        import zipfile as _zipfile

        # Build a minimal zip without manifest in metadata.json
        metadata = {"model": "gemma2:2b", "temperature": 0.2, "max_tokens": 120}
        payload = {
            "axes": {"health": {"label": "weary", "score": 0.5}},
            "policy_hash": "abc",
            "seed": 42,
            "world_id": "w",
        }
        prompt_md = "# System Prompt\n\n```text\nYou are a test prompt.\n```\n"

        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", json.dumps(metadata))
            zf.writestr("payload.json", json.dumps(payload))
            zf.writestr("system_prompt.md", prompt_md)
        zip_bytes = buf.getvalue()

        resp = client.post(
            "/api/import",
            files={"file": ("test.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert any("manifest" in w.lower() or "checksum" in w.lower() for w in data["warnings"])

    def test_import_checksum_mismatch_returns_400(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """A zip with tampered file contents must fail checksum validation."""
        import zipfile as _zipfile

        with patch("app.main._DATA_DIR", tmp_path):
            save_resp = client.post("/api/save", json=save_request_body)
            folder_name = save_resp.json()["folder_name"]
            export_resp = client.get(f"/api/save/{folder_name}/export")

        # Tamper with the zip: replace payload.json content
        original_zip = _zipfile.ZipFile(io.BytesIO(export_resp.content))
        tampered_buf = io.BytesIO()
        with _zipfile.ZipFile(tampered_buf, "w") as tampered:
            for name in original_zip.namelist():
                content = original_zip.read(name)
                if name == "payload.json":
                    content = b'{"tampered": true}'
                tampered.writestr(name, content)

        resp = client.post(
            "/api/import",
            files={"file": ("tampered.zip", tampered_buf.getvalue(), "application/zip")},
        )
        assert resp.status_code == 400
        assert (
            "checksum" in resp.json()["detail"].lower()
            or "mismatch" in resp.json()["detail"].lower()
        )

    def test_import_non_zip_returns_400(self, client: TestClient) -> None:
        """Uploading a non-zip file must return 400."""
        resp = client.post(
            "/api/import",
            files={"file": ("notazip.txt", b"This is not a zip file", "text/plain")},
        )
        assert resp.status_code == 400

    def test_import_missing_required_file_returns_422(self, client: TestClient) -> None:
        """A zip missing payload.json must return 422."""
        import zipfile as _zipfile

        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", '{"model": "test"}')
            zf.writestr("system_prompt.md", "# Prompt\n\n```text\ntest\n```\n")
            # payload.json intentionally omitted
        zip_bytes = buf.getvalue()

        resp = client.post(
            "/api/import",
            files={"file": ("incomplete.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 422
        assert "payload.json" in resp.json()["detail"]

    def test_import_files_list_is_sorted(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """The files list in ImportResponse must be sorted alphabetically."""
        with patch("app.main._DATA_DIR", tmp_path):
            save_resp = client.post("/api/save", json=save_request_body)
            folder_name = save_resp.json()["folder_name"]
            export_resp = client.get(f"/api/save/{folder_name}/export")

            import_resp = client.post(
                "/api/import",
                files={"file": (f"{folder_name}.zip", export_resp.content, "application/zip")},
            )

        files = import_resp.json()["files"]
        assert files == sorted(files)

    def test_import_oversized_upload_returns_400(self, client: TestClient) -> None:
        """An upload exceeding MAX_UPLOAD_SIZE must return 400."""
        import zipfile as _zipfile

        from unittest.mock import patch as _patch

        # Build a valid zip, then enforce a tiny upload limit
        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", '{"model": "test"}')
            zf.writestr("payload.json", '{"axes": {}}')
            zf.writestr("system_prompt.md", "```text\ntest\n```")
        zip_bytes = buf.getvalue()

        with _patch("app.main.MAX_UPLOAD_SIZE", 10):
            resp = client.post(
                "/api/import",
                files={"file": ("big.zip", zip_bytes, "application/zip")},
            )
        assert resp.status_code == 400
        assert "exceeds" in resp.json()["detail"].lower()

    def test_import_missing_metadata_json_returns_422(self, client: TestClient) -> None:
        """A zip without metadata.json must return 422."""
        import zipfile as _zipfile

        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "payload.json",
                '{"axes": {"health": {"label": "ok", "score": 0.5}}, "policy_hash": "a", "seed": 1, "world_id": "w"}',
            )
            zf.writestr("system_prompt.md", "```text\ntest\n```")
            # metadata.json intentionally omitted
        zip_bytes = buf.getvalue()

        resp = client.post(
            "/api/import",
            files={"file": ("no_meta.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 422
        assert "metadata.json" in resp.json()["detail"]

    def test_import_corrupt_metadata_json_returns_400(self, client: TestClient) -> None:
        """A zip with invalid JSON in metadata.json must return 400."""
        import zipfile as _zipfile

        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", "NOT VALID JSON {{{")
            zf.writestr("payload.json", '{"axes": {}}')
            zf.writestr("system_prompt.md", "```text\ntest\n```")
        zip_bytes = buf.getvalue()

        resp = client.post(
            "/api/import",
            files={"file": ("bad_meta.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 400
        assert "not valid json" in resp.json()["detail"].lower()

    def test_import_corrupt_payload_json_returns_400(self, client: TestClient) -> None:
        """A zip with invalid JSON in payload.json must return 400."""
        import zipfile as _zipfile

        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", '{"model": "test"}')
            zf.writestr("payload.json", "NOT VALID JSON")
            zf.writestr("system_prompt.md", "```text\ntest\n```")
        zip_bytes = buf.getvalue()

        resp = client.post(
            "/api/import",
            files={"file": ("bad_payload.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 400
        assert "payload.json" in resp.json()["detail"].lower()

    def test_import_missing_system_prompt_returns_422(self, client: TestClient) -> None:
        """A zip without system_prompt.md must return 422."""
        import zipfile as _zipfile

        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", '{"model": "test"}')
            zf.writestr(
                "payload.json",
                '{"axes": {"health": {"label": "ok", "score": 0.5}}, "policy_hash": "a", "seed": 1, "world_id": "w"}',
            )
            # system_prompt.md intentionally omitted
        zip_bytes = buf.getvalue()

        resp = client.post(
            "/api/import",
            files={"file": ("no_prompt.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 422
        assert "system_prompt.md" in resp.json()["detail"]


class TestTransformationMapSave:
    """Tests for transformation_map.json in the save package."""

    def test_tmap_json_written_when_provided(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """transformation_map.json must be written when rows are provided."""
        body = {
            **save_request_body,
            "transformation_map": [
                {"removed": "old dark", "added": "young bright"},
                {"removed": "stands", "added": "waits"},
            ],
        }
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]

        assert (save_dir / "transformation_map.json").exists()
        assert "transformation_map.json" in data["files"]

        tmap = json.loads((save_dir / "transformation_map.json").read_text(encoding="utf-8"))
        assert isinstance(tmap["rows"], list)
        assert tmap["row_count"] == 2
        assert tmap["rows"][0]["removed"] == "old dark"

    def test_tmap_json_omitted_when_not_provided(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """transformation_map.json must not be created when field is absent."""
        body = {**save_request_body}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]

        assert not (save_dir / "transformation_map.json").exists()
        assert "transformation_map.json" not in data["files"]

    def test_tmap_json_omitted_when_empty_list(
        self,
        client: TestClient,
        save_request_body: dict,
        tmp_path: Path,
    ) -> None:
        """transformation_map.json must not be created when rows list is empty."""
        body = {**save_request_body, "transformation_map": []}
        with patch("app.main._DATA_DIR", tmp_path):
            resp = client.post("/api/save", json=body)

        data = resp.json()
        save_dir = tmp_path / data["folder_name"]

        assert not (save_dir / "transformation_map.json").exists()
        assert "transformation_map.json" not in data["files"]

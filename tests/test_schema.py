"""Tests for app/schema.py – Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schema import (
    AxisPayload,
    AxisValue,
    GenerateRequest,
    GenerateResponse,
    LogEntry,
)

# ── AxisValue ────────────────────────────────────────────────────────────────


class TestAxisValue:
    def test_valid(self) -> None:
        av = AxisValue(label="weary", score=0.5)
        assert av.label == "weary"
        assert av.score == 0.5

    def test_score_boundaries(self) -> None:
        assert AxisValue(label="low", score=0.0).score == 0.0
        assert AxisValue(label="high", score=1.0).score == 1.0

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AxisValue(label="bad", score=-0.1)

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AxisValue(label="bad", score=1.1)

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="label must not be empty"):
            AxisValue(label="", score=0.5)

    def test_whitespace_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="label must not be empty"):
            AxisValue(label="   ", score=0.5)

    def test_label_stripped(self) -> None:
        av = AxisValue(label="  weary  ", score=0.5)
        assert av.label == "weary"


# ── AxisPayload ──────────────────────────────────────────────────────────────


class TestAxisPayload:
    def test_valid(self, sample_payload: AxisPayload) -> None:
        assert "health" in sample_payload.axes
        assert sample_payload.seed == 42

    def test_empty_axes_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one entry"):
            AxisPayload(axes={}, policy_hash="abc", seed=1, world_id="w")

    def test_multiple_axes(self) -> None:
        p = AxisPayload(
            axes={
                "a": AxisValue(label="x", score=0.1),
                "b": AxisValue(label="y", score=0.9),
            },
            policy_hash="hash",
            seed=99,
            world_id="world",
        )
        assert len(p.axes) == 2


# ── GenerateRequest ──────────────────────────────────────────────────────────


class TestGenerateRequest:
    def test_defaults(self, sample_payload: AxisPayload) -> None:
        req = GenerateRequest(payload=sample_payload, model="gemma2:2b")
        assert req.temperature == 0.2
        assert req.max_tokens == 120
        assert req.system_prompt is None

    def test_custom_values(self, sample_payload: AxisPayload) -> None:
        req = GenerateRequest(
            payload=sample_payload,
            model="llama3.2:1b",
            temperature=0.8,
            max_tokens=256,
            system_prompt="custom prompt",
        )
        assert req.temperature == 0.8
        assert req.max_tokens == 256
        assert req.system_prompt == "custom prompt"

    def test_temperature_too_high(self, sample_payload: AxisPayload) -> None:
        with pytest.raises(ValidationError):
            GenerateRequest(payload=sample_payload, model="m", temperature=2.5)

    def test_max_tokens_too_low(self, sample_payload: AxisPayload) -> None:
        with pytest.raises(ValidationError):
            GenerateRequest(payload=sample_payload, model="m", max_tokens=5)

    def test_max_tokens_too_high(self, sample_payload: AxisPayload) -> None:
        with pytest.raises(ValidationError):
            GenerateRequest(payload=sample_payload, model="m", max_tokens=9999)


# ── GenerateResponse ────────────────────────────────────────────────────────


class TestGenerateResponse:
    def test_valid(self) -> None:
        resp = GenerateResponse(text="A paragraph.", model="gemma2:2b", temperature=0.2)
        assert resp.text == "A paragraph."
        assert resp.usage is None

    def test_with_usage(self) -> None:
        resp = GenerateResponse(
            text="text",
            model="m",
            temperature=0.1,
            usage={"prompt_eval_count": 50, "eval_count": 30},
        )
        assert resp.usage["eval_count"] == 30


# ── LogEntry ─────────────────────────────────────────────────────────────────


class TestLogEntry:
    def test_valid(self, sample_payload: AxisPayload) -> None:
        entry = LogEntry(
            input_hash="abc123",
            payload=sample_payload,
            output="some text",
            model="gemma2:2b",
            temperature=0.2,
            max_tokens=120,
            timestamp="2026-01-01T00:00:00Z",
        )
        assert entry.input_hash == "abc123"
        assert entry.timestamp == "2026-01-01T00:00:00Z"

"""Shared fixtures for the Axis Descriptor Lab test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schema import AxisPayload, AxisValue


@pytest.fixture()
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture()
def sample_axis_value() -> AxisValue:
    return AxisValue(label="weary", score=0.5)


@pytest.fixture()
def sample_payload_dict() -> dict:
    """Minimal valid payload as a raw dict (for JSON POST bodies)."""
    return {
        "axes": {
            "health": {"label": "weary", "score": 0.5},
            "age": {"label": "old", "score": 0.7},
        },
        "policy_hash": "abc123",
        "seed": 42,
        "world_id": "test_world",
    }


@pytest.fixture()
def sample_payload(sample_payload_dict: dict) -> AxisPayload:
    """Minimal valid AxisPayload model instance."""
    return AxisPayload(**sample_payload_dict)

"""
Axis primitives shared across the Axis Descriptor Lab API.

This module contains the authoritative deterministic payload structures used
by both the character-description and chat-translation flows.  These models
are intentionally small and validation-focused because they are reused by
multiple higher-level request/response models across the API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AxisValue(BaseModel):
    """
    A single named axis entry consisting of a human-readable label and a
    normalised score in [0, 1].

    The *label* is the interpretive colour (e.g. "resentful", "weary").
    The *score* is the underlying deterministic value produced by the engine.
    Both are forwarded verbatim to the LLM; the LLM must not treat them as
    facts – they are tonal hints only.
    """

    label: str = Field(
        ...,
        description="Short human-readable descriptor for this axis position.",
        examples=["resentful", "weary", "questioned"],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised axis score in the closed interval [0, 1].",
        examples=[0.5, 0.62],
    )

    @field_validator("label")
    @classmethod
    def label_not_empty(cls, v: str) -> str:
        """Ensure the label is not a blank string."""
        v = v.strip()
        if not v:
            raise ValueError("label must not be empty or whitespace")
        return v


class AxisPayload(BaseModel):
    """
    The complete deterministic state object that drives a single descriptive
    generation.  This is the "source of truth" handed to the LLM as a JSON
    string in the user turn.

    `axes` is keyed by axis name (for example, `"demeanor"`) and stores
    `AxisValue` entries. `policy_hash` captures the SHA-256 digest of the
    policy rules in force when the payload was produced. `seed` records the
    deterministic RNG seed used to produce the scores, and `world_id`
    identifies the Pipe-Works world context.
    """

    axes: dict[str, AxisValue] = Field(
        ...,
        description="Map of axis name → AxisValue.  At least one entry required.",
    )
    policy_hash: str = Field(
        ...,
        description=(
            "SHA-256 hex digest of the axis policy in force.  "
            "Forwarded to the LLM but the system prompt instructs the model "
            "never to mention or interpret it."
        ),
    )
    seed: int = Field(
        ...,
        description="RNG seed that produced the axis scores.  Used for reproducibility.",
    )
    world_id: str = Field(
        ...,
        description="Identifier for the Pipe-Works world (e.g. 'pipeworks_web').",
    )

    @field_validator("axes")
    @classmethod
    def axes_not_empty(cls, v: dict[str, AxisValue]) -> dict[str, AxisValue]:
        """Require at least one axis."""
        if not v:
            raise ValueError("axes must contain at least one entry")
        return v

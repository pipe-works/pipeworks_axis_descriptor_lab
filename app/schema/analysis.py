"""
Schemas for signal isolation and transformation-map analysis endpoints.

These models describe the deterministic NLP analysis APIs that compare a
baseline output against a current output.  They are intentionally isolated
from generation and save concerns so the analysis surface stays easy to scan.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndicatorConfig(BaseModel):
    """
    Optional tuning parameters for micro-indicator detection.

    Sent as an optional field in :class:`TransformationMapRequest`.  When
    absent, conservative defaults apply.  All thresholds are designed to
    minimise false positives — users can loosen them for exploratory use.
    """

    compression_ratio: float = Field(
        default=2.0,
        ge=1.0,
        description=(
            "Minimum ratio of removed/added token counts to flag "
            "'compression'.  Default 2.0 means removed must be at least "
            "twice as long as added."
        ),
    )
    expansion_ratio: float = Field(
        default=2.0,
        ge=1.0,
        description="Minimum ratio of added/removed token counts to flag 'expansion'.  Default 2.0.",
    )
    min_tokens: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum token count on the larger side to consider "
            "compression/expansion.  Prevents flagging single-word swaps."
        ),
    )
    modality_density_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum absolute change in verb+adjective density "
            "(proportion of tokens) to flag 'modality shift'.  "
            "Default 0.3 (conservative)."
        ),
    )
    enabled: list[str] | None = Field(
        default=None,
        description=(
            "When not None, only compute indicators whose names appear "
            "in this list.  None means all indicators are active."
        ),
    )


class TransformationMapRequest(BaseModel):
    """
    Request body for ``POST /api/transformation-map``.

    Accepts two plain-text strings (baseline and current) for clause-level
    alignment analysis.  The backend runs sentence-aware alignment followed
    by token-level diffing to extract replacement pairs, then classifies
    each row with structural micro-indicators.
    """

    baseline_text: str = Field(
        ...,
        min_length=1,
        description="The reference text (A) — typically the stored baseline output.  Must not be empty.",
    )
    current_text: str = Field(
        ...,
        min_length=1,
        description=(
            "The comparison text (B) — typically the latest generated output.  "
            "Must not be empty."
        ),
    )
    include_all: bool = Field(
        default=False,
        description=(
            "When True, include insert-only and delete-only operations as "
            "rows (with an empty removed or added side).  When False "
            "(default), only replacement operations are returned."
        ),
    )
    indicator_config: IndicatorConfig | None = Field(
        default=None,
        description=(
            "Optional tuning parameters for micro-indicator detection.  "
            "When absent, conservative defaults apply."
        ),
    )


class TransformationMapRow(BaseModel):
    """
    A single clause-level replacement pair with optional micro-indicators.

    The ``indicators`` field contains zero or more structural pattern labels
    computed by the ``micro_indicators`` module.  Each label is a
    deterministic heuristic tag such as ``"compression"``,
    ``"embodiment shift"``, or ``"intensity ↑"``.
    """

    removed: str = Field(
        ...,
        description="The text chunk from the baseline (A) that was replaced.",
    )
    added: str = Field(
        ...,
        description="The text chunk from the current text (B) that replaced it.",
    )
    indicators: list[str] = Field(
        default_factory=list,
        description=(
            "Micro-indicator labels for this row (e.g. 'compression', "
            "'embodiment shift').  Empty list if no indicators match.  "
            "Computed server-side by deterministic heuristics."
        ),
    )


class TransformationMapResponse(BaseModel):
    """
    Response body for ``POST /api/transformation-map``.

    Contains a list of clause-level replacement pairs extracted by
    sentence-aware alignment and token-level diffing.
    """

    rows: list[TransformationMapRow] = Field(
        ...,
        description=(
            "Ordered list of clause-level replacement pairs.  Each row shows "
            "text removed from A and the corresponding text added in B."
        ),
    )


class DeltaRequest(BaseModel):
    """
    Request body for ``POST /api/analyze-delta``.

    Accepts two plain-text strings (baseline and current) for signal
    isolation analysis.  The backend runs both through the NLP pipeline
    (tokenise → lemmatise → filter stopwords) and returns the set
    difference as sorted word lists.
    """

    baseline_text: str = Field(
        ...,
        min_length=1,
        description="The reference text (A) — typically the stored baseline output.  Must not be empty.",
        examples=["The weathered figure stands near the threshold."],
    )
    current_text: str = Field(
        ...,
        min_length=1,
        description=(
            "The comparison text (B) — typically the latest generated output.  "
            "Must not be empty."
        ),
        examples=["A dark goblin lurks beyond the crumbling gate."],
    )


class DeltaResponse(BaseModel):
    """
    Response body for ``POST /api/analyze-delta``.

    Contains two alphabetically sorted lists of content lemmas that
    represent meaningful lexical differences between the baseline and
    current texts, after stopword removal and lemmatisation.

    The lists are **set differences**, not positional diffs:

    - ``removed`` = content lemmas in A but absent from B.
    - ``added``   = content lemmas in B but absent from A.
    """

    removed: list[str] = Field(
        ...,
        description=(
            "Content lemmas present in the baseline (A) but absent from the "
            "current text (B).  Alphabetically sorted."
        ),
    )
    added: list[str] = Field(
        ...,
        description=(
            "Content lemmas present in the current text (B) but absent from "
            "the baseline (A).  Alphabetically sorted."
        ),
    )

"""
Regression tests for the mud-server-aligned relabel policy mirror.

These tests intentionally verify the current mirrored policy definitions used
by the Axis Descriptor Lab against the shape and semantics expected from the
Pipe-Works mud server:

1. The canonical axis order matches the mud-server ``axes.yaml`` order.
2. The per-axis ordinal label order matches the mud-server ``ordering.values``
   sequences.
3. The relabel ranges are ordered, non-overlapping, and label-complete.
4. Boundary values resolve to the same labels the mud server would publish.
5. Example payloads ship in canonical order with labels already consistent
   with the relabel policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.relabel_policy import (
    AXIS_LABEL_ORDER,
    AXIS_ORDER,
    RELABEL_POLICY,
    apply_relabel_policy,
    resolve_axis_label,
)
from app.schema import AxisPayload, AxisValue

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "app" / "examples"


class TestRelabelPolicyStructure:
    """Verify that the mirrored mud-server policy tables stay internally consistent."""

    EXPECTED_AXIS_ORDER = [
        "physique",
        "wealth",
        "health",
        "demeanor",
        "age",
        "facial_signal",
        "legitimacy",
        "visibility",
        "moral_load",
        "dependency",
        "risk_exposure",
    ]

    def test_axis_order_matches_mud_server_policy(self) -> None:
        """The canonical axis order should match the current mud-server ``axes.yaml`` order."""
        assert AXIS_ORDER == self.EXPECTED_AXIS_ORDER

    def test_policy_keys_match_axis_order(self) -> None:
        """Every canonical axis must have both label-order and threshold definitions."""
        assert list(RELABEL_POLICY.keys()) == AXIS_ORDER
        assert list(AXIS_LABEL_ORDER.keys()) == AXIS_ORDER

    @pytest.mark.parametrize("axis_name", EXPECTED_AXIS_ORDER)
    def test_label_order_matches_threshold_labels(self, axis_name: str) -> None:
        """Ordinal labels should appear in the same low-to-high order in both tables."""
        labels_from_ranges = [label for _, _, label in RELABEL_POLICY[axis_name]]
        assert labels_from_ranges == AXIS_LABEL_ORDER[axis_name]

    @pytest.mark.parametrize("axis_name", EXPECTED_AXIS_ORDER)
    def test_ranges_are_sorted_and_non_overlapping(self, axis_name: str) -> None:
        """Mirrored threshold ranges must be ordered and must not overlap."""
        previous_max = None
        for min_score, max_score, _label in RELABEL_POLICY[axis_name]:
            assert 0.0 <= min_score <= max_score <= 1.0
            if previous_max is not None:
                assert min_score > previous_max
            previous_max = max_score


class TestResolveAxisLabel:
    """Boundary and fallback tests for the mud-server-aligned range resolver."""

    @pytest.mark.parametrize(
        ("axis_name", "score", "expected_label"),
        [
            ("physique", 0.00, "frail"),
            ("physique", 0.16, "frail"),
            ("physique", 0.17, "hunched"),
            ("physique", 0.64, "wiry"),
            ("physique", 0.80, "broad"),
            ("physique", 0.81, "stocky"),
            ("wealth", 0.00, "poor"),
            ("wealth", 0.20, "modest"),
            ("wealth", 0.59, "well-kept"),
            ("wealth", 0.60, "wealthy"),
            ("wealth", 0.80, "decadent"),
            ("health", 0.19, "sickly"),
            ("health", 0.20, "limping"),
            ("health", 0.60, "scarred"),
            ("health", 0.80, "hale"),
            ("demeanor", 0.19, "timid"),
            ("demeanor", 0.20, "suspicious"),
            ("demeanor", 0.60, "alert"),
            ("demeanor", 0.80, "proud"),
            ("age", 0.24, "young"),
            ("age", 0.25, "middle-aged"),
            ("age", 0.50, "old"),
            ("age", 0.75, "ancient"),
            ("facial_signal", 0.14, "understated"),
            ("facial_signal", 0.15, "pronounced"),
            ("facial_signal", 0.45, "asymmetrical"),
            ("facial_signal", 0.90, "sharp-featured"),
            ("legitimacy", 0.24, "sanctioned"),
            ("legitimacy", 0.25, "tolerated"),
            ("legitimacy", 0.50, "questioned"),
            ("legitimacy", 0.75, "illicit"),
            ("visibility", 0.24, "hidden"),
            ("visibility", 0.25, "discrete"),
            ("visibility", 0.50, "routine"),
            ("visibility", 0.75, "conspicuous"),
            ("moral_load", 0.24, "neutral"),
            ("moral_load", 0.25, "burdened"),
            ("moral_load", 0.50, "conflicted"),
            ("moral_load", 0.75, "corrosive"),
            ("dependency", 0.24, "optional"),
            ("dependency", 0.25, "useful"),
            ("dependency", 0.50, "necessary"),
            ("dependency", 0.75, "unavoidable"),
            ("risk_exposure", 0.24, "benign"),
            ("risk_exposure", 0.25, "straining"),
            ("risk_exposure", 0.50, "hazardous"),
            ("risk_exposure", 0.75, "eroding"),
        ],
    )
    def test_boundary_scores_resolve_to_expected_labels(
        self, axis_name: str, score: float, expected_label: str
    ) -> None:
        """Exact range boundaries should map to the mirrored mud-server label."""
        assert resolve_axis_label(axis_name, score, "fallback") == expected_label

    def test_unknown_axis_preserves_existing_label(self) -> None:
        """Unknown axes should preserve their current label unchanged."""
        assert resolve_axis_label("custom_axis", 0.5, "original") == "original"

    def test_gap_score_preserves_existing_label(self) -> None:
        """Scores outside the mirrored published ranges keep the existing label."""
        assert resolve_axis_label("wealth", 0.195, "keep-me") == "keep-me"


class TestApplyRelabelPolicy:
    """End-to-end relabel tests using the public payload API."""

    def test_relabels_known_axes(self) -> None:
        """Known axes should be rewritten to their mirrored mud-server labels."""
        payload = AxisPayload(
            axes={
                "physique": AxisValue(label="placeholder", score=0.81),
                "health": AxisValue(label="placeholder", score=0.90),
                "wealth": AxisValue(label="placeholder", score=0.30),
            },
            policy_hash="hash",
            seed=1,
            world_id="w",
        )

        result = apply_relabel_policy(payload)

        assert result.axes["physique"].label == "stocky"
        assert result.axes["health"].label == "hale"
        assert result.axes["wealth"].label == "modest"

    def test_preserves_unknown_axes_and_scores(self) -> None:
        """Unknown axes and all numeric scores must pass through unchanged."""
        payload = AxisPayload(
            axes={"custom_axis": AxisValue(label="original", score=0.42)},
            policy_hash="hash",
            seed=7,
            world_id="w",
        )

        result = apply_relabel_policy(payload)

        assert result.axes["custom_axis"].label == "original"
        assert result.axes["custom_axis"].score == pytest.approx(0.42)
        assert result.policy_hash == "hash"
        assert result.seed == 7
        assert result.world_id == "w"


class TestExamplePayloads:
    """Verify the shipped examples stay aligned with the mirrored mud-server policy."""

    @pytest.mark.parametrize(
        "example_path",
        sorted(EXAMPLES_DIR.glob("*.json")),
        ids=lambda path: path.stem,
    )
    def test_examples_use_canonical_axis_order(self, example_path: Path) -> None:
        """Examples should encode axes in the same order the UI now renders them."""
        payload = AxisPayload.model_validate(json.loads(example_path.read_text(encoding="utf-8")))
        assert list(payload.axes.keys()) == AXIS_ORDER

    @pytest.mark.parametrize(
        "example_path",
        sorted(EXAMPLES_DIR.glob("*.json")),
        ids=lambda path: path.stem,
    )
    def test_examples_are_already_policy_consistent(self, example_path: Path) -> None:
        """Relabelling an example should be a no-op for all shipped example payloads."""
        payload = AxisPayload.model_validate(json.loads(example_path.read_text(encoding="utf-8")))
        relabelled = apply_relabel_policy(payload)
        assert relabelled.axes == payload.axes

"""
Tests for app/relabel_policy.py — policy table structure and relabel logic.

Test strategy
-------------
1. **Table structure**: verify all 11 axes are present, thresholds are
   ascending, and each axis has a catch-all entry at 1.01.
2. **Boundary values**: parametrized tests at exact threshold boundaries
   to confirm the "< upper_bound" semantics (score *at* a boundary falls
   into the next bucket, not the current one).
3. **Unknown axes**: axes not in the policy table are passed through
   unchanged (labels and scores both preserved).
4. **Score preservation**: relabelling never modifies scores.
5. **Non-axis field preservation**: policy_hash, seed, world_id are
   unchanged after relabelling.
"""

from __future__ import annotations

import pytest

from app.relabel_policy import RELABEL_POLICY, apply_relabel_policy
from app.schema import AxisPayload, AxisValue

# ── Policy table structure ──────────────────────────────────────────────────


class TestRelabelPolicyStructure:
    """Verify the static RELABEL_POLICY dict is well-formed."""

    # All 11 axes that should be present in the policy table.
    EXPECTED_AXES = [
        "age",
        "demeanor",
        "dependency",
        "facial_signal",
        "health",
        "legitimacy",
        "moral_load",
        "physique",
        "risk_exposure",
        "visibility",
        "wealth",
    ]

    def test_all_expected_axes_present(self) -> None:
        """The policy table must contain exactly the 11 known axes."""
        assert sorted(RELABEL_POLICY.keys()) == sorted(self.EXPECTED_AXES)

    @pytest.mark.parametrize("axis_name", EXPECTED_AXES)
    def test_thresholds_ascending(self, axis_name: str) -> None:
        """Thresholds within each axis must be in strictly ascending order."""
        thresholds = [t for t, _ in RELABEL_POLICY[axis_name]]
        assert thresholds == sorted(thresholds)
        # No duplicate thresholds
        assert len(thresholds) == len(set(thresholds))

    @pytest.mark.parametrize("axis_name", EXPECTED_AXES)
    def test_catch_all_at_1_01(self, axis_name: str) -> None:
        """Each axis must end with a catch-all threshold at 1.01."""
        last_threshold, _ = RELABEL_POLICY[axis_name][-1]
        assert last_threshold == pytest.approx(1.01)

    @pytest.mark.parametrize("axis_name", EXPECTED_AXES)
    def test_labels_are_non_empty_strings(self, axis_name: str) -> None:
        """Every label in the policy table must be a non-empty string."""
        for _, label in RELABEL_POLICY[axis_name]:
            assert isinstance(label, str)
            assert len(label.strip()) > 0


# ── Boundary value tests ───────────────────────────────────────────────────


class TestRelabelBoundaryValues:
    """Test score-to-label mapping at exact boundary thresholds.

    The policy uses ``score < upper_bound`` semantics, so a score
    exactly at a boundary should fall into the *next* bucket.
    """

    @pytest.mark.parametrize(
        "axis, score, expected_label",
        [
            # age boundaries
            ("age", 0.0, "young"),
            ("age", 0.24, "young"),
            ("age", 0.25, "middle-aged"),  # at boundary → next bucket
            ("age", 0.49, "middle-aged"),
            ("age", 0.50, "old"),
            ("age", 0.74, "old"),
            ("age", 0.75, "ancient"),
            ("age", 1.0, "ancient"),
            # health boundaries
            ("health", 0.0, "vigorous"),
            ("health", 0.25, "weary"),
            ("health", 0.50, "ailing"),
            ("health", 0.75, "failing"),
            ("health", 1.0, "failing"),
            # demeanor boundaries
            ("demeanor", 0.0, "cordial"),
            ("demeanor", 0.2, "guarded"),
            ("demeanor", 0.4, "resentful"),
            ("demeanor", 0.6, "hostile"),
            ("demeanor", 0.8, "menacing"),
            ("demeanor", 1.0, "menacing"),
            # wealth boundaries
            ("wealth", 0.0, "destitute"),
            ("wealth", 0.25, "threadbare"),
            ("wealth", 0.45, "well-kept"),
            ("wealth", 0.55, "comfortable"),
            ("wealth", 0.75, "affluent"),
        ],
    )
    def test_score_produces_expected_label(
        self, axis: str, score: float, expected_label: str
    ) -> None:
        """Verify that a given score on a given axis produces the expected label."""
        payload = AxisPayload(
            axes={axis: AxisValue(label="placeholder", score=score)},
            policy_hash="test",
            seed=1,
            world_id="w",
        )
        result = apply_relabel_policy(payload)
        assert result.axes[axis].label == expected_label


# ── Unknown axis preservation ──────────────────────────────────────────────


class TestRelabelUnknownAxes:
    """Axes not in the policy table must pass through unchanged."""

    def test_unknown_axis_label_preserved(self) -> None:
        """An axis not in RELABEL_POLICY keeps its original label."""
        payload = AxisPayload(
            axes={"custom_axis": AxisValue(label="original_label", score=0.5)},
            policy_hash="h",
            seed=1,
            world_id="w",
        )
        result = apply_relabel_policy(payload)
        assert result.axes["custom_axis"].label == "original_label"

    def test_unknown_axis_score_preserved(self) -> None:
        """An axis not in RELABEL_POLICY keeps its original score."""
        payload = AxisPayload(
            axes={"custom_axis": AxisValue(label="x", score=0.42)},
            policy_hash="h",
            seed=1,
            world_id="w",
        )
        result = apply_relabel_policy(payload)
        assert result.axes["custom_axis"].score == pytest.approx(0.42)

    def test_mixed_known_and_unknown_axes(self) -> None:
        """Known axes get relabelled; unknown axes pass through in one call."""
        payload = AxisPayload(
            axes={
                "age": AxisValue(label="placeholder", score=0.1),
                "custom": AxisValue(label="keep_me", score=0.9),
            },
            policy_hash="h",
            seed=1,
            world_id="w",
        )
        result = apply_relabel_policy(payload)
        assert result.axes["age"].label == "young"
        assert result.axes["custom"].label == "keep_me"


# ── Score preservation ─────────────────────────────────────────────────────


class TestRelabelScorePreservation:
    """Relabelling must never modify axis scores."""

    def test_scores_unchanged_after_relabel(self) -> None:
        """All axis scores must be identical before and after relabelling."""
        payload = AxisPayload(
            axes={
                "age": AxisValue(label="x", score=0.33),
                "health": AxisValue(label="x", score=0.77),
                "wealth": AxisValue(label="x", score=0.5),
            },
            policy_hash="h",
            seed=1,
            world_id="w",
        )
        result = apply_relabel_policy(payload)
        for axis_name in payload.axes:
            assert result.axes[axis_name].score == pytest.approx(payload.axes[axis_name].score)


# ── Non-axis field preservation ────────────────────────────────────────────


class TestRelabelNonAxisFields:
    """Relabelling must preserve all non-axis fields on the payload."""

    def test_policy_hash_preserved(self) -> None:
        payload = AxisPayload(
            axes={"age": AxisValue(label="x", score=0.1)},
            policy_hash="keep_this_hash",
            seed=42,
            world_id="my_world",
        )
        result = apply_relabel_policy(payload)
        assert result.policy_hash == "keep_this_hash"

    def test_seed_preserved(self) -> None:
        payload = AxisPayload(
            axes={"age": AxisValue(label="x", score=0.1)},
            policy_hash="h",
            seed=999,
            world_id="w",
        )
        result = apply_relabel_policy(payload)
        assert result.seed == 999

    def test_world_id_preserved(self) -> None:
        payload = AxisPayload(
            axes={"age": AxisValue(label="x", score=0.1)},
            policy_hash="h",
            seed=1,
            world_id="my_world",
        )
        result = apply_relabel_policy(payload)
        assert result.world_id == "my_world"

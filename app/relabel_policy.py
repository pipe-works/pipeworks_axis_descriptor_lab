"""
app/relabel_policy.py
-----------------------------------------------------------------------------
Server-side policy table and score-to-label mapping for the Axis Descriptor Lab.

This module owns the lab's authoritative score-to-label mapping and the
canonical axis ordering used by the standalone UI.  The definitions here are
kept intentionally aligned with the current Pipe-Works mud-server policy files:

- ``pipeworks_mud_server/data/worlds/pipeworks_web/policies/axes.yaml``
- ``pipeworks_mud_server/data/worlds/pipeworks_web/policies/thresholds.yaml``

The lab remains a standalone tool, so it does not import those files directly.
Instead, this module mirrors their current ordering and threshold labels so the
lab's "auto-label" behaviour matches the mud server's policy semantics as
closely as possible.

Exports
-------
AXIS_ORDER : list[str]
    Canonical full-axis ordering, matching the mud-server ``axes.yaml`` key
    order for the bundled worlds.

AXIS_LABEL_ORDER : dict[str, list[str]]
    Canonical low-to-high ordinal label sequence for each axis, matching the
    mud-server ``ordering.values`` payloads.

RELABEL_POLICY : dict[str, list[tuple[float, float, str]]]
    Module-level constant mapping each known axis name to an ordered list of
    inclusive ``(min_score, max_score, label)`` tuples derived from the
    mud-server ``thresholds.yaml`` ranges.

apply_relabel_policy(payload) -> AxisPayload
    Walk the payload's axes, recompute labels from ``RELABEL_POLICY`` for
    known axes, and return a new ``AxisPayload`` with updated labels.
    Unknown axes are passed through unchanged.  Scores are never modified.

Design notes
------------
The policy table lives in its own module (rather than inline in a route
handler) so that:

1. Unit tests can validate the mirrored mud-server policy in isolation
   without hitting the HTTP layer.
2. The table can be imported by other modules (e.g. future CLI tools)
   without pulling in all of FastAPI.
3. ``main.py`` stays a thin routing layer — it calls
   ``apply_relabel_policy(payload)`` and returns the result.
"""

from __future__ import annotations

from app.schema import AxisPayload

# -----------------------------------------------------------------------------
# Canonical axis ordering
# -----------------------------------------------------------------------------
#
# ``AXIS_ORDER`` mirrors the current mud-server ``axes.yaml`` order.  The
# frontend uses it to render rows deterministically instead of depending on
# incoming JSON insertion order.

AXIS_ORDER: list[str] = [
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

# Each label list is low → high, matching the mud-server ``ordering.values``
# payload for that axis.
AXIS_LABEL_ORDER: dict[str, list[str]] = {
    "physique": ["frail", "hunched", "skinny", "wiry", "broad", "stocky"],
    "wealth": ["poor", "modest", "well-kept", "wealthy", "decadent"],
    "health": ["sickly", "limping", "weary", "scarred", "hale"],
    "demeanor": ["timid", "suspicious", "resentful", "alert", "proud"],
    "age": ["young", "middle-aged", "old", "ancient"],
    "facial_signal": [
        "understated",
        "pronounced",
        "exaggerated",
        "asymmetrical",
        "weathered",
        "soft-featured",
        "sharp-featured",
    ],
    "legitimacy": ["sanctioned", "tolerated", "questioned", "illicit"],
    "visibility": ["hidden", "discrete", "routine", "conspicuous"],
    "moral_load": ["neutral", "burdened", "conflicted", "corrosive"],
    "dependency": ["optional", "useful", "necessary", "unavoidable"],
    "risk_exposure": ["benign", "straining", "hazardous", "eroding"],
}

# -----------------------------------------------------------------------------
# Policy table
# -----------------------------------------------------------------------------
#
# Structure: axis_name -> ordered list of inclusive (min_score, max_score, label)
#
# These ranges mirror the mud-server ``thresholds.yaml`` values exactly.  The
# server currently resolves labels with inclusive ``min``/``max`` checks, so
# the lab uses the same contract here rather than the previous simplified
# ``score < upper_bound`` approximation.

type PolicyRange = tuple[float, float, str]

RELABEL_POLICY: dict[str, list[PolicyRange]] = {
    "physique": [
        (0.00, 0.16, "frail"),
        (0.17, 0.32, "hunched"),
        (0.33, 0.48, "skinny"),
        (0.49, 0.64, "wiry"),
        (0.65, 0.80, "broad"),
        (0.81, 1.00, "stocky"),
    ],
    "wealth": [
        (0.00, 0.19, "poor"),
        (0.20, 0.39, "modest"),
        (0.40, 0.59, "well-kept"),
        (0.60, 0.79, "wealthy"),
        (0.80, 1.00, "decadent"),
    ],
    "health": [
        (0.00, 0.19, "sickly"),
        (0.20, 0.39, "limping"),
        (0.40, 0.59, "weary"),
        (0.60, 0.79, "scarred"),
        (0.80, 1.00, "hale"),
    ],
    "demeanor": [
        (0.00, 0.19, "timid"),
        (0.20, 0.39, "suspicious"),
        (0.40, 0.59, "resentful"),
        (0.60, 0.79, "alert"),
        (0.80, 1.00, "proud"),
    ],
    "age": [
        (0.00, 0.24, "young"),
        (0.25, 0.49, "middle-aged"),
        (0.50, 0.74, "old"),
        (0.75, 1.00, "ancient"),
    ],
    "facial_signal": [
        (0.00, 0.14, "understated"),
        (0.15, 0.29, "pronounced"),
        (0.30, 0.44, "exaggerated"),
        (0.45, 0.59, "asymmetrical"),
        (0.60, 0.74, "weathered"),
        (0.75, 0.89, "soft-featured"),
        (0.90, 1.00, "sharp-featured"),
    ],
    "legitimacy": [
        (0.00, 0.24, "sanctioned"),
        (0.25, 0.49, "tolerated"),
        (0.50, 0.74, "questioned"),
        (0.75, 1.00, "illicit"),
    ],
    "visibility": [
        (0.00, 0.24, "hidden"),
        (0.25, 0.49, "discrete"),
        (0.50, 0.74, "routine"),
        (0.75, 1.00, "conspicuous"),
    ],
    "moral_load": [
        (0.00, 0.24, "neutral"),
        (0.25, 0.49, "burdened"),
        (0.50, 0.74, "conflicted"),
        (0.75, 1.00, "corrosive"),
    ],
    "dependency": [
        (0.00, 0.24, "optional"),
        (0.25, 0.49, "useful"),
        (0.50, 0.74, "necessary"),
        (0.75, 1.00, "unavoidable"),
    ],
    "risk_exposure": [
        (0.00, 0.24, "benign"),
        (0.25, 0.49, "straining"),
        (0.50, 0.74, "hazardous"),
        (0.75, 1.00, "eroding"),
    ],
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def resolve_axis_label(axis_name: str, score: float, fallback_label: str) -> str:
    """
    Resolve ``score`` to the mud-server-aligned label for ``axis_name``.

    The lookup uses inclusive ``min <= score <= max`` range checks to mirror
    the mud server's axis-value resolution logic.  If the axis is unknown, or
    if the score falls outside every mirrored range, ``fallback_label`` is
    returned unchanged.

    Returning the existing label for unmatched scores is intentional: the lab's
    schema requires a non-empty label string, while the mud server stores label
    absence as ``None`` in the database layer.

    Parameters
    ----------
    axis_name : str
        Axis key to resolve.
    score : float
        Normalised axis score in ``[0.0, 1.0]``.
    fallback_label : str
        Existing label to preserve when no mirrored range matches.

    Returns
    -------
    str
        The resolved canonical label, or ``fallback_label`` when the axis or
        score is not covered by the mirrored policy.
    """
    for min_score, max_score, label in RELABEL_POLICY.get(axis_name, []):
        if min_score <= score <= max_score:
            return label
    return fallback_label


def apply_relabel_policy(payload: AxisPayload) -> AxisPayload:
    """
    Recompute axis labels from the policy table and return an updated payload.

    For each axis in *payload*, if the axis name appears in
    :data:`RELABEL_POLICY`, the label is rewritten to the matching mirrored
    mud-server threshold range.  Unknown axes (those not in the policy table)
    are passed through with their existing labels intact.

    Scores are **never** modified — only labels change.  All non-axis fields
    (``policy_hash``, ``seed``, ``world_id``) are preserved verbatim.

    Parameters
    ----------
    payload : AxisPayload
        The current axis payload with scores and (possibly stale) labels.

    Returns
    -------
    AxisPayload
        A new payload instance with labels recomputed from scores.
    """
    updated_axes = {}

    for axis_name, axis_val in payload.axes.items():
        if axis_name in RELABEL_POLICY:
            new_label = resolve_axis_label(
                axis_name=axis_name,
                score=axis_val.score,
                fallback_label=axis_val.label,
            )
            # Create a new AxisValue with the updated label, same score.
            updated_axes[axis_name] = axis_val.model_copy(update={"label": new_label})
        else:
            # Unknown axis — pass through unchanged.
            updated_axes[axis_name] = axis_val

    # Return a new payload with updated axes; all other fields unchanged.
    return payload.model_copy(update={"axes": updated_axes})

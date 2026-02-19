"""
app/relabel_policy.py
-----------------------------------------------------------------------------
Server-side policy table and score-to-label mapping for the Axis Descriptor Lab.

This module owns the authoritative score-to-label mapping — a simple piecewise
function that translates a normalised axis score (0.0–1.0) into a human-readable
label string.  The policy is intentionally simple and Pipe-Works-flavoured: it
is NOT a substitute for a production policy engine but demonstrates how label
changes propagate through to LLM output.

Exports
-------
RELABEL_POLICY : dict[str, list[tuple[float, str]]]
    Module-level constant mapping each known axis name to an ordered list of
    (upper_bound_exclusive, label) pairs.  Thresholds are checked in order;
    the first pair whose upper_bound exceeds the score wins.  A final entry
    with upper_bound = 1.01 acts as the catch-all for scores up to 1.0.

apply_relabel_policy(payload) -> AxisPayload
    Walk the payload's axes, recompute labels from ``RELABEL_POLICY`` for
    known axes, and return a new ``AxisPayload`` with updated labels.
    Unknown axes are passed through unchanged.  Scores are never modified.

Design notes
------------
The policy table lives in its own module (rather than inline in a route
handler) so that:

1. Unit tests can validate the table structure (ascending thresholds,
   catch-all present) without hitting the HTTP layer.
2. The table can be imported by other modules (e.g. future CLI tools)
   without pulling in all of FastAPI.
3. ``main.py`` stays a thin routing layer — it calls
   ``apply_relabel_policy(payload)`` and returns the result.
"""

from __future__ import annotations

from app.schema import AxisPayload

# -----------------------------------------------------------------------------
# Policy table
# -----------------------------------------------------------------------------
#
# Structure:  axis_name -> list of (upper_bound_exclusive, label)
#
# For each axis, thresholds are checked in order; the first pair whose
# upper_bound exceeds the score wins.  A sentinel entry at 1.01 catches
# any score up to (and including) 1.0.
#
# Example — "age" axis:
#   score 0.10 → "young"    (0.10 < 0.25)
#   score 0.25 → "middle-aged" (0.25 is NOT < 0.25, so skip; 0.25 < 0.50 → hit)
#   score 0.80 → "ancient"  (0.80 < 1.01 → catch-all)

RELABEL_POLICY: dict[str, list[tuple[float, str]]] = {
    "age": [
        (0.25, "young"),
        (0.5, "middle-aged"),
        (0.75, "old"),
        (1.01, "ancient"),
    ],
    "demeanor": [
        (0.2, "cordial"),
        (0.4, "guarded"),
        (0.6, "resentful"),
        (0.8, "hostile"),
        (1.01, "menacing"),
    ],
    "dependency": [
        (0.33, "dispensable"),
        (0.66, "necessary"),
        (1.01, "indispensable"),
    ],
    "facial_signal": [
        (0.3, "open"),
        (0.6, "asymmetrical"),
        (1.01, "closed"),
    ],
    "health": [
        (0.25, "vigorous"),
        (0.5, "weary"),
        (0.75, "ailing"),
        (1.01, "failing"),
    ],
    "legitimacy": [
        (0.25, "unchallenged"),
        (0.5, "tolerated"),
        (0.65, "questioned"),
        (0.8, "contested"),
        (1.01, "illegitimate"),
    ],
    "moral_load": [
        (0.3, "clear"),
        (0.6, "conflicted"),
        (1.01, "burdened"),
    ],
    "physique": [
        (0.3, "gaunt"),
        (0.45, "lean"),
        (0.55, "stocky"),
        (0.7, "hunched"),
        (1.01, "imposing"),
    ],
    "risk_exposure": [
        (0.33, "sheltered"),
        (0.66, "hazardous"),
        (1.01, "perilous"),
    ],
    "visibility": [
        (0.33, "obscure"),
        (0.66, "routine"),
        (1.01, "prominent"),
    ],
    "wealth": [
        (0.25, "destitute"),
        (0.45, "threadbare"),
        (0.55, "well-kept"),
        (0.75, "comfortable"),
        (1.01, "affluent"),
    ],
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def apply_relabel_policy(payload: AxisPayload) -> AxisPayload:
    """
    Recompute axis labels from the policy table and return an updated payload.

    For each axis in *payload*, if the axis name appears in
    :data:`RELABEL_POLICY`, the label is rewritten to the first entry whose
    ``upper_bound`` exceeds the axis score.  Unknown axes (those not in the
    policy table) are passed through with their existing labels intact.

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
            # Walk the threshold list; first match wins.
            new_label = axis_val.label  # fallback: keep existing
            for upper_bound, label in RELABEL_POLICY[axis_name]:
                if axis_val.score < upper_bound:
                    new_label = label
                    break
            # Create a new AxisValue with the updated label, same score.
            updated_axes[axis_name] = axis_val.model_copy(update={"label": new_label})
        else:
            # Unknown axis — pass through unchanged.
            updated_axes[axis_name] = axis_val

    # Return a new payload with updated axes; all other fields unchanged.
    return payload.model_copy(update={"axes": updated_axes})

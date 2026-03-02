/**
 * mod-axis-policy.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Canonical axis ordering and score quantisation helpers shared by the
 * standalone and chat slider UIs.
 *
 * The values in this module intentionally mirror the current Pipe-Works mud
 * server policy files for the bundled worlds:
 *
 *   - policies/axes.yaml        → full axis order
 *   - world.json active_axes    → chat/server preferred order when present
 *   - policies/thresholds.yaml  → hundredth-resolution score bands
 *
 * The lab stays decoupled from the mud-server repository, so these values are
 * mirrored here rather than imported at runtime.
 *
 * Imports: none (leaf module)
 */

/**
 * Full canonical axis order from the mud-server axis policy.
 *
 * @type {string[]}
 */
export const CANONICAL_AXIS_ORDER = [
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
];

/**
 * Axis score step used by the lab sliders and randomisation helpers.
 *
 * Mud-server thresholds are currently authored at hundredth precision
 * (`0.19`, `0.20`, etc.), so the lab uses `0.01` increments to stay aligned
 * with those published bands.
 *
 * @type {number}
 */
export const AXIS_SCORE_STEP = 0.01;

/**
 * Return axis keys in a deterministic order.
 *
 * If `preferredOrder` is provided, axes in that list are surfaced first in
 * that exact sequence.  Remaining known axes fall back to the canonical
 * mud-server order.  Unknown axes are appended in their original input order.
 *
 * This lets the standalone page use the canonical full-axis order while the
 * chat page can prioritize the selected world's `active_axes` order.
 *
 * @param {string[]} axisKeys - Raw axis keys from the payload.
 * @param {string[]|null|undefined} [preferredOrder=[]] - Optional order to
 *   prioritize before the canonical fallback order.
 * @returns {string[]} Ordered axis keys.
 */
export function orderAxisKeys(axisKeys, preferredOrder = []) {
  const preferred = Array.isArray(preferredOrder) ? preferredOrder : [];
  const preferredIndex = new Map(preferred.map((axis, index) => [axis, index]));
  const canonicalIndex = new Map(CANONICAL_AXIS_ORDER.map((axis, index) => [axis, index]));
  const inputIndex = new Map(axisKeys.map((axis, index) => [axis, index]));

  return [...axisKeys].sort((left, right) => {
    const leftPreferred = preferredIndex.has(left);
    const rightPreferred = preferredIndex.has(right);
    if (leftPreferred && rightPreferred) {
      return preferredIndex.get(left) - preferredIndex.get(right);
    }
    if (leftPreferred) return -1;
    if (rightPreferred) return 1;

    const leftCanonical = canonicalIndex.has(left);
    const rightCanonical = canonicalIndex.has(right);
    if (leftCanonical && rightCanonical) {
      return canonicalIndex.get(left) - canonicalIndex.get(right);
    }
    if (leftCanonical) return -1;
    if (rightCanonical) return 1;

    return inputIndex.get(left) - inputIndex.get(right);
  });
}

/**
 * Quantise an arbitrary score to the lab's canonical axis resolution.
 *
 * @param {number} score - Raw score in `[0, 1]`.
 * @returns {number} Score rounded to the nearest `AXIS_SCORE_STEP`.
 */
export function quantiseAxisScore(score) {
  return Math.round(score / AXIS_SCORE_STEP) * AXIS_SCORE_STEP;
}

/**
 * Format a score for slider and badge display.
 *
 * @param {number} score - Axis score.
 * @returns {string} Fixed-width score string at the canonical resolution.
 */
export function formatAxisScore(score) {
  return score.toFixed(2);
}

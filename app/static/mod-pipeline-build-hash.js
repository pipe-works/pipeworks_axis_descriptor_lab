/**
 * mod-pipeline-build-hash.js
 * -----------------------------------------------------------------------------
 * Deterministic hash helpers for Pipeline Build inputs.
 *
 * Phase A provides stable normalization + SHA-256 helpers that later stages
 * will use for `axis_hash` and `compiler_input_hash` display.
 */

/**
 * Deterministically stringify an object by sorting keys recursively.
 *
 * @param {unknown} value - Any JSON-serializable value.
 * @returns {string}
 */
export function stableStringify(value) {
  const normalize = (node) => {
    if (Array.isArray(node)) return node.map(normalize);
    if (node && typeof node === "object") {
      const sorted = {};
      for (const key of Object.keys(node).sort()) {
        sorted[key] = normalize(node[key]);
      }
      return sorted;
    }
    return node;
  };
  return JSON.stringify(normalize(value));
}

/**
 * Compute SHA-256 hex digest for one UTF-8 string.
 *
 * @param {string} value - Input string.
 * @returns {Promise<string>}
 */
export async function sha256Hex(value) {
  const data = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Normalize one payload and return its SHA-256 hash.
 *
 * @param {unknown} value - Any JSON-serializable payload.
 * @returns {Promise<string>}
 */
export async function hashNormalizedPayload(value) {
  return sha256Hex(stableStringify(value));
}

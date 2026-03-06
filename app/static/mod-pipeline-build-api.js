/**
 * mod-pipeline-build-api.js
 * -----------------------------------------------------------------------------
 * Thin fetch helpers for Pipeline Build mud-server and local artifact requests.
 *
 * The helpers return parsed JSON and throw on non-2xx responses so callers can
 * apply consistent stage-level error handling.
 */

/**
 * Structured API error for pipeline fetch helpers.
 */
export class PipelineApiError extends Error {
  /**
   * @param {string} message - Human-readable message.
   * @param {number} status - HTTP status code.
   * @param {string|null} detail - Server-provided detail text.
   */
  constructor(message, status, detail = null) {
    super(message);
    this.name = "PipelineApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function requestJson(url, init, defaultMessage) {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = null;
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body);
    } catch {
      detail = await res.text().catch(() => null);
    }
    throw new PipelineApiError(
      `${defaultMessage} (${res.status})`,
      res.status,
      detail
    );
  }
  return res.json();
}

/**
 * Fetch the active mud-session status.
 *
 * @returns {Promise<object>}
 */
export async function fetchMudSession() {
  return requestJson("/api/mud/session", undefined, "session request failed");
}

/**
 * Fetch mud worlds available to the authenticated session.
 *
 * @returns {Promise<object>}
 */
export async function fetchMudWorlds() {
  return requestJson("/api/mud/worlds", undefined, "world list request failed");
}

/**
 * Persist selected world id to the mud client session.
 *
 * @param {string} worldId - Mud world id.
 * @returns {Promise<object>}
 */
export async function selectMudWorld(worldId) {
  return requestJson(
    "/api/mud/select-world",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ world_id: worldId }),
    },
    "world select failed"
  );
}

/**
 * Fetch mud world configuration metadata for one world id.
 *
 * @param {string} worldId - Mud world id.
 * @returns {Promise<object>}
 */
export async function fetchMudWorldConfig(worldId) {
  return requestJson(
    `/api/mud/world-config/${encodeURIComponent(worldId)}`,
    undefined,
    "world config request failed"
  );
}

/**
 * Fetch canonical image policy bundle metadata for one world id.
 *
 * @param {string} worldId - Mud world id.
 * @returns {Promise<object>}
 */
export async function fetchMudImagePolicyBundle(worldId) {
  return requestJson(
    `/api/mud/world-image-policy-bundle/${encodeURIComponent(worldId)}`,
    undefined,
    "image policy bundle request failed"
  );
}

/**
 * Fetch local axis payload preset listing.
 *
 * @returns {Promise<object>}
 */
export async function fetchLocalAxisPayloads() {
  return requestJson(
    "/api/artifacts/local/axis-payloads",
    undefined,
    "axis preset list request failed"
  );
}

/**
 * Fetch one local axis payload preset document.
 *
 * @param {string} name - Axis preset stem.
 * @returns {Promise<object>}
 */
export async function fetchLocalAxisPayload(name) {
  return requestJson(
    `/api/artifacts/local/axis-payloads/${encodeURIComponent(name)}`,
    undefined,
    "axis preset load request failed"
  );
}

/**
 * Relabel one axis payload using policy rules.
 *
 * @param {object} payload - Axis payload body.
 * @returns {Promise<object>}
 */
export async function relabelAxisPayload(payload) {
  return requestJson(
    "/api/relabel",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    "axis relabel request failed"
  );
}

/**
 * Compile canonical image prompt through mud-server proxy endpoint.
 *
 * @param {object} body - Canonical compile request body.
 * @returns {Promise<object>}
 */
export async function compileImagePrompt(body) {
  return requestJson(
    "/api/mud/compile-image-prompt",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    "image compile request failed"
  );
}

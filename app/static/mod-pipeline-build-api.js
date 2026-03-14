/**
 * mod-pipeline-build-api.js
 * -----------------------------------------------------------------------------
 * Thin fetch helpers for Pipeline Build mud-server requests.
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
   * @param {string|null} code - Stable machine-readable pipeline error code.
   * @param {string|null} stage - Pipeline stage key associated with the error.
   */
  constructor(message, status, detail = null, code = null, stage = null) {
    super(message);
    this.name = "PipelineApiError";
    this.status = status;
    this.detail = detail;
    this.code = code;
    this.stage = stage;
  }
}

async function requestJson(url, init, defaultMessage) {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = null;
    let code = null;
    let stage = null;
    try {
      const body = await res.json();
      if (body && typeof body === "object") {
        if (typeof body.detail === "string") {
          detail = body.detail;
        } else if (body.detail && typeof body.detail === "object") {
          detail = typeof body.detail.detail === "string"
            ? body.detail.detail
            : JSON.stringify(body.detail);
          code = typeof body.detail.code === "string" ? body.detail.code : null;
          stage = typeof body.detail.stage === "string" ? body.detail.stage : null;
        } else {
          detail = JSON.stringify(body);
        }

        // New pipeline endpoints return top-level code/stage fields.
        if (typeof body.code === "string") code = body.code;
        if (typeof body.stage === "string") stage = body.stage;
      } else {
        detail = JSON.stringify(body);
      }
    } catch {
      detail = await res.text().catch(() => null);
    }
    throw new PipelineApiError(
      `${defaultMessage} (${res.status})`,
      res.status,
      detail,
      code,
      stage
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
 * Fetch aggregated stage-1/2 metadata for Pipeline Build.
 *
 * @param {string} worldId - Mud world id.
 * @returns {Promise<object>}
 */
export async function fetchPipelineBuildBootstrap(worldId) {
  return requestJson(
    `/api/mud/pipeline-build/bootstrap/${encodeURIComponent(worldId)}`,
    undefined,
    "pipeline bootstrap request failed"
  );
}

/**
 * Resolve selection metadata for Pipeline Build stages 5-7.
 *
 * @param {object} body - Resolve request body.
 * @returns {Promise<object>}
 */
export async function resolvePipelineImageSelection(body) {
  return requestJson(
    "/api/mud/pipeline-build/resolve-image-selection",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    "pipeline resolve request failed"
  );
}

/**
 * Generate canonical condition-axis payload for Stage 4.
 *
 * @param {object} body - Axis generation request body.
 * @returns {Promise<object>}
 */
export async function generatePipelineConditionAxis(body) {
  return requestJson(
    "/api/mud/pipeline-build/generate-condition-axis",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    "pipeline condition-axis generate request failed"
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

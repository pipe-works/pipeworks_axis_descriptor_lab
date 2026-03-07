/**
 * mod-pipeline-build-state.js
 * -----------------------------------------------------------------------------
 * Shared client-side state for the Pipeline Build page.
 *
 * Phase B scope
 * ─────────────
 * This module tracks Session + World + Policy Bundle data and stage locking.
 * Later phases extend the same state object for identity, axis input, and
 * compile output details.
 */

/**
 * Locked stage order for the canonical pipeline workflow UI.
 *
 * @type {Array<string>}
 */
export const PIPELINE_STAGE_ORDER = [
  "session_world",
  "policy_bundle",
  "identity",
  "axis_input",
  "block_selection",
  "descriptor_tone",
  "composition_hashes",
  "compile_output",
];

/**
 * Stage status token enum values used by the navigator.
 *
 * @type {Record<string, string>}
 */
export const PIPELINE_STAGE_STATUS = {
  LOCKED: "locked",
  READY: "ready",
  ERROR: "error",
  COMPLETE: "complete",
};

/**
 * Mutable singleton state for the Pipeline Build page.
 *
 * @type {{
 *   activeStage: string,
 *   stageStatus: Record<string, string>,
 *   busy: boolean,
 *   lastError: string|null,
 *   session: { authenticated: boolean, modeKey: string|null, serverUrl: string|null },
  *   worlds: Array<object>,
 *   selectedWorldId: string|null,
 *   worldConfig: object|null,
 *   policyBundle: object|null,
 *   policySource: object|null,
 *   runtimeOptions: {
 *     species: Array<string>,
 *     gender: Array<string>,
 *     worldContextTags: Array<string>,
 *     occupationTags: Array<string>
 *   },
 *   identity: { species: string, gender: string },
 *   axis: {
 *     sourceMode: string,
 *     presets: Array<object>,
 *     selectedPresetName: string|null,
 *     selectedPresetMeta: object|null,
 *     payload: object|null
 *   },
 *   runtime: {
 *     worldContext: Array<string>,
 *     occupationSignals: Array<string>,
 *     modelId: string|null,
 *     aspectRatio: string|null,
 *     seed: number|null
 *   },
 *   compile: { requestBody: object|null, result: object|null },
 *   resolve: { requestBody: object|null, result: object|null },
 *   policyHash: string|null,
 *   axisHash: string|null,
 *   compilerInputHash: string|null,
 *   actionLog: Array<{ timestamp: string, level: string, message: string }>
 * }}
 */
export const pipelineBuildState = {
  activeStage: PIPELINE_STAGE_ORDER[0],
  stageStatus: Object.fromEntries(
    PIPELINE_STAGE_ORDER.map((stage) => [stage, PIPELINE_STAGE_STATUS.LOCKED])
  ),
  busy: false,
  lastError: null,
  session: {
    authenticated: false,
    modeKey: null,
    serverUrl: null,
  },
  worlds: [],
  selectedWorldId: null,
  worldConfig: null,
  policyBundle: null,
  policySource: null,
  runtimeOptions: {
    species: [],
    gender: ["male", "female"],
    worldContextTags: [],
    occupationTags: [],
  },
  identity: {
    species: "",
    gender: "male",
  },
  axis: {
    sourceMode: "preset",
    presets: [],
    selectedPresetName: null,
    selectedPresetMeta: null,
    payload: null,
  },
  runtime: {
    worldContext: [],
    occupationSignals: [],
    modelId: null,
    aspectRatio: null,
    seed: null,
  },
  compile: {
    requestBody: null,
    result: null,
  },
  resolve: {
    requestBody: null,
    result: null,
  },
  policyHash: null,
  axisHash: null,
  compilerInputHash: null,
  actionLog: [],
};

pipelineBuildState.stageStatus.session_world = PIPELINE_STAGE_STATUS.READY;

/**
 * Reset mutable pipeline state to initial defaults.
 *
 * @returns {void}
 */
export function resetPipelineBuildState() {
  pipelineBuildState.activeStage = PIPELINE_STAGE_ORDER[0];
  pipelineBuildState.busy = false;
  pipelineBuildState.lastError = null;
  pipelineBuildState.session.authenticated = false;
  pipelineBuildState.session.modeKey = null;
  pipelineBuildState.session.serverUrl = null;
  pipelineBuildState.worlds = [];
  pipelineBuildState.selectedWorldId = null;
  pipelineBuildState.worldConfig = null;
  pipelineBuildState.policyBundle = null;
  pipelineBuildState.policySource = null;
  pipelineBuildState.runtimeOptions.species = [];
  pipelineBuildState.runtimeOptions.gender = ["male", "female"];
  pipelineBuildState.runtimeOptions.worldContextTags = [];
  pipelineBuildState.runtimeOptions.occupationTags = [];
  pipelineBuildState.identity.species = "";
  pipelineBuildState.identity.gender = "male";
  pipelineBuildState.axis.sourceMode = "preset";
  pipelineBuildState.axis.presets = [];
  pipelineBuildState.axis.selectedPresetName = null;
  pipelineBuildState.axis.selectedPresetMeta = null;
  pipelineBuildState.axis.payload = null;
  pipelineBuildState.runtime.worldContext = [];
  pipelineBuildState.runtime.occupationSignals = [];
  pipelineBuildState.runtime.modelId = null;
  pipelineBuildState.runtime.aspectRatio = null;
  pipelineBuildState.runtime.seed = null;
  pipelineBuildState.compile.requestBody = null;
  pipelineBuildState.compile.result = null;
  pipelineBuildState.resolve.requestBody = null;
  pipelineBuildState.resolve.result = null;
  pipelineBuildState.policyHash = null;
  pipelineBuildState.axisHash = null;
  pipelineBuildState.compilerInputHash = null;
  pipelineBuildState.actionLog = [];

  for (const stage of PIPELINE_STAGE_ORDER) {
    pipelineBuildState.stageStatus[stage] =
      stage === "session_world" ? PIPELINE_STAGE_STATUS.READY : PIPELINE_STAGE_STATUS.LOCKED;
  }
}

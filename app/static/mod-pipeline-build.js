/**
 * mod-pipeline-build.js
 * -----------------------------------------------------------------------------
 * Pipeline Build page controller.
 *
 * Phase D scope
 * ─────────────
 * Implements Stage 1 (Session + World), Stage 2 (Policy Bundle), Stage 3
 * (Character Identity), Stage 4 (Axis Input), and Stage 8 compile execution
 * against the canonical mud-server compile endpoint.
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import {
  PIPELINE_STAGE_ORDER,
  PIPELINE_STAGE_STATUS,
  pipelineBuildState,
  resetPipelineBuildState,
} from "./mod-pipeline-build-state.js";
import {
  PipelineApiError,
  compileImagePrompt,
  fetchLocalAxisPayload,
  fetchLocalAxisPayloads,
  fetchPipelineBuildBootstrap,
  fetchMudSession,
  fetchMudWorlds,
  relabelAxisPayload,
  resolvePipelineImageSelection,
  selectMudWorld,
} from "./mod-pipeline-build-api.js";
import { hashNormalizedPayload } from "./mod-pipeline-build-hash.js";
import { renderSourceHint } from "./mod-source-paths.js";

const STAGE_LABEL = {
  session_world: "Session + World",
  policy_bundle: "Policy Bundle",
  identity: "Character Identity",
  axis_input: "Axis Input",
  block_selection: "Block Selection",
  descriptor_tone: "Descriptor + Tone",
  composition_hashes: "Composition + Hashes",
  compile_output: "Compile + Output",
};

const MAX_ACTION_LOG_ENTRIES = 24;
const AXIS_HASH_INPUT_FIELDS = [
  "world_id",
  "policy_hash",
  "seed",
  "axes[*].label",
  "axes[*].score",
];
const COMPILER_INPUT_HASH_FIELDS = [
  "world_id",
  "species",
  "gender",
  "axes",
  "world_context",
  "occupation_signals",
  "model_id",
  "aspect_ratio",
  "seed",
];

function sanitiseActionLogMessage(rawMessage) {
  const compact = String(rawMessage || "").replace(/\s+/g, " ").trim();
  let redacted = compact.replace(
    /\b(password|token|secret|authorization)\b\s*[:=]\s*([^\s,;]+)/gi,
    "$1=[redacted]"
  );
  redacted = redacted.replace(/\bBearer\s+[A-Za-z0-9\-._~+/]+=*/gi, "Bearer [redacted]");
  redacted = redacted.replace(/\b(api[_-]?key)\b\s*[:=]\s*([^\s,;]+)/gi, "$1=[redacted]");
  if (!redacted) return "(message omitted)";
  return redacted.slice(0, 280);
}

function appendActionLog(message, level = "info") {
  const timestamp = new Date().toISOString();
  const safeMessage = sanitiseActionLogMessage(message);
  pipelineBuildState.actionLog.unshift({
    timestamp,
    level,
    message: safeMessage,
  });
  if (pipelineBuildState.actionLog.length > MAX_ACTION_LOG_ENTRIES) {
    pipelineBuildState.actionLog.length = MAX_ACTION_LOG_ENTRIES;
  }
}

function setStageStatus(stageKey, status) {
  pipelineBuildState.stageStatus[stageKey] = status;
}

function lockAfterAxis() {
  setStageStatus("block_selection", PIPELINE_STAGE_STATUS.LOCKED);
  setStageStatus("descriptor_tone", PIPELINE_STAGE_STATUS.LOCKED);
  setStageStatus("composition_hashes", PIPELINE_STAGE_STATUS.LOCKED);
  setStageStatus("compile_output", PIPELINE_STAGE_STATUS.LOCKED);
}

function isIdentityValid() {
  const species = String(pipelineBuildState.identity.species || "").trim();
  const gender = String(pipelineBuildState.identity.gender || "").trim();
  return Boolean(species) && (gender === "male" || gender === "female");
}

function isAxisPayloadValid(payload) {
  return validateAxisPayloadSchema(payload).length === 0;
}

function validateAxisPayloadSchema(payload) {
  const errors = [];

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    errors.push("Payload must be a JSON object.");
    return errors;
  }
  if (!payload.axes || typeof payload.axes !== "object" || Array.isArray(payload.axes)) {
    errors.push("Payload must include an 'axes' object.");
    return errors;
  }

  const axisEntries = Object.entries(payload.axes);
  if (axisEntries.length === 0) {
    errors.push("Payload 'axes' object must include at least one axis.");
  }

  for (const [axisName, axisValue] of axisEntries) {
    if (!axisValue || typeof axisValue !== "object" || Array.isArray(axisValue)) {
      errors.push(`Axis '${axisName}' must be an object.`);
      continue;
    }
    if (typeof axisValue.label !== "string" || !axisValue.label.trim()) {
      errors.push(`Axis '${axisName}' label must be a non-empty string.`);
    }
    const score = Number(axisValue.score);
    if (!Number.isFinite(score) || score < 0 || score > 1) {
      errors.push(`Axis '${axisName}' score must be a number between 0 and 1.`);
    }
  }

  if (
    payload.world_id !== undefined &&
    (typeof payload.world_id !== "string" || !payload.world_id.trim())
  ) {
    errors.push("'world_id' must be a non-empty string when present.");
  }
  if (
    payload.policy_hash !== undefined &&
    payload.policy_hash !== null &&
    typeof payload.policy_hash !== "string"
  ) {
    errors.push("'policy_hash' must be a string or null when present.");
  }
  if (
    payload.seed !== undefined &&
    payload.seed !== null &&
    !Number.isInteger(Number(payload.seed))
  ) {
    errors.push("'seed' must be an integer or null when present.");
  }

  return errors;
}

function quantizeScore(value) {
  const clamped = Math.min(1, Math.max(0, Number(value)));
  return Math.round(clamped * 100) / 100;
}

function parseCsvList(raw) {
  return String(raw || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function worldIdsMatch() {
  const payloadWorldId = pipelineBuildState.axis.payload?.world_id;
  const selectedWorldId = pipelineBuildState.selectedWorldId;
  if (!payloadWorldId || !selectedWorldId) return true;
  return payloadWorldId === selectedWorldId;
}

function buildCompileRequest() {
  if (!pipelineBuildState.selectedWorldId) return null;
  if (pipelineBuildState.stageStatus.policy_bundle !== PIPELINE_STAGE_STATUS.COMPLETE) return null;
  if (!isIdentityValid()) return null;
  if (!isAxisPayloadValid(pipelineBuildState.axis.payload)) return null;
  if (!worldIdsMatch()) return null;

  const runtime = pipelineBuildState.runtime;
  return {
    world_id: pipelineBuildState.selectedWorldId,
    species: pipelineBuildState.identity.species,
    gender: pipelineBuildState.identity.gender,
    axes: pipelineBuildState.axis.payload.axes,
    world_context: runtime.worldContext,
    occupation_signals: runtime.occupationSignals,
    model_id: runtime.modelId,
    aspect_ratio: runtime.aspectRatio,
    seed: runtime.seed,
  };
}

function buildResolveRequest() {
  if (!pipelineBuildState.selectedWorldId) return null;
  if (pipelineBuildState.stageStatus.policy_bundle !== PIPELINE_STAGE_STATUS.COMPLETE) return null;
  if (!isIdentityValid()) return null;
  if (!isAxisPayloadValid(pipelineBuildState.axis.payload)) return null;
  if (!worldIdsMatch()) return null;

  const runtime = pipelineBuildState.runtime;
  return {
    world_id: pipelineBuildState.selectedWorldId,
    species: pipelineBuildState.identity.species,
    gender: pipelineBuildState.identity.gender,
    axes: pipelineBuildState.axis.payload.axes,
    world_context: runtime.worldContext,
    occupation_signals: runtime.occupationSignals,
  };
}

function applyStageProgression() {
  const isAuthenticated = pipelineBuildState.session.authenticated;
  const hasWorld = Boolean(pipelineBuildState.selectedWorldId);
  const policyStatus = pipelineBuildState.stageStatus.policy_bundle;
  const identityValid = isIdentityValid();
  const axisValid = isAxisPayloadValid(pipelineBuildState.axis.payload);
  const hasCompileResult = Boolean(pipelineBuildState.compile.result);
  const hasResolvePreview = Boolean(pipelineBuildState.resolve.result);

  if (!isAuthenticated) {
    setStageStatus("session_world", PIPELINE_STAGE_STATUS.READY);
    setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.LOCKED);
    setStageStatus("identity", PIPELINE_STAGE_STATUS.LOCKED);
    setStageStatus("axis_input", PIPELINE_STAGE_STATUS.LOCKED);
    lockAfterAxis();
    pipelineBuildState.activeStage = "session_world";
    return;
  }

  setStageStatus(
    "session_world",
    hasWorld ? PIPELINE_STAGE_STATUS.COMPLETE : PIPELINE_STAGE_STATUS.READY
  );

  if (!hasWorld) {
    setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.LOCKED);
    setStageStatus("identity", PIPELINE_STAGE_STATUS.LOCKED);
    setStageStatus("axis_input", PIPELINE_STAGE_STATUS.LOCKED);
    lockAfterAxis();
    pipelineBuildState.activeStage = "session_world";
    return;
  }

  if (policyStatus !== PIPELINE_STAGE_STATUS.COMPLETE) {
    setStageStatus("identity", PIPELINE_STAGE_STATUS.LOCKED);
    setStageStatus("axis_input", PIPELINE_STAGE_STATUS.LOCKED);
    lockAfterAxis();
    pipelineBuildState.activeStage = "policy_bundle";
    return;
  }

  setStageStatus(
    "identity",
    identityValid ? PIPELINE_STAGE_STATUS.COMPLETE : PIPELINE_STAGE_STATUS.READY
  );

  if (!identityValid) {
    setStageStatus("axis_input", PIPELINE_STAGE_STATUS.LOCKED);
    lockAfterAxis();
    pipelineBuildState.activeStage = "identity";
    return;
  }

  setStageStatus(
    "axis_input",
    axisValid ? PIPELINE_STAGE_STATUS.COMPLETE : PIPELINE_STAGE_STATUS.READY
  );

  if (axisValid) {
    const previewStatus = hasCompileResult || hasResolvePreview
      ? PIPELINE_STAGE_STATUS.COMPLETE
      : PIPELINE_STAGE_STATUS.READY;
    setStageStatus("block_selection", previewStatus);
    setStageStatus("descriptor_tone", previewStatus);
    setStageStatus("composition_hashes", previewStatus);
    setStageStatus(
      "compile_output",
      hasCompileResult ? PIPELINE_STAGE_STATUS.COMPLETE : PIPELINE_STAGE_STATUS.READY
    );
    if (hasCompileResult) {
      pipelineBuildState.activeStage = "compile_output";
    } else if (hasResolvePreview) {
      pipelineBuildState.activeStage = "composition_hashes";
    } else {
      pipelineBuildState.activeStage = "axis_input";
    }
  } else {
    lockAfterAxis();
    pipelineBuildState.activeStage = "axis_input";
  }
}

async function recomputeHashes() {
  const axisPayload = pipelineBuildState.axis.payload;
  pipelineBuildState.axisHash =
    isAxisPayloadValid(axisPayload) ? await hashNormalizedPayload(axisPayload) : null;

  const resolveRequest = buildResolveRequest();
  pipelineBuildState.resolve.requestBody = resolveRequest;

  const compileRequest = buildCompileRequest();
  pipelineBuildState.compile.requestBody = compileRequest;
  pipelineBuildState.compilerInputHash =
    compileRequest ? await hashNormalizedPayload(compileRequest) : null;
}

function clearResolvePreview() {
  pipelineBuildState.resolve.result = null;
}

function applyStageErrorFromPipelineApi(err) {
  if (!(err instanceof PipelineApiError)) return;
  const stageKey = typeof err.stage === "string" ? err.stage : null;
  if (!stageKey) return;
  if (!Object.prototype.hasOwnProperty.call(pipelineBuildState.stageStatus, stageKey)) return;
  setStageStatus(stageKey, PIPELINE_STAGE_STATUS.ERROR);
}

async function refreshResolvePreview({ quiet = false } = {}) {
  const requestBody = buildResolveRequest();
  pipelineBuildState.resolve.requestBody = requestBody;
  if (!requestBody) {
    clearResolvePreview();
    applyStageProgression();
    return;
  }

  try {
    const result = await resolvePipelineImageSelection(requestBody);
    pipelineBuildState.resolve.result = result;
    pipelineBuildState.lastError = null;

    if (result.policy_hash) {
      pipelineBuildState.policyHash = String(result.policy_hash);
    }
    if (result.axis_hash) {
      pipelineBuildState.axisHash = String(result.axis_hash);
    }
    if (result.compiler_input_hash) {
      pipelineBuildState.compilerInputHash = String(result.compiler_input_hash);
    }

    applyStageProgression();
    if (!quiet) {
      setStatus("Pipeline Build — resolved canonical selection preview.");
    }
    appendActionLog("Resolved stage 5-7 selection preview.");
  } catch (err) {
    const detail =
      err instanceof PipelineApiError
        ? err.detail || err.message
        : err?.message || String(err);

    clearResolvePreview();
    applyStageErrorFromPipelineApi(err);
    applyStageProgression();

    if (err instanceof PipelineApiError && err.status === 401) {
      applyUnauthenticatedState(detail);
      setStatus("Pipeline Build — mud session expired. Please reconnect.");
      return;
    }

    pipelineBuildState.lastError = detail;
    if (!quiet) {
      setStatus(`Pipeline Build — resolve preview failed: ${detail}`);
    }
    appendActionLog(`Resolve preview failed: ${detail}`, "warn");
  }
}

function renderStageStatuses() {
  if (!dom.pipelineStageList) return;
  const rows = dom.pipelineStageList.querySelectorAll("li[data-stage]");
  for (const row of rows) {
    const stageKey = row.dataset.stage;
    if (!stageKey) continue;

    const stageStatus =
      pipelineBuildState.stageStatus[stageKey] || PIPELINE_STAGE_STATUS.LOCKED;
    const badge = row.querySelector("span.badge");
    if (badge) {
      badge.textContent = stageStatus;
      if (stageStatus === PIPELINE_STAGE_STATUS.COMPLETE) {
        badge.className = "badge badge--active";
      } else if (stageStatus === PIPELINE_STAGE_STATUS.READY) {
        badge.className = "badge";
      } else {
        badge.className = "badge badge--muted";
      }
    }

    row.classList.toggle("is-stage-active", stageKey === pipelineBuildState.activeStage);
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `${STAGE_LABEL[stageKey] || stageKey} stage (${stageStatus})`);
  }
}

function focusStageControl(stageKey) {
  const focusTargetMap = {
    session_world: () => dom.pipelineWorldSelect,
    policy_bundle: () => dom.pipelinePolicyRefresh,
    identity: () => dom.pipelineSpeciesInput,
    axis_input: () => dom.pipelineAxisSourceMode,
    block_selection: () => dom.pipelineBlockSelectionSummary,
    descriptor_tone: () => dom.pipelineDescriptorToneSummary,
    composition_hashes: () => dom.pipelineCompositionPreview,
    compile_output: () => dom.pipelineCompileButton,
  };
  const target = focusTargetMap[stageKey]?.();
  if (target && typeof target.focus === "function") {
    target.focus();
    return true;
  }
  return false;
}

function renderSessionHeader() {
  if (dom.pipelineSessionBadge) {
    dom.pipelineSessionBadge.textContent = pipelineBuildState.session.authenticated
      ? "authenticated"
      : "not connected";
    dom.pipelineSessionBadge.className = pipelineBuildState.session.authenticated
      ? "badge badge--active"
      : "badge badge--muted";
  }

  if (dom.pipelineModeText) {
    const mode = pipelineBuildState.session.modeKey || "unknown";
    const server = pipelineBuildState.session.serverUrl || "(no server)";
    dom.pipelineModeText.textContent = `Mode: ${mode} · Server: ${server}`;
  }

  if (dom.pipelineWorldSourceHint) {
    const server = pipelineBuildState.session.serverUrl || "(no server)";
    dom.pipelineWorldSourceHint.textContent =
      `Source: mud server canonical @ ${server} (/api/mud/worlds, /api/mud/world-config/{world_id}).`;
  }
  if (dom.pipelinePolicySourceHint) {
    const server = pipelineBuildState.session.serverUrl || "(no server)";
    dom.pipelinePolicySourceHint.textContent =
      `Source: mud server canonical @ ${server} (/api/mud/world-image-policy-bundle/{world_id}).`;
  }
}

function renderWorldSelect() {
  if (!dom.pipelineWorldSelect) return;

  const worlds = Array.isArray(pipelineBuildState.worlds) ? pipelineBuildState.worlds : [];
  const selectedWorldId = pipelineBuildState.selectedWorldId || "";

  dom.pipelineWorldSelect.innerHTML = '<option value="">— select world —</option>';
  for (const world of worlds) {
    const worldId = String(world.world_id || "");
    if (!worldId) continue;

    const option = document.createElement("option");
    option.value = worldId;
    option.textContent = String(world.name || worldId);
    dom.pipelineWorldSelect.appendChild(option);
  }

  dom.pipelineWorldSelect.value = selectedWorldId;
  dom.pipelineWorldSelect.disabled =
    !pipelineBuildState.session.authenticated || pipelineBuildState.busy;
}

function renderSessionSummary() {
  if (!dom.pipelineSessionSummary) return;

  const worldCount = Array.isArray(pipelineBuildState.worlds)
    ? pipelineBuildState.worlds.length
    : 0;
  const lines = [
    `active_stage: ${STAGE_LABEL[pipelineBuildState.activeStage] || pipelineBuildState.activeStage}`,
    `authenticated: ${pipelineBuildState.session.authenticated ? "yes" : "no"}`,
    `world_count: ${worldCount}`,
    `selected_world: ${pipelineBuildState.selectedWorldId || "(none)"}`,
  ];

  if (pipelineBuildState.lastError) {
    lines.push(`last_error: ${pipelineBuildState.lastError}`);
  }

  dom.pipelineSessionSummary.textContent = lines.join("\n");
}

function renderWorldConfig() {
  if (!dom.pipelineWorldConfig) return;

  const cfg = pipelineBuildState.worldConfig;
  if (!cfg) {
    dom.pipelineWorldConfig.textContent =
      "World configuration metadata will appear here.";
    return;
  }

  const activeAxes = Array.isArray(cfg.active_axes)
    ? cfg.active_axes.join(", ")
    : "(none)";

  dom.pipelineWorldConfig.textContent = [
    "World Config",
    `name: ${cfg.name || pipelineBuildState.selectedWorldId || "(unknown)"}`,
    `world_id: ${cfg.world_id || pipelineBuildState.selectedWorldId || "(unknown)"}`,
    `version: ${cfg.version || "(unknown)"}`,
    `active_axes: ${activeAxes}`,
  ].join("\n");
}

function renderPolicyBundleSummary() {
  if (!dom.pipelinePolicySummary) return;

  const bundle = pipelineBuildState.policyBundle;
  if (!bundle) {
    dom.pipelinePolicySummary.textContent =
      "Policy bundle metadata will appear here.";
    return;
  }

  const composition = Array.isArray(bundle.composition_order)
    ? bundle.composition_order.join(" -> ")
    : "(none)";
  const requiredInputs = Array.isArray(bundle.required_runtime_inputs)
    ? bundle.required_runtime_inputs.join(", ")
    : "(none)";
  const missingComponents = Array.isArray(bundle.missing_components)
    ? bundle.missing_components
    : [];

  dom.pipelinePolicySummary.textContent = [
    "Policy Bundle",
    `policy_schema: ${bundle.policy_schema || "(unknown)"}`,
    `policy_bundle_id: ${bundle.policy_bundle_id || "(unknown)"}`,
    `policy_bundle_version: ${bundle.policy_bundle_version ?? "(unknown)"}`,
    `policy_hash: ${bundle.policy_hash || "(unknown)"}`,
    `composition_order: ${composition}`,
    `required_runtime_inputs: ${requiredInputs}`,
    `missing_components: ${missingComponents.length ? missingComponents.join(", ") : "none"}`,
  ].join("\n");
}

function renderIdentityControls() {
  if (dom.pipelineSpeciesInput) {
    dom.pipelineSpeciesInput.value = pipelineBuildState.identity.species;
    dom.pipelineSpeciesInput.disabled =
      pipelineBuildState.stageStatus.identity === PIPELINE_STAGE_STATUS.LOCKED;
  }
  if (dom.pipelineGenderSelect) {
    dom.pipelineGenderSelect.value = pipelineBuildState.identity.gender;
    dom.pipelineGenderSelect.disabled =
      pipelineBuildState.stageStatus.identity === PIPELINE_STAGE_STATUS.LOCKED;
  }
}

function renderAxisPresetSelect() {
  if (!dom.pipelineAxisPresetSelect) return;

  const selected = pipelineBuildState.axis.selectedPresetName || "";
  const presets = Array.isArray(pipelineBuildState.axis.presets)
    ? pipelineBuildState.axis.presets
    : [];

  dom.pipelineAxisPresetSelect.innerHTML = '<option value="">— select axis preset —</option>';
  for (const row of presets) {
    const option = document.createElement("option");
    option.value = row.name;
    option.textContent = row.name;
    dom.pipelineAxisPresetSelect.appendChild(option);
  }
  dom.pipelineAxisPresetSelect.value = selected;

  const axisLocked = pipelineBuildState.stageStatus.axis_input === PIPELINE_STAGE_STATUS.LOCKED;
  dom.pipelineAxisPresetSelect.disabled = axisLocked || pipelineBuildState.busy;
  if (dom.pipelineAxisLoadPreset) {
    dom.pipelineAxisLoadPreset.disabled = axisLocked || pipelineBuildState.busy;
  }

  renderSourceHint(
    dom.pipelineAxisPresetSourceHint,
    pipelineBuildState.axis.selectedPresetMeta,
    "axis_payload",
    "",
    "Source: select an axis preset to view path."
  );
}

function renderAxisJsonEditor() {
  if (!dom.pipelineAxisJson) return;

  const payload = pipelineBuildState.axis.payload;
  dom.pipelineAxisJson.value = payload ? JSON.stringify(payload, null, 2) : "";

  const axisLocked = pipelineBuildState.stageStatus.axis_input === PIPELINE_STAGE_STATUS.LOCKED;
  dom.pipelineAxisJson.disabled = axisLocked || pipelineBuildState.busy;
  if (dom.pipelineAxisRelabel) {
    dom.pipelineAxisRelabel.disabled =
      axisLocked || pipelineBuildState.busy || !isAxisPayloadValid(payload);
  }
}

function renderAxisManualPanel() {
  if (!dom.pipelineAxisManualPanel) return;

  const payload = pipelineBuildState.axis.payload;
  const axisLocked = pipelineBuildState.stageStatus.axis_input === PIPELINE_STAGE_STATUS.LOCKED;
  const manualMode = pipelineBuildState.axis.sourceMode === "manual";

  if (!payload?.axes || Object.keys(payload.axes).length === 0) {
    dom.pipelineAxisManualPanel.innerHTML =
      '<p class="placeholder-text">Load a preset or paste JSON to edit axis scores manually.</p>';
    return;
  }

  const rows = [];
  for (const [axisName, axisValue] of Object.entries(payload.axes)) {
    const row = document.createElement("div");
    row.className = "axis-row";

    const name = document.createElement("div");
    name.className = "axis-name";
    name.textContent = axisName;

    const range = document.createElement("input");
    range.type = "range";
    range.min = "0";
    range.max = "1";
    range.step = "0.01";
    range.value = String(quantizeScore(axisValue.score));
    range.disabled = !manualMode || axisLocked || pipelineBuildState.busy;

    const box = document.createElement("input");
    box.type = "number";
    box.className = "input input--sm";
    box.min = "0";
    box.max = "1";
    box.step = "0.01";
    box.value = String(quantizeScore(axisValue.score));
    box.disabled = !manualMode || axisLocked || pipelineBuildState.busy;

    range.addEventListener("input", async () => {
      const score = quantizeScore(range.value);
      box.value = String(score);
      payload.axes[axisName].score = score;
      await updateAxisStateFromPayload(payload, { syncAxisJsonEditor: true });
    });

    box.addEventListener("change", async () => {
      const score = quantizeScore(box.value);
      range.value = String(score);
      box.value = String(score);
      payload.axes[axisName].score = score;
      await updateAxisStateFromPayload(payload, { syncAxisJsonEditor: true });
    });

    row.appendChild(name);
    row.appendChild(range);
    row.appendChild(box);
    rows.push(row);
  }

  dom.pipelineAxisManualPanel.innerHTML = "";
  for (const row of rows) dom.pipelineAxisManualPanel.appendChild(row);
}

function renderAxisControls() {
  if (dom.pipelineAxisSourceMode) {
    dom.pipelineAxisSourceMode.value = pipelineBuildState.axis.sourceMode;
    dom.pipelineAxisSourceMode.disabled =
      pipelineBuildState.stageStatus.axis_input === PIPELINE_STAGE_STATUS.LOCKED;
  }

  renderAxisPresetSelect();
  renderAxisManualPanel();
}

function renderRuntimeControls() {
  if (dom.pipelineWorldContextInput) {
    dom.pipelineWorldContextInput.value = pipelineBuildState.runtime.worldContext.join(", ");
  }
  if (dom.pipelineOccupationSignalsInput) {
    dom.pipelineOccupationSignalsInput.value =
      pipelineBuildState.runtime.occupationSignals.join(", ");
  }
  if (dom.pipelineModelIdInput) {
    dom.pipelineModelIdInput.value = pipelineBuildState.runtime.modelId || "";
  }
  if (dom.pipelineAspectRatioInput) {
    dom.pipelineAspectRatioInput.value = pipelineBuildState.runtime.aspectRatio || "";
  }
  if (dom.pipelineSeedInput) {
    dom.pipelineSeedInput.value =
      pipelineBuildState.runtime.seed !== null ? String(pipelineBuildState.runtime.seed) : "";
  }
}

function renderBlockSelectionSummary() {
  if (!dom.pipelineBlockSelectionSummary) return;

  const bundle = pipelineBuildState.policyBundle;
  const result = pipelineBuildState.compile.result || pipelineBuildState.resolve.result;

  if (!bundle) {
    dom.pipelineBlockSelectionSummary.textContent =
      "Load policy bundle metadata to inspect block-family resolution.";
    return;
  }

  const speciesBlock =
    result?.selected_species_block_id || result?.selected_blocks?.species_canon_block || null;
  const clothingProfile = result?.selected_clothing_profile_id || null;
  const clothingSlots = result?.selected_blocks?.clothing_block;

  const lines = [
    "Block Selection",
    `species_canon_block: ${speciesBlock || "(pending resolve preview)"}`,
    `clothing_profile: ${clothingProfile || "(pending resolve preview)"}`,
  ];

  if (clothingSlots && typeof clothingSlots === "object" && !Array.isArray(clothingSlots)) {
    const slotPairs = Object.entries(clothingSlots).sort(([a], [b]) => a.localeCompare(b));
    if (slotPairs.length > 0) {
      lines.push("clothing_slots:");
      for (const [slotName, slotValue] of slotPairs) {
        lines.push(`  ${slotName}: ${slotValue}`);
      }
    }
  } else if (typeof clothingSlots === "string" && clothingSlots.trim()) {
    lines.push(`clothing_block: ${clothingSlots}`);
  } else if (!result) {
    lines.push("awaiting: resolve preview or compile response");
  }

  dom.pipelineBlockSelectionSummary.textContent = lines.join("\n");
}

function renderDescriptorToneSummary() {
  if (!dom.pipelineDescriptorToneSummary) return;

  const bundle = pipelineBuildState.policyBundle;
  const result = pipelineBuildState.compile.result || pipelineBuildState.resolve.result;

  if (!bundle) {
    dom.pipelineDescriptorToneSummary.textContent =
      "Load policy bundle metadata to inspect descriptor/tone resolution.";
    return;
  }

  const descriptorLayer =
    result?.selected_descriptor_layer_id || result?.descriptor_layer_id || null;
  const toneProfile = result?.selected_tone_profile_id || result?.tone_profile_id || null;

  const lines = [
    "Descriptor + Tone",
    `descriptor_layer: ${descriptorLayer || "(pending resolve preview)"}`,
    `tone_profile: ${toneProfile || "(pending resolve preview)"}`,
  ];
  if (!result) {
    lines.push("awaiting: resolve preview or compile response");
  }

  dom.pipelineDescriptorToneSummary.textContent = lines.join("\n");
}

function renderCompositionPreview() {
  if (!dom.pipelineCompositionPreview) return;

  const bundle = pipelineBuildState.policyBundle;
  const result = pipelineBuildState.compile.result || pipelineBuildState.resolve.result;
  const compositionOrder = Array.isArray(result?.composition_order) && result.composition_order.length
    ? result.composition_order
    : Array.isArray(bundle?.composition_order)
      ? bundle.composition_order
      : [];

  dom.pipelineCompositionPreview.textContent = [
    "Composition + Hashes",
    `composition_order: ${compositionOrder.length ? compositionOrder.join(" -> ") : "(none)"}`,
    `policy_hash: ${pipelineBuildState.policyHash || "(not loaded)"}`,
    `axis_hash: ${pipelineBuildState.axisHash || "(not computed)"}`,
    `axis_hash_inputs: ${AXIS_HASH_INPUT_FIELDS.join(", ")}`,
    `compiler_input_hash: ${pipelineBuildState.compilerInputHash || "(not computed)"}`,
    `compiler_hash_inputs: ${COMPILER_INPUT_HASH_FIELDS.join(", ")}`,
    "excluded_from_hashes: compiled_prompt",
    `source: ${
      pipelineBuildState.compile.result
        ? "compile_response"
        : pipelineBuildState.resolve.result
          ? "resolve_preview"
          : "policy_bundle/runtime"
    }`,
  ].join("\n");
}

function renderActionLog() {
  if (!dom.pipelineActionLog) return;

  const rows = Array.isArray(pipelineBuildState.actionLog) ? pipelineBuildState.actionLog : [];
  if (rows.length === 0) {
    dom.pipelineActionLog.textContent = "API and workflow actions will appear here.";
    return;
  }

  dom.pipelineActionLog.textContent = rows
    .map((row) => `[${row.timestamp}] (${row.level}) ${row.message}`)
    .join("\n");
}

function renderCompilePanels() {
  const request = pipelineBuildState.compile.requestBody;
  const result = pipelineBuildState.compile.result;
  const canCompile =
    Boolean(request) && pipelineBuildState.stageStatus.compile_output === PIPELINE_STAGE_STATUS.READY;

  if (dom.pipelineCompileRequest) {
    dom.pipelineCompileRequest.textContent = request
      ? JSON.stringify(request, null, 2)
      : "Compile request will appear here.";
  }

  if (dom.pipelineCompileButton) {
    dom.pipelineCompileButton.disabled = !canCompile || pipelineBuildState.busy;
  }
  if (dom.pipelineCopyResponseJson) {
    dom.pipelineCopyResponseJson.disabled = !result || pipelineBuildState.busy;
  }
  if (dom.pipelineExportResponseJson) {
    dom.pipelineExportResponseJson.disabled = !result || pipelineBuildState.busy;
  }

  if (dom.pipelineCompileResult) {
    dom.pipelineCompileResult.textContent = result?.compiled_prompt
      ? String(result.compiled_prompt)
      : "Compile output will appear here.";
  }

  if (dom.pipelineProvenancePanel) {
    if (!result) {
      dom.pipelineProvenancePanel.textContent = "Returned provenance fields will appear here.";
    } else {
      dom.pipelineProvenancePanel.textContent = [
        `world_id: ${result.world_id || pipelineBuildState.selectedWorldId || "(unknown)"}`,
        `policy_hash: ${result.policy_hash || pipelineBuildState.policyHash || "(unknown)"}`,
        `axis_hash: ${result.axis_hash || pipelineBuildState.axisHash || "(unknown)"}`,
      ].join("\n");
    }
  }
}

function renderBuildSummary() {
  if (!dom.pipelineBuildSummary) return;

  const payload = pipelineBuildState.axis.payload;
  const mismatch =
    payload?.world_id &&
    pipelineBuildState.selectedWorldId &&
    payload.world_id !== pipelineBuildState.selectedWorldId;

  const lines = [
    "Pipeline Build summary",
    `species: ${pipelineBuildState.identity.species || "(unset)"}`,
    `gender: ${pipelineBuildState.identity.gender || "(unset)"}`,
    `policy_hash: ${pipelineBuildState.policyHash || "(not loaded)"}`,
    `axis_hash: ${pipelineBuildState.axisHash || "(not computed)"}`,
    `compiler_input_hash: ${pipelineBuildState.compilerInputHash || "(not computed)"}`,
    `resolve_ready: ${pipelineBuildState.resolve.requestBody ? "yes" : "no"}`,
    `resolve_available: ${pipelineBuildState.resolve.result ? "yes" : "no"}`,
    `compile_ready: ${pipelineBuildState.compile.requestBody ? "yes" : "no"}`,
  ];
  if (mismatch) {
    lines.push("warning: axis payload world_id does not match selected world.");
  }
  if (pipelineBuildState.lastError) {
    lines.push(`last_error: ${pipelineBuildState.lastError}`);
  }

  dom.pipelineBuildSummary.textContent = lines.join("\n");
}

function renderStageEditorHint() {
  if (!dom.pipelineStageEditor) return;

  const policyStatus = pipelineBuildState.stageStatus.policy_bundle;
  const identityStatus = pipelineBuildState.stageStatus.identity;
  const axisStatus = pipelineBuildState.stageStatus.axis_input;

  if (!pipelineBuildState.session.authenticated) {
    dom.pipelineStageEditor.textContent =
      "Authenticate mud-server session first. Policy and downstream stages are locked.";
    return;
  }

  if (policyStatus === PIPELINE_STAGE_STATUS.ERROR) {
    dom.pipelineStageEditor.textContent =
      "Policy bundle has missing components. Fix policy requirements before continuing.";
    return;
  }

  if (policyStatus !== PIPELINE_STAGE_STATUS.COMPLETE) {
    dom.pipelineStageEditor.textContent =
      "Select world and load policy bundle metadata to unlock Character Identity.";
    return;
  }

  if (identityStatus !== PIPELINE_STAGE_STATUS.COMPLETE) {
    dom.pipelineStageEditor.textContent =
      "Set species and gender to unlock Axis Input.";
    return;
  }

  if (axisStatus !== PIPELINE_STAGE_STATUS.COMPLETE) {
    dom.pipelineStageEditor.textContent =
      "Load axis preset, edit manually, or paste valid axis JSON to unlock compile.";
    return;
  }

  if (!worldIdsMatch()) {
    dom.pipelineStageEditor.textContent =
      "Axis payload world_id must match selected world before compile.";
    return;
  }

  dom.pipelineStageEditor.textContent =
    "Resolve preview is active for stages 5-7. Run canonical compile for final prompt/provenance.";
}

function renderPipelinePanels({ syncAxisJsonEditor = false } = {}) {
  renderStageStatuses();
  renderSessionHeader();
  renderWorldSelect();
  renderSessionSummary();
  renderWorldConfig();
  renderPolicyBundleSummary();
  renderIdentityControls();
  renderAxisControls();
  if (syncAxisJsonEditor) {
    renderAxisJsonEditor();
  }
  renderBlockSelectionSummary();
  renderDescriptorToneSummary();
  renderRuntimeControls();
  renderCompositionPreview();
  renderCompilePanels();
  renderActionLog();
  renderBuildSummary();
  renderStageEditorHint();
}

function applyUnauthenticatedState(errorMessage = null, { preserveEnteredState = true } = {}) {
  pipelineBuildState.session.authenticated = false;

  // Re-auth flow should keep user-entered form/context state by default.
  // We only wipe cached world/policy/compile context when explicitly requested.
  if (!preserveEnteredState) {
    pipelineBuildState.worlds = [];
    pipelineBuildState.selectedWorldId = null;
    pipelineBuildState.worldConfig = null;
    pipelineBuildState.policyBundle = null;
    pipelineBuildState.policyHash = null;
    pipelineBuildState.axisHash = null;
    pipelineBuildState.compilerInputHash = null;
    pipelineBuildState.resolve.requestBody = null;
    pipelineBuildState.resolve.result = null;
    pipelineBuildState.compile.requestBody = null;
    pipelineBuildState.compile.result = null;
  }

  setStageStatus("session_world", PIPELINE_STAGE_STATUS.READY);
  setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.LOCKED);
  setStageStatus("identity", PIPELINE_STAGE_STATUS.LOCKED);
  setStageStatus("axis_input", PIPELINE_STAGE_STATUS.LOCKED);
  lockAfterAxis();
  pipelineBuildState.activeStage = "session_world";

  pipelineBuildState.lastError = errorMessage;
  if (preserveEnteredState) {
    appendActionLog(
      `${errorMessage || "Mud session unauthenticated."} Entered state preserved for re-auth.`,
      "warn"
    );
  } else {
    appendActionLog(errorMessage || "Mud session unauthenticated.", "warn");
  }
  renderPipelinePanels({ syncAxisJsonEditor: true });
}

function derivePreferredWorld(session, worlds) {
  const worldIds = new Set(worlds.map((row) => String(row.world_id || "")).filter(Boolean));
  const candidates = [
    pipelineBuildState.selectedWorldId,
    session.selected_world_id,
    worlds[0]?.world_id,
  ]
    .filter(Boolean)
    .map((value) => String(value));

  for (const candidate of candidates) {
    if (worldIds.has(candidate)) return candidate;
  }
  return null;
}

function applyRuntimeOptions(runtimeOptions) {
  const options = runtimeOptions && typeof runtimeOptions === "object" ? runtimeOptions : {};
  const species = Array.isArray(options.species)
    ? options.species.map((value) => String(value).trim()).filter(Boolean)
    : [];
  const gender = Array.isArray(options.gender)
    ? options.gender.map((value) => String(value).trim()).filter(Boolean)
    : [];
  const worldContextTags = Array.isArray(options.world_context_tags)
    ? options.world_context_tags.map((value) => String(value).trim()).filter(Boolean)
    : [];
  const occupationTags = Array.isArray(options.occupation_tags)
    ? options.occupation_tags.map((value) => String(value).trim()).filter(Boolean)
    : [];

  pipelineBuildState.runtimeOptions.species = species;
  pipelineBuildState.runtimeOptions.gender = gender.length > 0 ? gender : ["male", "female"];
  pipelineBuildState.runtimeOptions.worldContextTags = worldContextTags;
  pipelineBuildState.runtimeOptions.occupationTags = occupationTags;

  if (species.length > 0 && !species.includes(pipelineBuildState.identity.species)) {
    pipelineBuildState.identity.species = species[0];
  }
  if (
    pipelineBuildState.runtimeOptions.gender.length > 0 &&
    !pipelineBuildState.runtimeOptions.gender.includes(pipelineBuildState.identity.gender)
  ) {
    pipelineBuildState.identity.gender = pipelineBuildState.runtimeOptions.gender[0];
  }
}

async function loadPolicyBundleForWorld(worldId, { quiet = false } = {}) {
  pipelineBuildState.compile.result = null;
  clearResolvePreview();

  if (!worldId) {
    pipelineBuildState.policyBundle = null;
    pipelineBuildState.policyHash = null;
    setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.LOCKED);
    applyStageProgression();
    await recomputeHashes();
    await refreshResolvePreview({ quiet: true });
    renderPipelinePanels({ syncAxisJsonEditor: true });
    return;
  }

  setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.READY);
  pipelineBuildState.activeStage = "policy_bundle";

  const bootstrap = await fetchPipelineBuildBootstrap(worldId);
  const bundle = bootstrap.policy_bundle || null;
  pipelineBuildState.selectedWorldId = bootstrap.world_id || worldId;
  pipelineBuildState.policyBundle = bundle;
  pipelineBuildState.policyHash = bundle?.policy_hash || null;
  pipelineBuildState.worldConfig = bootstrap.world_summary?.world_config || null;
  applyRuntimeOptions(bootstrap.runtime_options);

  const missingComponents = Array.isArray(bundle?.missing_components)
    ? bundle.missing_components
    : [];
  if (missingComponents.length > 0) {
    setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.ERROR);
    pipelineBuildState.lastError =
      `Policy bundle missing components: ${missingComponents.join(", ")}`;
    if (!quiet) {
      setStatus(
        "Pipeline Build — policy bundle has missing components. Downstream stages remain locked."
      );
    }
    appendActionLog(
      `Policy bundle for '${worldId}' reported missing components: ${missingComponents.join(", ")}`,
      "warn"
    );
  } else {
    setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.COMPLETE);
    pipelineBuildState.lastError = null;
    if (!quiet) {
      setStatus("Pipeline Build — policy bundle loaded.");
    }
    appendActionLog(`Policy bundle loaded for '${worldId}'.`);
  }

  applyStageProgression();
  await recomputeHashes();
  await refreshResolvePreview({ quiet: true });
  renderPipelinePanels({ syncAxisJsonEditor: true });
}

async function applyWorldSelection(worldId, { quiet = false } = {}) {
  pipelineBuildState.compile.result = null;
  clearResolvePreview();

  if (!worldId) {
    pipelineBuildState.selectedWorldId = null;
    pipelineBuildState.worldConfig = null;
    pipelineBuildState.policyBundle = null;
    pipelineBuildState.policyHash = null;
    pipelineBuildState.compile.result = null;
    pipelineBuildState.resolve.result = null;
    applyStageProgression();
    await recomputeHashes();
    await refreshResolvePreview({ quiet: true });
    renderPipelinePanels({ syncAxisJsonEditor: true });
    return;
  }

  pipelineBuildState.busy = true;
  renderPipelinePanels({ syncAxisJsonEditor: false });

  try {
    await selectMudWorld(worldId);
    pipelineBuildState.selectedWorldId = worldId;
    await loadPolicyBundleForWorld(worldId, { quiet: true });
    if (!quiet) {
      setStatus(`Pipeline Build — selected world '${worldId}'.`);
    }
    appendActionLog(`Selected world '${worldId}'.`);
  } catch (err) {
    const detail =
      err instanceof PipelineApiError
        ? err.detail || err.message
        : err?.message || String(err);

    if (err instanceof PipelineApiError && err.status === 401) {
      applyUnauthenticatedState(detail);
      setStatus("Pipeline Build — mud session expired. Please reconnect.");
      return;
    }

    applyStageErrorFromPipelineApi(err);
    setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.ERROR);
    applyStageProgression();
    pipelineBuildState.lastError = detail;
    renderPipelinePanels({ syncAxisJsonEditor: true });
    setStatus(`Pipeline Build — failed to load world context: ${detail}`);
    appendActionLog(`World selection failed for '${worldId}': ${detail}`, "error");
  } finally {
    pipelineBuildState.busy = false;
    renderPipelinePanels({ syncAxisJsonEditor: false });
  }
}

async function refreshSessionAndWorlds({ quiet = false } = {}) {
  pipelineBuildState.busy = true;
  renderPipelinePanels({ syncAxisJsonEditor: false });

  try {
    const session = await fetchMudSession();
    pipelineBuildState.session.authenticated = Boolean(session.authenticated);
    pipelineBuildState.session.modeKey = session.mode_key || null;
    pipelineBuildState.session.serverUrl = session.active_server_url || null;

    if (!pipelineBuildState.session.authenticated) {
      applyUnauthenticatedState("Canonical mode requires an authenticated mud-server session.");
      if (!quiet) {
        setStatus("Pipeline Build — connect to mud server to unlock canonical workflow.");
      }
      appendActionLog("Mud session is unauthenticated.", "warn");
      return;
    }

    const worldsPayload = await fetchMudWorlds();
    const worlds = Array.isArray(worldsPayload.worlds) ? worldsPayload.worlds : [];
    pipelineBuildState.worlds = worlds;

    const preferredWorld = derivePreferredWorld(session, worlds);
    if (!preferredWorld) {
      pipelineBuildState.selectedWorldId = null;
      pipelineBuildState.worldConfig = null;
      pipelineBuildState.policyBundle = null;
      pipelineBuildState.policyHash = null;
      clearResolvePreview();
      pipelineBuildState.lastError = "No translation-enabled worlds returned by mud server.";
      applyStageProgression();
      await recomputeHashes();
      await refreshResolvePreview({ quiet: true });
      renderPipelinePanels({ syncAxisJsonEditor: true });
      return;
    }

    await applyWorldSelection(preferredWorld, { quiet: true });
    if (!quiet) {
      setStatus("Pipeline Build — session, world, and policy bundle refreshed.");
    }
    appendActionLog(
      `Session refreshed (${pipelineBuildState.session.modeKey || "unknown"} mode).`,
      "info"
    );
  } catch (err) {
    const detail =
      err instanceof PipelineApiError
        ? err.detail || err.message
        : err?.message || String(err);

    if (err instanceof PipelineApiError && err.status === 401) {
      applyUnauthenticatedState(detail);
      setStatus("Pipeline Build — mud session expired. Please reconnect.");
      return;
    }

    pipelineBuildState.lastError = detail;
    applyStageErrorFromPipelineApi(err);
    setStageStatus("session_world", PIPELINE_STAGE_STATUS.ERROR);
    setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.LOCKED);
    setStageStatus("identity", PIPELINE_STAGE_STATUS.LOCKED);
    setStageStatus("axis_input", PIPELINE_STAGE_STATUS.LOCKED);
    lockAfterAxis();
    renderPipelinePanels({ syncAxisJsonEditor: true });
    setStatus(`Pipeline Build — session/world refresh failed: ${detail}`);
    appendActionLog(`Session/world refresh failed: ${detail}`, "error");
  } finally {
    pipelineBuildState.busy = false;
    renderPipelinePanels({ syncAxisJsonEditor: false });
  }
}

async function refreshAxisPresetList() {
  try {
    const payload = await fetchLocalAxisPayloads();
    const presets = Array.isArray(payload.payloads) ? payload.payloads : [];
    pipelineBuildState.axis.presets = presets;

    if (!pipelineBuildState.axis.selectedPresetName && presets[0]) {
      pipelineBuildState.axis.selectedPresetName = presets[0].name;
      pipelineBuildState.axis.selectedPresetMeta = presets[0];
    }
    renderAxisPresetSelect();
    appendActionLog(`Loaded ${presets.length} local axis presets.`);
  } catch (err) {
    pipelineBuildState.lastError = err.message || String(err);
    renderPipelinePanels({ syncAxisJsonEditor: false });
    setStatus(`Pipeline Build — failed to load axis presets: ${pipelineBuildState.lastError}`);
    appendActionLog(`Axis preset list load failed: ${pipelineBuildState.lastError}`, "error");
  }
}

async function updateAxisStateFromPayload(payload, { syncAxisJsonEditor = false } = {}) {
  pipelineBuildState.compile.result = null;
  clearResolvePreview();
  pipelineBuildState.axis.payload = payload;
  applyStageProgression();
  await recomputeHashes();
  await refreshResolvePreview({ quiet: true });
  renderPipelinePanels({ syncAxisJsonEditor });
}

async function loadAxisPresetByName(name) {
  if (!name) {
    setStatus("Pipeline Build — choose an axis preset first.");
    return;
  }

  try {
    const doc = await fetchLocalAxisPayload(name);
    const parsed = JSON.parse(doc.content);

    pipelineBuildState.axis.selectedPresetName = name;
    pipelineBuildState.axis.selectedPresetMeta =
      pipelineBuildState.axis.presets.find((row) => row.name === name) || doc;

    await updateAxisStateFromPayload(parsed, { syncAxisJsonEditor: true });
    setStatus(`Pipeline Build — axis preset '${name}' loaded.`);
    appendActionLog(`Axis preset '${name}' loaded.`);
  } catch (err) {
    pipelineBuildState.lastError = err.message || String(err);
    renderPipelinePanels({ syncAxisJsonEditor: false });
    setStatus(`Pipeline Build — failed to load axis preset: ${pipelineBuildState.lastError}`);
    appendActionLog(`Axis preset '${name}' failed to load: ${pipelineBuildState.lastError}`, "error");
  }
}

async function handleIdentityChange() {
  pipelineBuildState.compile.result = null;
  clearResolvePreview();
  pipelineBuildState.identity.species = String(dom.pipelineSpeciesInput?.value || "").trim();
  pipelineBuildState.identity.gender = String(dom.pipelineGenderSelect?.value || "male");
  applyStageProgression();
  await recomputeHashes();
  await refreshResolvePreview({ quiet: true });
  renderPipelinePanels({ syncAxisJsonEditor: false });
}

async function handleAxisJsonInput() {
  if (!dom.pipelineAxisJson) return;

  const raw = dom.pipelineAxisJson.value;
  if (!raw.trim()) {
    pipelineBuildState.compile.result = null;
    clearResolvePreview();
    pipelineBuildState.axis.payload = null;
    applyStageProgression();
    await recomputeHashes();
    await refreshResolvePreview({ quiet: true });
    renderBuildSummary();
    renderStageStatuses();
    renderStageEditorHint();
    renderCompilePanels();
    return;
  }

  try {
    const parsed = JSON.parse(raw);
    const schemaErrors = validateAxisPayloadSchema(parsed);
    if (schemaErrors.length > 0) {
      pipelineBuildState.compile.result = null;
      clearResolvePreview();
      pipelineBuildState.axis.payload = null;
      applyStageProgression();
      setStageStatus("axis_input", PIPELINE_STAGE_STATUS.ERROR);
      lockAfterAxis();
      await recomputeHashes();
      await refreshResolvePreview({ quiet: true });
      pipelineBuildState.lastError = `Axis JSON schema error: ${schemaErrors[0]}`;
      renderBuildSummary();
      renderStageStatuses();
      renderStageEditorHint();
      renderCompilePanels();
      setStatus(`Pipeline Build — ${pipelineBuildState.lastError}`);
      return;
    }
    pipelineBuildState.lastError = null;
    await updateAxisStateFromPayload(parsed, { syncAxisJsonEditor: false });
  } catch {
    pipelineBuildState.compile.result = null;
    clearResolvePreview();
    pipelineBuildState.axis.payload = null;
    applyStageProgression();
    setStageStatus("axis_input", PIPELINE_STAGE_STATUS.ERROR);
    lockAfterAxis();
    await recomputeHashes();
    await refreshResolvePreview({ quiet: true });
    pipelineBuildState.lastError = "Axis JSON parse error.";
    renderBuildSummary();
    renderStageStatuses();
    renderStageEditorHint();
    renderCompilePanels();
    setStatus("Pipeline Build — Axis JSON parse error.");
  }
}

async function handleAxisRelabel() {
  const payload = pipelineBuildState.axis.payload;
  if (!isAxisPayloadValid(payload)) {
    setStatus("Pipeline Build — load or enter a valid axis payload first.");
    return;
  }

  try {
    const relabeled = await relabelAxisPayload(payload);
    await updateAxisStateFromPayload(relabeled, { syncAxisJsonEditor: true });
    setStatus("Pipeline Build — axis labels recomputed.");
    appendActionLog("Axis labels recomputed via /api/relabel.");
  } catch (err) {
    const detail =
      err instanceof PipelineApiError
        ? err.detail || err.message
        : err?.message || String(err);
    pipelineBuildState.lastError = detail;
    renderPipelinePanels({ syncAxisJsonEditor: false });
    setStatus(`Pipeline Build — axis relabel failed: ${detail}`);
    appendActionLog(`Axis relabel failed: ${detail}`, "error");
  }
}

async function handleRuntimeInputChange() {
  pipelineBuildState.compile.result = null;
  clearResolvePreview();
  pipelineBuildState.runtime.worldContext = parseCsvList(dom.pipelineWorldContextInput?.value);
  pipelineBuildState.runtime.occupationSignals = parseCsvList(
    dom.pipelineOccupationSignalsInput?.value
  );
  pipelineBuildState.runtime.modelId = String(dom.pipelineModelIdInput?.value || "").trim() || null;
  pipelineBuildState.runtime.aspectRatio =
    String(dom.pipelineAspectRatioInput?.value || "").trim() || null;

  const seedRaw = String(dom.pipelineSeedInput?.value || "").trim();
  pipelineBuildState.runtime.seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
  if (!Number.isInteger(pipelineBuildState.runtime.seed)) {
    pipelineBuildState.runtime.seed = null;
  }

  applyStageProgression();
  await recomputeHashes();
  await refreshResolvePreview({ quiet: true });
  renderPipelinePanels({ syncAxisJsonEditor: false });
}

async function copyCompileResponseJson() {
  const result = pipelineBuildState.compile.result;
  if (!result) {
    setStatus("Pipeline Build — compile a prompt before copying response JSON.");
    return;
  }

  const text = JSON.stringify(result, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Pipeline Build — copied response JSON to clipboard.");
    appendActionLog("Copied compile response JSON.");
  } catch {
    setStatus("Pipeline Build — clipboard copy failed.");
    appendActionLog("Compile response JSON copy failed.", "error");
  }
}

function exportCompileResponseJson() {
  const result = pipelineBuildState.compile.result;
  if (!result) {
    setStatus("Pipeline Build — compile a prompt before exporting response JSON.");
    return;
  }

  const worldId = pipelineBuildState.selectedWorldId || "world";
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `pipeline-build-response-${worldId}-${timestamp}.json`;
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);

  setStatus(`Pipeline Build — exported response JSON as ${filename}.`);
  appendActionLog(`Exported compile response JSON (${filename}).`);
}

async function handleCompileRequest() {
  const requestBody = buildCompileRequest();
  if (!requestBody) {
    if (!worldIdsMatch()) {
      setStatus("Pipeline Build — axis payload world_id must match selected world.");
      return;
    }
    setStatus("Pipeline Build — complete Stage 1-4 inputs before compile.");
    return;
  }

  pipelineBuildState.busy = true;
  appendActionLog("Starting canonical compile request.");
  renderPipelinePanels({ syncAxisJsonEditor: false });

  try {
    const result = await compileImagePrompt(requestBody);
    pipelineBuildState.compile.result = result;
    pipelineBuildState.lastError = null;

    if (result.policy_hash) {
      pipelineBuildState.policyHash = String(result.policy_hash);
    }
    if (result.axis_hash) {
      pipelineBuildState.axisHash = String(result.axis_hash);
    }

    applyStageProgression();
    renderPipelinePanels({ syncAxisJsonEditor: false });
    setStatus("Pipeline Build — canonical compile complete.");
    appendActionLog("Canonical compile completed.");
  } catch (err) {
    const detail =
      err instanceof PipelineApiError
        ? err.detail || err.message
        : err?.message || String(err);

    if (err instanceof PipelineApiError && err.status === 401) {
      applyUnauthenticatedState(detail);
      setStatus("Pipeline Build — mud session expired. Please reconnect.");
      appendActionLog("Canonical compile failed: mud session expired.", "warn");
      return;
    }

    pipelineBuildState.lastError = detail;
    applyStageErrorFromPipelineApi(err);
    setStageStatus("compile_output", PIPELINE_STAGE_STATUS.ERROR);
    renderPipelinePanels({ syncAxisJsonEditor: false });
    setStatus(`Pipeline Build — compile failed: ${detail}`);
    appendActionLog(`Canonical compile failed: ${detail}`, "error");
  } finally {
    pipelineBuildState.busy = false;
    renderPipelinePanels({ syncAxisJsonEditor: false });
  }
}

/**
 * Initialize Pipeline Build page state and hydrate Session + World + Policy.
 *
 * @returns {Promise<void>}
 */
export async function initPipelineBuild() {
  resetPipelineBuildState();
  appendActionLog("Pipeline Build initialized.");
  renderPipelinePanels({ syncAxisJsonEditor: true });
  await refreshAxisPresetList();
  await refreshSessionAndWorlds({ quiet: true });
}

function wireStageListInteractions() {
  if (!dom.pipelineStageList) return;

  dom.pipelineStageList.addEventListener("click", (event) => {
    const row = event.target?.closest?.("li[data-stage]");
    if (!row) return;
    const stageKey = row.dataset.stage;
    if (!stageKey) return;
    focusStageControl(stageKey);
  });

  dom.pipelineStageList.addEventListener("keydown", (event) => {
    const row = event.target?.closest?.("li[data-stage]");
    if (!row) return;

    const rows = Array.from(dom.pipelineStageList.querySelectorAll("li[data-stage]"));
    const idx = rows.indexOf(row);
    if (idx < 0) return;

    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const delta = event.key === "ArrowDown" ? 1 : -1;
      const next = rows[idx + delta] || row;
      next.focus();
      return;
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const stageKey = row.dataset.stage;
      if (!stageKey) return;
      if (focusStageControl(stageKey)) {
        setStatus(`Pipeline Build — focused ${STAGE_LABEL[stageKey] || stageKey} controls.`);
      }
    }
  });
}

/**
 * Wire Pipeline Build page events.
 *
 * @returns {void}
 */
export function wirePipelineBuildEvents() {
  wireStageListInteractions();

  dom.pipelineWorldRefresh?.addEventListener("click", () => {
    refreshSessionAndWorlds();
  });

  dom.pipelineWorldSelect?.addEventListener("change", () => {
    applyWorldSelection(dom.pipelineWorldSelect.value);
  });

  dom.pipelinePolicyRefresh?.addEventListener("click", () => {
    if (!pipelineBuildState.selectedWorldId) {
      setStatus("Pipeline Build — select a world before reloading policy bundle.");
      return;
    }
    loadPolicyBundleForWorld(pipelineBuildState.selectedWorldId);
  });

  dom.pipelineSpeciesInput?.addEventListener("input", () => {
    handleIdentityChange();
  });

  dom.pipelineGenderSelect?.addEventListener("change", () => {
    handleIdentityChange();
  });

  dom.pipelineAxisSourceMode?.addEventListener("change", () => {
    pipelineBuildState.axis.sourceMode = dom.pipelineAxisSourceMode.value;
    renderAxisControls();
  });

  dom.pipelineAxisPresetSelect?.addEventListener("change", () => {
    const selected = dom.pipelineAxisPresetSelect.value;
    pipelineBuildState.axis.selectedPresetName = selected || null;
    pipelineBuildState.axis.selectedPresetMeta =
      pipelineBuildState.axis.presets.find((row) => row.name === selected) || null;
    renderAxisControls();
  });

  dom.pipelineAxisLoadPreset?.addEventListener("click", () => {
    loadAxisPresetByName(dom.pipelineAxisPresetSelect?.value || "");
  });

  dom.pipelineAxisJson?.addEventListener("input", () => {
    handleAxisJsonInput();
  });

  dom.pipelineAxisRelabel?.addEventListener("click", () => {
    handleAxisRelabel();
  });

  dom.pipelineWorldContextInput?.addEventListener("input", () => {
    handleRuntimeInputChange();
  });
  dom.pipelineOccupationSignalsInput?.addEventListener("input", () => {
    handleRuntimeInputChange();
  });
  dom.pipelineModelIdInput?.addEventListener("input", () => {
    handleRuntimeInputChange();
  });
  dom.pipelineAspectRatioInput?.addEventListener("input", () => {
    handleRuntimeInputChange();
  });
  dom.pipelineSeedInput?.addEventListener("input", () => {
    handleRuntimeInputChange();
  });

  dom.pipelineCompileButton?.addEventListener("click", () => {
    handleCompileRequest();
  });
  dom.pipelineCopyResponseJson?.addEventListener("click", () => {
    copyCompileResponseJson();
  });
  dom.pipelineExportResponseJson?.addEventListener("click", () => {
    exportCompileResponseJson();
  });

  document.addEventListener("pipeline-build-activated", () => {
    setStatus("Pipeline Build — active.");
    appendActionLog("Pipeline Build tab activated.");
    refreshSessionAndWorlds({ quiet: true });
  });
}

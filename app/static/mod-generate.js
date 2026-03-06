/**
 * mod-generate.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Character Description generation and metadata rendering.
 *
 * This module now supports two explicit generation modes:
 *
 * 1. exploratory (existing): POST /api/generate (local Ollama)
 * 2. canonical: POST /api/mud/compile-image-prompt (mud server authority)
 *
 * Canonical mode is deliberately opt-in and keeps the mud server as source
 * of truth for policy selection, block composition, and provenance hashes.
 *
 * Imports: mod-state, mod-utils, mod-status, mod-sync, mod-diff
 */

import { state, dom } from "./mod-state.js";
import { makePlaceholder } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";
import { getModelName, getOllamaHost, resolveSeed, syncJsonTextarea } from "./mod-sync.js";
import { updateDiff } from "./mod-diff.js";

function truncateHash(value) {
  if (!value) return null;
  return `${String(value).slice(0, 16)}\u2026`;
}

function renderError(message) {
  const errSpan = document.createElement("span");
  errSpan.style.color = "var(--col-err)";
  errSpan.textContent = `Error: ${message}`;
  dom.outputBox.textContent = "";
  dom.outputBox.appendChild(errSpan);
  setStatus(`Error: ${message}`);
}

function renderMetaTable(metaRows, { seed, wasRandom }) {
  const table = document.createElement("table");
  table.className = "meta-table";

  state.lastMeta = {};
  for (const [key, val] of metaRows) {
    state.lastMeta[key] = String(val);
    const tr = document.createElement("tr");

    const tdKey = document.createElement("td");
    tdKey.className = "meta-key";
    tdKey.textContent = key;
    tr.appendChild(tdKey);

    const tdVal = document.createElement("td");
    tdVal.className = "meta-val";
    tdVal.textContent = val;

    if (key === "seed" && wasRandom) {
      const copyBtn = document.createElement("button");
      copyBtn.className = "meta-copy-btn";
      copyBtn.type = "button";
      copyBtn.textContent = "copy";
      copyBtn.title = "Copy seed to clipboard";
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(String(seed)).then(() => {
          copyBtn.textContent = "copied";
          setTimeout(() => {
            copyBtn.textContent = "copy";
          }, 1200);
        });
      });
      tdVal.appendChild(document.createTextNode(" "));
      tdVal.appendChild(copyBtn);
    }

    tr.appendChild(tdVal);

    if (state.baselineMeta && state.baselineMeta[key] !== undefined) {
      if (String(val) !== state.baselineMeta[key]) {
        tr.classList.add("meta-changed");
      }
    }
    table.appendChild(tr);
  }

  dom.outputMeta.textContent = "";
  dom.outputMeta.appendChild(table);
  dom.outputMeta.classList.remove("hidden");
}

function applyGeneratedText(text) {
  dom.outputBox.textContent = text;
  state.current = text;
  dom.diffB.textContent = text;
  updateDiff();
}

function normaliseCanonicalAxes(rawPayload) {
  const axes = rawPayload?.axes || {};
  const result = {};
  for (const [axisName, axisValue] of Object.entries(axes)) {
    if (!axisValue || typeof axisValue !== "object") continue;
    result[axisName] = {
      label: String(axisValue.label ?? "").trim(),
      score: Number(axisValue.score),
    };
  }
  return result;
}

async function generateExploratory({
  model,
  temperature,
  maxTokens,
  systemPromptVal,
  ollamaHost,
  seed,
  wasRandom,
}) {
  const reqBody = {
    payload: state.payload,
    model,
    temperature,
    max_tokens: maxTokens,
    system_prompt: systemPromptVal || null,
    ollama_host: ollamaHost,
  };

  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(reqBody),
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `HTTP ${res.status}`);
  }

  const data = await res.json();
  applyGeneratedText(data.text);

  const seedVal = wasRandom ? `${seed} (random)` : `${seed}`;
  const metaRows = [
    ["mode", "exploratory"],
    ["model", data.model],
    ["temp", data.temperature],
    ["seed", seedVal],
  ];

  if (data.usage) {
    const promptTokens = data.usage.prompt_eval_count;
    const genTokens = data.usage.eval_count;
    if (promptTokens !== null && promptTokens !== undefined) {
      metaRows.push(["prompt tokens", promptTokens]);
    }
    if (genTokens !== null && genTokens !== undefined) {
      metaRows.push(["gen tokens", genTokens]);
    }
  }

  if (data.input_hash) metaRows.push(["input", truncateHash(data.input_hash)]);
  if (data.system_prompt_hash) metaRows.push(["prompt", truncateHash(data.system_prompt_hash)]);
  if (data.output_hash) metaRows.push(["output", truncateHash(data.output_hash)]);
  if (data.ipc_id) metaRows.push(["ipc", truncateHash(data.ipc_id)]);

  renderMetaTable(metaRows, { seed, wasRandom });
  setStatus(`Done (${data.model}).`);
}

async function generateCanonical({ model, seed, wasRandom }) {
  const sessionRes = await fetch("/api/mud/session");
  if (!sessionRes.ok) {
    throw new Error("Cannot read mud session status.");
  }
  const session = await sessionRes.json();
  if (!session.authenticated) {
    throw new Error("Canonical mode requires an authenticated mud-server session.");
  }

  const worldId = session.selected_world_id || state.payload.world_id;
  if (!worldId) {
    throw new Error("Canonical mode requires a selected world.");
  }

  const species = (dom.canonicalSpeciesInput?.value || "goblin").trim() || "goblin";
  const gender = dom.canonicalGenderSelect?.value || "male";
  const axes = normaliseCanonicalAxes(state.payload);

  const compileRes = await fetch("/api/mud/compile-image-prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      world_id: worldId,
      species,
      gender,
      axes,
      model_id: model || null,
      aspect_ratio: "1:1",
      seed,
    }),
  });
  if (!compileRes.ok) {
    const errData = await compileRes.json().catch(() => ({ detail: compileRes.statusText }));
    throw new Error(errData.detail || `HTTP ${compileRes.status}`);
  }

  const data = await compileRes.json();
  const compiledPrompt = String(data.compiled_prompt || "").trim();
  if (!compiledPrompt) {
    throw new Error("Canonical compile returned an empty prompt.");
  }

  applyGeneratedText(compiledPrompt);

  const seedVal = wasRandom ? `${seed} (random)` : `${seed}`;
  const defaults = data.generation_defaults || {};
  const metaRows = [
    ["mode", "canonical"],
    ["world", data.world_id || worldId],
    ["species", species],
    ["gender", gender],
    ["model", defaults.model_id || model || "--"],
    ["seed", seedVal],
  ];
  if (data.policy_bundle_id) metaRows.push(["bundle", data.policy_bundle_id]);
  if (data.policy_bundle_version !== undefined && data.policy_bundle_version !== null) {
    metaRows.push(["bundle ver", data.policy_bundle_version]);
  }
  if (data.policy_hash) metaRows.push(["policy", truncateHash(data.policy_hash)]);
  if (data.axis_hash) metaRows.push(["axis", truncateHash(data.axis_hash)]);
  if (data.selected_species_block_id) metaRows.push(["species block", data.selected_species_block_id]);
  if (data.selected_clothing_profile_id) {
    metaRows.push(["clothing profile", data.selected_clothing_profile_id]);
  }
  if (data.selected_descriptor_layer_id) {
    metaRows.push(["descriptor", data.selected_descriptor_layer_id]);
  }
  if (data.selected_tone_profile_id) {
    metaRows.push(["tone", data.selected_tone_profile_id]);
  }

  renderMetaTable(metaRows, { seed, wasRandom });
  setStatus("Done (canonical compile).");
}

function updateGenerationModeVisibility() {
  const mode = dom.generateModeSelect?.value || "exploratory";
  const showCanonical = mode === "canonical";
  for (const node of document.querySelectorAll(".canonical-only")) {
    node.classList.toggle("hidden", !showCanonical);
  }
}

/**
 * Run one generation cycle according to the selected generation mode.
 *
 * Exploratory mode uses local Ollama (`/api/generate`). Canonical mode uses
 * mud-server policy compilation (`/api/mud/compile-image-prompt`).
 */
export async function generate() {
  if (state.busy) return;

  if (!state.payload) {
    setStatus("No payload loaded – paste JSON or load an example.");
    return;
  }

  const model = getModelName();
  const temperature = parseFloat(dom.tempInput.value);
  const maxTokens = parseInt(dom.tokensInput.value, 10);
  const rawSeed = parseInt(dom.seedInput.value, 10);
  const wasRandom = Number.isNaN(rawSeed) || rawSeed < 0;
  const seed = resolveSeed();
  const mode = dom.generateModeSelect?.value || "exploratory";

  if (!model) {
    setStatus("No model specified.");
    return;
  }

  state.payload.seed = seed;
  syncJsonTextarea();

  const systemPromptVal = dom.systemPromptTextarea.value.trim();
  const ollamaHost = getOllamaHost() || null;

  state.busy = true;
  dom.btnGenerate.disabled = true;
  dom.outputBox.textContent = "";
  dom.outputMeta.textContent = "";
  dom.outputMeta.classList.add("hidden");

  try {
    if (mode === "canonical") {
      setStatus("Compiling canonical prompt via mud server…", true);
      await generateCanonical({ model, seed, wasRandom });
    } else {
      setStatus(`Generating via ${model}…`, true);
      await generateExploratory({
        model,
        temperature,
        maxTokens,
        systemPromptVal,
        ollamaHost,
        seed,
        wasRandom,
      });
    }
  } catch (err) {
    renderError(err.message);
  } finally {
    state.busy = false;
    dom.btnGenerate.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

/**
 * Wire generate-related event listeners.
 *
 * This includes:
 * - Generate button click
 * - Baseline snapshot button
 * - Canonical/exploratory mode visibility toggles
 */
export function wireGenerateEvents() {
  dom.btnGenerate.addEventListener("click", () => {
    generate();
  });

  dom.btnSetBaseline.addEventListener("click", () => {
    if (!state.current) {
      setStatus("Generate something first.");
      return;
    }
    state.baseline = state.current;
    state.baselineMeta = state.lastMeta ? { ...state.lastMeta } : null;
    dom.diffA.textContent = state.baseline;
    dom.btnSetBaseline.classList.add("is-active");
    setStatus("Baseline A set.");
  });

  if (dom.generateModeSelect) {
    dom.generateModeSelect.addEventListener("change", updateGenerationModeVisibility);
    updateGenerationModeVisibility();
  }
}

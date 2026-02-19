/**
 * mod-generate.js
 * ─────────────────────────────────────────────────────────────────────────────
 * LLM generate call, output rendering, and metadata table.
 *
 * This module owns the core generation cycle:
 *
 *   1. Collect payload + settings from state/DOM
 *   2. POST /api/generate → Ollama via backend
 *   3. Render generated text in the output box
 *   4. Build a metadata table (model, temp, seed, tokens, IPC hashes)
 *   5. Update diff B and trigger downstream diff analysis
 *
 * It also manages the "Set as A" baseline snapshot, which promotes the
 * current output to become the diff baseline for subsequent generations.
 *
 * Data flow
 * ─────────
 *   btnGenerate click → generate()
 *     → POST /api/generate → data.text → outputBox + state.current
 *     → meta table → outputMeta
 *     → diffB → updateDiff() (in mod-diff)
 *
 *   btnSetBaseline click
 *     → state.baseline = state.current
 *     → state.baselineMeta = state.lastMeta (snapshot)
 *
 * Imports: mod-state, mod-utils, mod-status, mod-sync, mod-diff
 */

import { state, dom } from "./mod-state.js";
import { makePlaceholder } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";
import { getModelName, getOllamaHost, resolveSeed, syncJsonTextarea } from "./mod-sync.js";
import { updateDiff } from "./mod-diff.js";

/**
 * POST /api/generate with current state, then update the output box,
 * metadata table, and diff B.
 *
 * Disables the generate button while in-flight to prevent double-submissions.
 * On success, the response is rendered in the output panel and the diff
 * is recomputed.  On failure, an inline error message is shown.
 *
 * The seed is resolved before the request: negative values trigger a
 * random 32-bit seed which is written back into the payload for
 * reproducibility tracking.
 */
export async function generate() {
  if (state.busy) return;

  if (!state.payload) {
    setStatus("No payload loaded – paste JSON or load an example.");
    return;
  }

  // ── Gather generation settings from DOM ────────────────────────── //
  const model       = getModelName();
  const temperature = parseFloat(dom.tempInput.value);
  const max_tokens  = parseInt(dom.tokensInput.value, 10);
  const rawSeed     = parseInt(dom.seedInput.value, 10);
  const wasRandom   = isNaN(rawSeed) || rawSeed < 0;
  const seed        = resolveSeed();

  if (!model) {
    setStatus("No model specified.");
    return;
  }

  // Write the resolved seed into the payload so the request carries
  // the actual seed used (important for logging and reproducibility)
  state.payload.seed = seed;
  syncJsonTextarea();

  const systemPromptVal = dom.systemPromptTextarea.value.trim();
  const ollamaHost = getOllamaHost() || null;

  const reqBody = {
    payload:       state.payload,
    model,
    temperature,
    max_tokens,
    system_prompt: systemPromptVal || null,
    ollama_host:   ollamaHost,
  };

  // ── Pre-request UI state ───────────────────────────────────────── //
  state.busy = true;
  dom.btnGenerate.disabled = true;
  setStatus(`Generating via ${model}…`, true);

  // Clear previous output
  dom.outputBox.textContent   = "";
  dom.outputMeta.textContent  = "";
  dom.outputMeta.classList.add("hidden");

  try {
    const res = await fetch("/api/generate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(reqBody),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // ── Update output box ────────────────────────────────────────── //
    dom.outputBox.textContent = data.text;
    state.current = data.text;

    // ── Build meta info table ────────────────────────────────────── //
    // A two-column key-value table: labels on the left, values on the
    // right.  Rows are conditional on what the backend returns.
    const seedVal = wasRandom ? `${seed} (random)` : `${seed}`;

    const metaRows = [
      ["model",         data.model],
      ["temp",          data.temperature],
      ["seed",          seedVal],
    ];

    // Token usage rows (only if reported by Ollama)
    if (data.usage) {
      const p = data.usage.prompt_eval_count;
      const e = data.usage.eval_count;
      if (p !== null && p !== undefined) metaRows.push(["prompt tokens", p]);
      if (e !== null && e !== undefined) metaRows.push(["gen tokens", e]);
    }

    // IPC provenance hashes (truncated to 16 chars for display)
    if (data.input_hash)         metaRows.push(["input",  data.input_hash.slice(0, 16) + "\u2026"]);
    if (data.system_prompt_hash) metaRows.push(["prompt", data.system_prompt_hash.slice(0, 16) + "\u2026"]);
    if (data.output_hash)        metaRows.push(["output", data.output_hash.slice(0, 16) + "\u2026"]);
    if (data.ipc_id)             metaRows.push(["ipc",    data.ipc_id.slice(0, 16) + "\u2026"]);

    // Store current metadata for baseline snapshot comparisons
    state.lastMeta = {};
    for (const [key, val] of metaRows) {
      state.lastMeta[key] = String(val);
    }

    // ── Render the meta table ────────────────────────────────────── //
    const table = document.createElement("table");
    table.className = "meta-table";

    for (const [key, val] of metaRows) {
      const tr = document.createElement("tr");

      const tdKey = document.createElement("td");
      tdKey.className = "meta-key";
      tdKey.textContent = key;
      tr.appendChild(tdKey);

      const tdVal = document.createElement("td");
      tdVal.className = "meta-val";
      tdVal.textContent = val;

      // When the seed was randomly generated, append a copy button so
      // the user can grab the numeric value for deterministic replay
      if (key === "seed" && wasRandom) {
        const copyBtn = document.createElement("button");
        copyBtn.className = "meta-copy-btn";
        copyBtn.type = "button";
        copyBtn.textContent = "copy";
        copyBtn.title = "Copy seed to clipboard";
        copyBtn.addEventListener("click", () => {
          navigator.clipboard.writeText(String(seed)).then(() => {
            copyBtn.textContent = "copied";
            setTimeout(() => { copyBtn.textContent = "copy"; }, 1200);
          });
        });
        tdVal.appendChild(document.createTextNode(" "));
        tdVal.appendChild(copyBtn);
      }

      tr.appendChild(tdVal);

      // Highlight rows that differ from the baseline meta snapshot
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

    // ── Update diff B and trigger downstream analysis ────────────── //
    dom.diffB.textContent = data.text;
    updateDiff();

    setStatus(`Done (${data.model}).`);
  } catch (err) {
    // Show error inline in the output box
    const errSpan = document.createElement("span");
    errSpan.style.color = "var(--col-err)";
    errSpan.textContent = `Error: ${err.message}`;
    dom.outputBox.textContent = "";
    dom.outputBox.appendChild(errSpan);
    setStatus(`Error: ${err.message}`);
  } finally {
    // Always restore interactive state, even on error
    state.busy           = false;
    dom.btnGenerate.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

/**
 * Wire generate-related event listeners.
 *
 * Registers handlers for:
 *   - Generate button click → call generate()
 *   - Set as A button click → snapshot current output as baseline
 *
 * The "Set as A" handler also snapshots the current meta table into
 * `state.baselineMeta` so subsequent generations can highlight which
 * IPC hashes and settings have changed.
 *
 * Called once during startup by the mod-events coordinator.
 */
export function wireGenerateEvents() {
  // ── Generate ───────────────────────────────────────────────────── //
  dom.btnGenerate.addEventListener("click", () => {
    generate();
  });

  // ── Set baseline (A) ──────────────────────────────────────────── //
  // Promotes the current output (B) to become baseline (A) for
  // subsequent diff comparisons.
  dom.btnSetBaseline.addEventListener("click", () => {
    if (!state.current) {
      setStatus("Generate something first.");
      return;
    }
    state.baseline = state.current;
    // Shallow copy of meta so mutations to lastMeta don't affect baseline
    state.baselineMeta = state.lastMeta ? { ...state.lastMeta } : null;
    dom.diffA.textContent = state.baseline;
    dom.btnSetBaseline.classList.add("is-active");
    setStatus("Baseline A set.");
  });
}

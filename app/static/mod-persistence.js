/**
 * mod-persistence.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Session persistence: save, export, import, restore, and logging.
 *
 * This module manages the full lifecycle of session data:
 *
 *   1. **Save** — Persist the current session state (payload, output,
 *      baseline, settings, diff metadata) to a timestamped folder on
 *      the server via `POST /api/save`.
 *
 *   2. **Export** — Download the last saved session as a `.zip` archive
 *      via `GET /api/save/{folder}/export`.
 *
 *   3. **Import** — Upload a `.zip` save package via `POST /api/import`
 *      and fully restore the session state from the imported data.
 *
 *   4. **Restore** — Rebuild all frontend state (payload, sliders, model,
 *      temperature, system prompt, output, baseline, diff) from an
 *      `ImportResponse` object.  Used by `importSave()` but also
 *      available for other restore scenarios.
 *
 *   5. **Log** — Append a fire-and-forget entry to the server's JSONL
 *      run log via `POST /api/log`.
 *
 * Data flow
 * ─────────
 *   btnSave click → saveRun()
 *     → POST /api/save → state.lastSaveFolderName (enables export)
 *
 *   btnExport click → exportSave()
 *     → GET /api/save/{folder}/export → browser download (.zip)
 *
 *   btnImport click → importFileInput.click() → importSave()
 *     → POST /api/import (FormData) → restoreSessionState(data)
 *
 *   btnClearOutput click
 *     → reset state (current, baseline, diff, meta) + clear all panels
 *
 * Imports: mod-state, mod-utils, mod-status, mod-sync, mod-diff
 */

import { state, dom } from "./mod-state.js";
import { extractTransformationRows, makePlaceholder } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";
import {
  getModelName, syncJsonTextarea, buildSlidersFromJson,
  setJsonBadge, updateSystemPromptBadge,
} from "./mod-sync.js";
import { updateDiff } from "./mod-diff.js";

/**
 * Save the current session state to a timestamped folder on the server.
 *
 * Collects all relevant state:
 *   - The axis payload (JSON)
 *   - The current output text
 *   - The baseline text (if set)
 *   - Generation settings (model, temperature, max_tokens)
 *   - The system prompt (fetches server default if textarea is empty)
 *   - Transformation map rows (if a diff exists)
 *   - Diff change percentage (if a diff exists)
 *
 * The server creates a timestamped directory under `data/` containing
 * individual files (payload.json, output.md, baseline.md, etc.) and
 * returns the folder name.  The folder name is stored in
 * `state.lastSaveFolderName` to enable subsequent export.
 *
 * The save button is disabled during the request to prevent double-saves.
 *
 * @returns {Promise<void>} Resolves after save completes or error is shown.
 */
export async function saveRun() {
  if (!state.payload) {
    setStatus("Nothing to save \u2013 load a payload first.");
    return;
  }

  // Resolve the system prompt: use the textarea value if the user has
  // overridden it, otherwise fetch the server default for archival
  let systemPromptVal = dom.systemPromptTextarea.value.trim();
  if (!systemPromptVal) {
    try {
      const defaultRes = await fetch("/api/system-prompt");
      if (defaultRes.ok) {
        systemPromptVal = await defaultRes.text();
      } else {
        systemPromptVal = "(server default \u2013 see system_prompt_v01.txt)";
      }
    } catch {
      systemPromptVal = "(server default \u2013 see system_prompt_v01.txt)";
    }
  }

  // ── Collect generation settings from DOM ──────────────────────────── //
  const model       = getModelName();
  const temperature = parseFloat(dom.tempInput.value);
  const max_tokens  = parseInt(dom.tokensInput.value, 10);

  // ── Extract diff metadata (if available) ──────────────────────────── //
  // Prefer server-computed rows (with indicators) when available.
  let tmapRows = null;
  let diffChangePct = null;
  if (state.lastTmapResponse) {
    const filtered = state.tmapIncludeAll
      ? state.lastTmapResponse
      : state.lastTmapResponse.filter(r => r.removed && r.added);
    if (filtered.length > 0) tmapRows = filtered;
  } else if (state.lastDiff) {
    const rows = extractTransformationRows(state.lastDiff, state.tmapIncludeAll);
    if (rows.length > 0) tmapRows = rows;
  }
  if (state.lastDiff) {

    // Recompute change percentage from the cached diff
    const eqCount  = state.lastDiff.filter(([op]) => op === "=").length;
    const addCount = state.lastDiff.filter(([op]) => op === "+").length;
    const delCount = state.lastDiff.filter(([op]) => op === "-").length;
    const total    = eqCount + addCount + delCount;
    if (total > 0) {
      diffChangePct = Math.round(((addCount + delCount) / total) * 100);
    }
  }

  const reqBody = {
    payload:            state.payload,
    output:             state.current,
    baseline:           state.baseline,
    model,
    temperature,
    max_tokens,
    system_prompt:      systemPromptVal,
    transformation_map: tmapRows,
    diff_change_pct:    diffChangePct,
  };

  // ── Pre-request UI state ──────────────────────────────────────────── //
  dom.btnSave.disabled = true;
  setStatus("Saving\u2026", true);

  try {
    const res = await fetch("/api/save", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(reqBody),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Store folder name so export can reference it
    state.lastSaveFolderName = data.folder_name;
    if (dom.btnExport) dom.btnExport.disabled = false;

    setStatus(`Saved \u2192 data/${data.folder_name}/ (${data.files.join(", ")})`);
  } catch (err) {
    setStatus(`Save error: ${err.message}`);
  } finally {
    dom.btnSave.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

/**
 * Download the last saved session as a `.zip` archive.
 *
 * Fetches the binary blob from the server's export endpoint, creates a
 * temporary `<a>` link with a blob URL, and triggers a programmatic
 * click to initiate the browser download.  The temporary DOM node and
 * blob URL are cleaned up after a short delay.
 *
 * Requires a prior successful `saveRun()` call — the folder name is
 * stored in `state.lastSaveFolderName`.
 *
 * @returns {Promise<void>} Resolves after the download is triggered.
 */
export async function exportSave() {
  if (!state.lastSaveFolderName) {
    setStatus("Nothing to export \u2013 save first.");
    return;
  }

  setStatus("Exporting zip\u2026", true);

  try {
    const res = await fetch(
      `/api/save/${encodeURIComponent(state.lastSaveFolderName)}/export`
    );

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    // Create a temporary download link from the response blob
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href     = url;
    a.download = `${state.lastSaveFolderName}.zip`;
    document.body.appendChild(a);
    a.click();

    // Clean up: remove the link and revoke the blob URL
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);

    setStatus(`Exported \u2192 ${state.lastSaveFolderName}.zip`);
  } catch (err) {
    setStatus(`Export error: ${err.message}`);
  } finally {
    dom.spinner.classList.add("hidden");
  }
}

/**
 * Upload a `.zip` save package and restore the full session state.
 *
 * Reads the file from the hidden `<input type="file">`, sends it as
 * `multipart/form-data` to `POST /api/import`, and calls
 * `restoreSessionState()` with the parsed response.
 *
 * The file input is reset after import (success or failure) so the
 * same file can be re-imported if needed.
 *
 * @returns {Promise<void>} Resolves after restore completes or error
 *   is shown.
 */
export async function importSave() {
  const file = dom.importFileInput.files[0];
  if (!file) return;

  setStatus("Importing zip\u2026", true);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/import", {
      method: "POST",
      body:   formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Rebuild all frontend state from the imported data
    restoreSessionState(data);

    let msg = `Imported ${data.folder_name} (${data.files.length} files)`;
    if (data.warnings.length > 0) {
      msg += ` \u2014 warnings: ${data.warnings.join("; ")}`;
    }
    setStatus(msg);
  } catch (err) {
    setStatus(`Import error: ${err.message}`);
  } finally {
    // Reset file input so the same file can be re-imported
    dom.importFileInput.value = "";
    dom.spinner.classList.add("hidden");
  }
}

/**
 * Restore all frontend state from an `ImportResponse` object.
 *
 * This function is the inverse of `saveRun()`: it takes the server's
 * import response and rebuilds every piece of UI and state to match
 * the saved session.
 *
 * Restoration follows a strict 6-step sequence:
 *
 *   1. **Payload** → state → JSON textarea → slider panel
 *   2. **Model / temperature / max_tokens / seed** → form controls
 *      (model is set in either the `<select>` or manual `<input>`
 *      depending on whether it exists in the current dropdown)
 *   3. **System prompt** → textarea → override badge
 *   4. **Output and baseline** → state + display panels
 *   5. **Diff recomputation** → if both A and B exist, call `updateDiff()`
 *      which cascades to signal isolation and transformation map
 *   6. **Enable export** → store folder name, enable export button
 *
 * @param {object} data - The `ImportResponse` JSON from `POST /api/import`.
 *   Expected fields: `payload`, `model`, `temperature`, `max_tokens`,
 *   `system_prompt`, `output`, `baseline`, `metadata`, `folder_name`,
 *   `files`, `warnings`.
 */
export function restoreSessionState(data) {
  // ── 1. Payload → textarea → sliders ──────────────────────────────── //
  state.payload = data.payload;
  syncJsonTextarea();
  buildSlidersFromJson();
  setJsonBadge(true);

  // ── 2. Model / temperature / max_tokens / seed ────────────────────── //
  // If the saved model exists in the current Ollama dropdown, select it;
  // otherwise fall back to the manual text input
  const modelInSelect = Array.from(dom.modelSelect.options).some(
    (opt) => opt.value === data.model
  );
  if (modelInSelect) {
    dom.modelSelect.value = data.model;
    dom.modelSelect.classList.remove("hidden");
    dom.modelInput.classList.add("hidden");
  } else {
    dom.modelInput.value = data.model;
    dom.modelInput.classList.remove("hidden");
    dom.modelSelect.classList.add("hidden");
  }

  dom.tempInput.value = data.temperature;
  dom.tempRange.value = data.temperature;
  dom.tokensInput.value = data.max_tokens;

  if (data.metadata && data.metadata.seed !== undefined) {
    dom.seedInput.value = data.metadata.seed;
  }

  // ── 3. System prompt ──────────────────────────────────────────────── //
  dom.systemPromptTextarea.value = data.system_prompt || "";
  updateSystemPromptBadge();

  // ── 4. Output and baseline ────────────────────────────────────────── //
  state.current = data.output || null;
  if (state.current) {
    dom.outputBox.textContent = state.current;
  } else {
    dom.outputBox.textContent = "";
    dom.outputBox.appendChild(makePlaceholder("Click Generate to produce a description."));
  }

  state.baseline = data.baseline || null;
  if (state.baseline) {
    dom.diffA.textContent = state.baseline;
    dom.btnSetBaseline.classList.add("is-active");
  } else {
    dom.diffA.textContent = "";
    dom.diffA.appendChild(makePlaceholder("No baseline set."));
    dom.btnSetBaseline.classList.remove("is-active");
  }

  // ── 5. Diff recomputation ─────────────────────────────────────────── //
  // If both baseline and current exist, run the full diff pipeline
  // (word diff → signal isolation → transformation map).  Otherwise
  // reset the diff panels to their empty placeholder state.
  if (state.baseline && state.current) {
    updateDiff();
  } else {
    dom.diffB.textContent = state.current || "";
    if (!state.current) {
      dom.diffB.textContent = "";
      dom.diffB.appendChild(makePlaceholder("Generate to populate B."));
    }
    dom.diffDelta.textContent = "";
    dom.diffDelta.appendChild(makePlaceholder("Set a baseline and generate to compare."));
    dom.diffPct.style.display = "none";
    state.lastDiff = null;
    state.lastTmapResponse = null;
  }

  // ── 6. Enable export ──────────────────────────────────────────────── //
  state.lastSaveFolderName = data.folder_name;
  if (dom.btnExport) dom.btnExport.disabled = false;
}

/**
 * Append a fire-and-forget log entry to the server's JSONL run log.
 *
 * Sends the current payload plus generation metadata to `POST /api/log`
 * via query parameters (for scalar values) and a JSON body (for the
 * payload).  The server appends a timestamped entry to
 * `logs/run_log.jsonl` with a SHA-256 input hash for grouping.
 *
 * Errors are logged to the browser console but not surfaced to the user
 * — logging is informational and should never block the UI.
 *
 * @param {string} output      - The generated text to log.
 * @param {string} model       - The model name used for generation.
 * @param {number} temperature - The temperature setting used.
 * @param {number} max_tokens  - The token budget used.
 * @returns {Promise<void>} Resolves silently; errors are console-warned.
 */
export async function logRun(output, model, temperature, max_tokens) {
  if (!state.payload) return;

  const params = new URLSearchParams({
    output,
    model,
    temperature: String(temperature),
    max_tokens:  String(max_tokens),
  });

  try {
    await fetch(`/api/log?${params.toString()}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(state.payload),
    });
  } catch (err) {
    console.warn("[ADL] Log request failed:", err);
  }
}

/**
 * Wire persistence-related event listeners.
 *
 * Registers handlers for:
 *   - **Save button** click → `saveRun()`
 *   - **Export button** click → `exportSave()`
 *   - **Import button** click → triggers hidden file input
 *   - **Import file input** change → `importSave()`
 *   - **Clear output button** click → resets all output/diff/meta state
 *
 * The import flow uses a hidden `<input type="file">` that is
 * programmatically clicked when the visible Import button is pressed.
 * The `change` event on the file input triggers the actual import.
 *
 * The clear-output handler performs a comprehensive reset:
 *   - Clears output box, meta panel, and all diff panels
 *   - Resets `state.current`, `state.baseline`, `state.lastMeta`,
 *     `state.baselineMeta`, `state.lastDiff`, `state.lastSaveFolderName`
 *   - Disables the export button and removes baseline highlight
 *   - Restores all panels to their empty placeholder states
 *
 * Called once during startup by the `mod-events.js` coordinator.
 */
export function wirePersistenceEvents() {
  // ── Save session state ────────────────────────────────────────────── //
  dom.btnSave.addEventListener("click", () => {
    saveRun();
  });

  // ── Export Zip ────────────────────────────────────────────────────── //
  if (dom.btnExport) {
    dom.btnExport.addEventListener("click", () => {
      exportSave();
    });
  }

  // ── Import Zip ────────────────────────────────────────────────────── //
  // The visible Import button triggers the hidden file input;
  // the file input's change event triggers the actual import
  if (dom.btnImport) {
    dom.btnImport.addEventListener("click", () => {
      dom.importFileInput.click();
    });
  }

  if (dom.importFileInput) {
    dom.importFileInput.addEventListener("change", () => {
      importSave();
    });
  }

  // ── Clear output ──────────────────────────────────────────────────── //
  // Comprehensive reset: clears all output, diff, and meta state,
  // returning every panel to its initial placeholder appearance.
  dom.btnClearOutput.addEventListener("click", () => {
    // Clear output panel
    dom.outputBox.textContent = "";
    dom.outputBox.appendChild(makePlaceholder("Click Generate to produce a description."));
    dom.outputMeta.textContent = "";
    dom.outputMeta.classList.add("hidden");

    // Reset all transient state
    state.current        = null;
    state.baseline       = null;
    state.lastMeta       = null;
    state.baselineMeta   = null;
    state.lastDiff       = null;
    state.lastTmapResponse = null;
    state.lastSaveFolderName = null;

    // Disable export and hide diff percentage badge
    dom.diffPct.style.display = "none";
    if (dom.btnExport) dom.btnExport.disabled = true;

    // Reset diff panels to placeholder state
    dom.diffA.textContent = "";
    dom.diffA.appendChild(makePlaceholder("No baseline set."));
    dom.diffB.textContent = "";
    dom.diffB.appendChild(makePlaceholder("Generate to populate B."));
    dom.diffDelta.textContent = "";
    dom.diffDelta.appendChild(makePlaceholder("\u2014"));

    // Reset signal isolation panel
    dom.signalPanel.textContent = "";
    dom.signalPanel.appendChild(
      makePlaceholder("Set a baseline (A) and generate to analyze content-word changes.")
    );

    // Reset transformation map panel
    dom.tmapPanel.textContent = "";
    dom.tmapPanel.appendChild(
      makePlaceholder("Set a baseline (A) and generate to see clause-level substitutions.")
    );

    // Remove baseline highlight
    dom.btnSetBaseline.classList.remove("is-active");
    setStatus("Output cleared.");
  });
}

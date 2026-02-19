/**
 * mod-axis-actions.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Axis mutation actions: relabel, randomise, and auto-label toggle.
 *
 * This module provides the two primary ways a user can batch-modify axis
 * values in the current payload:
 *
 *   1. **Relabel** — POST the current payload to `/api/relabel`, which
 *      applies a server-side policy table mapping score ranges to
 *      canonical labels (e.g. age 0.25 → "young", 0.75 → "old").
 *      The authoritative policy lives on the server; the frontend
 *      never hardcodes label mappings.
 *
 *   2. **Randomise** — Assign every axis a fresh random score using
 *      `cryptoRandomFloat()` (Web Crypto API), then optionally auto-
 *      relabel if the toggle is ON.
 *
 * It also owns:
 *   - The **auto-label toggle** handler, which rebuilds sliders
 *     (enabling/disabling label inputs) and triggers relabel when
 *     checked.  This handler lives here (not in mod-sync.js) to avoid
 *     a circular dependency: mod-sync.js cannot import from this module
 *     because this module imports from mod-sync.js.
 *   - The **Ollama host input** debounced handler, which refreshes the
 *     model dropdown when the user changes the server URL.
 *
 * Data flow
 * ─────────
 *   btnRelabel click → relabel()
 *     → POST /api/relabel → updated payload → state + textarea + sliders
 *
 *   btnRandomise click → randomiseAxes()
 *     → cryptoRandomFloat per axis → state + textarea + sliders
 *     → (if auto-label ON) → relabel()
 *
 *   autoLabelToggle change
 *     → buildSlidersFromJson() (re-render with inputs enabled/disabled)
 *     → (if checked) → relabel()
 *
 *   ollamaHostInput input (debounced 600ms)
 *     → refreshModels()
 *
 * Imports: mod-state, mod-utils, mod-status, mod-sync
 */

import { state, dom } from "./mod-state.js";
import { cryptoRandomFloat, debounce } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";
import { syncJsonTextarea, buildSlidersFromJson, refreshModels } from "./mod-sync.js";

/**
 * Request the server to recompute all axis labels from the current scores
 * using the server-side policy table.
 *
 * Sends the full `state.payload` to `POST /api/relabel`.  The server
 * applies its policy (score-range → label mapping) and returns the
 * updated payload.  The frontend replaces `state.payload` wholesale
 * and rebuilds the slider panel to reflect the new labels.
 *
 * This is a key part of the "server is authoritative" design: label
 * logic never lives in the frontend.  The JS only sends scores and
 * receives the policy-derived labels.
 *
 * @returns {Promise<void>} Resolves when the relabel round-trip completes
 *   or an error is shown in the status bar.
 */
export async function relabel() {
  if (!state.payload) {
    setStatus("No payload to relabel.");
    return;
  }

  setStatus("Recomputing labels…", true);

  try {
    const res = await fetch("/api/relabel", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(state.payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const updated = await res.json();
    state.payload = updated;
    syncJsonTextarea();
    buildSlidersFromJson();
    setStatus("Labels recomputed from policy.");
  } catch (err) {
    setStatus(`Relabel error: ${err.message}`);
  }
}

/**
 * Assign every axis in the current payload a new random score and
 * optionally trigger relabelling.
 *
 * Each score is generated via `cryptoRandomFloat()` (backed by the
 * Web Crypto API) and quantised to 3 decimal places to match the
 * slider step resolution (0.005 rounds nicely at 3 d.p.).
 *
 * If the auto-label toggle is checked, `relabel()` is called after
 * randomisation so labels stay consistent with the new scores.
 *
 * The function is `async` solely because it may `await relabel()`.
 * The randomisation itself is synchronous.
 *
 * @returns {Promise<void>} Resolves after scores are set (and optionally
 *   relabelled).
 */
export async function randomiseAxes() {
  if (!state.payload || !state.payload.axes) {
    setStatus("No payload to randomise.");
    return;
  }

  // Generate a new random score for every axis
  for (const axisKey of Object.keys(state.payload.axes)) {
    const newScore = Math.round(cryptoRandomFloat() * 1000) / 1000;
    state.payload.axes[axisKey] = {
      ...state.payload.axes[axisKey],
      score: newScore,
    };
  }

  // Sync the JSON textarea and rebuild sliders with new scores
  syncJsonTextarea();
  buildSlidersFromJson();

  // If auto-label is active, request server-side relabelling
  if (dom.autoLabelToggle.checked) {
    await relabel();
  }

  setStatus("Axis scores randomised.");
}

/**
 * Wire axis-action event listeners.
 *
 * Registers handlers for:
 *   - **Relabel button** click → `relabel()`
 *   - **Randomise button** click → `randomiseAxes()`
 *   - **Auto-label toggle** change → rebuild sliders + conditional relabel
 *   - **Ollama host input** (debounced 600ms) → `refreshModels()`
 *
 * The auto-label toggle lives here rather than in `wireSyncEvents()`
 * to avoid a circular import: `mod-sync.js` cannot import `relabel()`
 * from this module because this module already imports from `mod-sync.js`.
 * Placing the handler here keeps the import graph acyclic.
 *
 * Called once during startup by the `mod-events.js` coordinator.
 */
export function wireAxisEvents() {
  // ── Relabel button ───────────────────────────────────────────────── //
  dom.btnRelabel.addEventListener("click", () => {
    relabel();
  });

  // ── Randomise button ─────────────────────────────────────────────── //
  dom.btnRandomise.addEventListener("click", () => {
    randomiseAxes();
  });

  // ── Auto-label toggle ────────────────────────────────────────────── //
  // When toggled, rebuild sliders so label inputs become enabled or
  // disabled.  When checked ON, also trigger relabel so labels are
  // immediately policy-consistent with current scores.
  dom.autoLabelToggle.addEventListener("change", () => {
    buildSlidersFromJson();
    if (dom.autoLabelToggle.checked) {
      relabel();
    }
  });

  // ── Ollama host URL (debounced 600ms) ────────────────────────────── //
  // Refresh the model dropdown whenever the user changes the Ollama
  // server URL.  Debounced to avoid hammering the Ollama instance
  // on every keystroke.
  dom.ollamaHostInput.addEventListener(
    "input",
    debounce(() => {
      refreshModels();
    }, 600)
  );
}

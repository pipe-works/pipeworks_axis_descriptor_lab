/**
 * mod-sync.js
 * ─────────────────────────────────────────────────────────────────────────────
 * JSON textarea / slider / badge synchronisation, form readers, and model
 * refresh.
 *
 * This module owns the bidirectional sync between three representations
 * of the axis payload:
 *
 *   state.payload  ←→  JSON textarea  ←→  slider panel
 *
 * Data flow
 * ─────────
 *   JSON textarea edit → safeParse → state.payload → buildSlidersFromJson
 *   Slider change      → state.payload → syncJsonTextarea
 *   Label input change → state.payload → syncJsonTextarea
 *
 * It also provides "form reader" functions (getModelName, getOllamaHost,
 * resolveSeed) that extract generation settings from DOM inputs, and the
 * refreshModels function that repopulates the model <select> from Ollama.
 *
 * Imports: mod-state, mod-utils, mod-status
 */

import { state, dom } from "./mod-state.js";
import { clamp, debounce, safeParse } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";

/**
 * Reflect `state.payload` back into the JSON textarea as pretty-printed
 * JSON (2-space indent).
 *
 * Called after every slider or label change so the textarea stays in sync
 * with the canonical `state.payload` object.
 */
export function syncJsonTextarea() {
  dom.jsonTextarea.value = JSON.stringify(state.payload, null, 2);
}

/**
 * Mark the JSON status badge as valid ("OK") or error ("ERR").
 *
 * The badge sits next to the "Axis JSON" label in the left panel and
 * provides immediate visual feedback on whether the textarea contains
 * well-formed JSON.
 *
 * @param {boolean} valid - True for OK (green), false for ERR (red).
 */
export function setJsonBadge(valid) {
  if (valid) {
    dom.jsonStatusBadge.textContent = "OK";
    dom.jsonStatusBadge.className   = "badge";
  } else {
    dom.jsonStatusBadge.textContent = "ERR";
    dom.jsonStatusBadge.className   = "badge badge--err";
  }
}

/**
 * Update the system prompt "override" badge to reflect whether a custom
 * prompt is active.
 *
 * When the textarea contains user text the badge switches to amber
 * (`badge--active`) to clearly signal that the server default is being
 * overridden.  When the textarea is empty (server default in effect),
 * the badge reverts to the muted grey style.
 */
export function updateSystemPromptBadge() {
  const hasContent = dom.systemPromptTextarea.value.trim().length > 0;
  if (hasContent) {
    dom.systemPromptBadge.className = "badge badge--active";
  } else {
    dom.systemPromptBadge.className = "badge badge--muted";
  }
}

/**
 * Completely rebuild the slider panel from the current `state.payload`.
 *
 * One axis row is created per key in `payload.axes`.  Each row contains:
 *   - axis name (non-interactive label)
 *   - range input for score (0–1, step 0.005)
 *   - numeric score display (live, 3 d.p.)
 *   - text input for label (editable when auto-label mode is OFF)
 *
 * Event listeners on the sliders and label inputs write changes back into
 * `state.payload` and call `syncJsonTextarea()` to keep the JSON editor
 * in sync.
 *
 * Scores and labels that differ from the originally loaded example are
 * highlighted with the `axis-modified` CSS class so the user can see at
 * a glance what they have changed.
 */
export function buildSlidersFromJson() {
  const payload = state.payload;

  // Guard: nothing to render — show placeholder
  if (!payload || typeof payload.axes !== "object" || payload.axes === null) {
    dom.sliderPanel.textContent = "";
    const noAxesP = document.createElement("p");
    noAxesP.className = "placeholder-text";
    noAxesP.textContent = "No axes found in payload.";
    dom.sliderPanel.appendChild(noAxesP);
    return;
  }

  const axes = payload.axes;
  const keys = Object.keys(axes);

  if (keys.length === 0) {
    dom.sliderPanel.textContent = "";
    const emptyP = document.createElement("p");
    emptyP.className = "placeholder-text";
    emptyP.textContent = "axes object is empty.";
    dom.sliderPanel.appendChild(emptyP);
    return;
  }

  // Build all rows in a DocumentFragment to minimise reflows
  const fragment = document.createDocumentFragment();

  for (const axisKey of keys) {
    const axisVal = axes[axisKey];

    // Normalise in case the value came from malformed JSON
    const score  = clamp(parseFloat(axisVal.score)  || 0, 0, 1);
    const label  = String(axisVal.label || "");

    const row = document.createElement("div");
    row.className    = "axis-row";
    row.dataset.axis = axisKey;

    // ── Axis name (read-only label) ────────────────────────────────── //
    const nameEl = document.createElement("span");
    nameEl.className   = "axis-name";
    nameEl.textContent = axisKey;
    nameEl.title       = axisKey;

    // ── Centre column: slider + numeric score display ──────────────── //
    const sliderWrap = document.createElement("div");
    sliderWrap.className = "axis-slider-row";

    const slider = document.createElement("input");
    slider.type      = "range";
    slider.className = "range-input";
    slider.min       = "0";
    slider.max       = "1";
    slider.step      = "0.005";
    slider.value     = score.toFixed(3);
    slider.setAttribute("aria-label", `${axisKey} score`);

    const scoreDisplay = document.createElement("span");
    scoreDisplay.className   = "axis-score";
    scoreDisplay.textContent = score.toFixed(3);

    // Highlight the score if it differs from the originally loaded example
    const orig = state.originalAxes && state.originalAxes[axisKey];
    if (orig && Math.abs(score - orig.score) > 0.0001) {
      scoreDisplay.classList.add("axis-modified");
    }

    sliderWrap.appendChild(slider);
    sliderWrap.appendChild(scoreDisplay);

    // ── Label input (editable text field) ──────────────────────────── //
    const labelInput = document.createElement("input");
    labelInput.type      = "text";
    labelInput.className = "axis-label-input";
    labelInput.value     = label;
    labelInput.setAttribute("aria-label", `${axisKey} label`);

    // Disable label editing when auto-label (policy) mode is active
    labelInput.disabled = dom.autoLabelToggle.checked;

    // Highlight the label if it differs from the originally loaded example
    if (orig && label !== orig.label) {
      labelInput.classList.add("axis-modified");
    }

    // ── Wire per-axis event listeners ──────────────────────────────── //

    /** Slider input: update score in state + textarea + score display. */
    slider.addEventListener("input", () => {
      const newScore = parseFloat(slider.value);
      scoreDisplay.textContent = newScore.toFixed(3);

      // Toggle modification highlight relative to original example
      const origAxis = state.originalAxes && state.originalAxes[axisKey];
      if (origAxis) {
        scoreDisplay.classList.toggle(
          "axis-modified",
          Math.abs(newScore - origAxis.score) > 0.0001
        );
      }

      // Write back into state (mutate in place to preserve other fields)
      state.payload.axes[axisKey] = {
        ...state.payload.axes[axisKey],
        score: newScore,
      };

      syncJsonTextarea();
    });

    /** Label input: update label in state + textarea. */
    labelInput.addEventListener("input", () => {
      // Toggle modification highlight relative to original example
      const origAxis = state.originalAxes && state.originalAxes[axisKey];
      if (origAxis) {
        labelInput.classList.toggle(
          "axis-modified",
          labelInput.value !== origAxis.label
        );
      }

      state.payload.axes[axisKey] = {
        ...state.payload.axes[axisKey],
        label: labelInput.value,
      };
      syncJsonTextarea();
    });

    // Assemble row: name | slider+score | label
    row.appendChild(nameEl);
    row.appendChild(sliderWrap);
    row.appendChild(labelInput);

    fragment.appendChild(row);
  }

  // Replace slider panel contents in one DOM operation
  dom.sliderPanel.textContent = "";
  dom.sliderPanel.appendChild(fragment);
}

/**
 * Return the effective model name: prefers the `<select>` if it has a
 * non-empty value, otherwise falls back to the manual text `<input>`.
 *
 * The `<select>` is populated from Ollama's model list; the `<input>` is
 * shown as a fallback when Ollama is unreachable.
 *
 * @returns {string} The trimmed model name (may be empty if nothing is set).
 */
export function getModelName() {
  const sel = dom.modelSelect.value.trim();
  return sel || dom.modelInput.value.trim();
}

/**
 * Return the current Ollama server URL from the host input field.
 *
 * Strips trailing whitespace; the value is sent as-is to the backend,
 * which handles trailing-slash normalisation.
 *
 * @returns {string} The Ollama base URL (e.g. "http://localhost:11434").
 */
export function getOllamaHost() {
  return dom.ollamaHostInput.value.trim();
}

/**
 * Resolve the seed value from the seed input field.
 *
 * If the input is -1 (or any negative value), generate a random 32-bit
 * unsigned integer using `Math.random()`.  This does NOT pollute any
 * global RNG state — `Math.random()` is stateless from the caller's
 * perspective and the generated seed is used solely to populate the
 * payload's `seed` field for this single request.
 *
 * Positive values are passed through as-is, providing deterministic
 * reproducibility when the same seed is reused.
 *
 * @returns {number} A non-negative integer seed (0 to 2^32 - 1).
 */
export function resolveSeed() {
  const raw = parseInt(dom.seedInput.value, 10);
  if (isNaN(raw) || raw < 0) {
    // Generate a random 32-bit unsigned integer (0 to 4294967295).
    // Math.random() is not a CSPRNG but is perfectly adequate for
    // non-security seed generation in a local lab tool.
    return Math.floor(Math.random() * 0x100000000);
  }
  return raw;
}

/**
 * Fetch the model list from the Ollama instance at the given (or current)
 * host URL and repopulate the model `<select>` dropdown.
 *
 * On success the dropdown is shown and the manual text input is hidden.
 * On failure (unreachable host, empty list) the dropdown is hidden and
 * the manual input is shown so the user can type a model name directly.
 *
 * The previously selected model is re-selected if it exists in the new
 * list, preserving the user's choice across host changes.
 *
 * @param {string} [host] - Ollama host URL.  Defaults to the input field value.
 */
export async function refreshModels(host) {
  const h = host || getOllamaHost();
  const url = h ? `/api/models?host=${encodeURIComponent(h)}` : "/api/models";

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const models = await res.json();

    if (models.length > 0) {
      // Remember the currently selected model so we can re-select it
      const prev = getModelName();
      dom.modelSelect.innerHTML = "";

      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === prev) opt.selected = true;
        dom.modelSelect.appendChild(opt);
      }

      dom.modelSelect.classList.remove("hidden");
      dom.modelInput.classList.add("hidden");
    } else {
      // No models found — show manual input
      dom.modelSelect.classList.add("hidden");
      dom.modelInput.classList.remove("hidden");
    }
  } catch {
    // Ollama unreachable at this host — show manual input
    dom.modelSelect.classList.add("hidden");
    dom.modelInput.classList.remove("hidden");
  }
}

/**
 * Keep the temperature range slider and number input in bidirectional sync.
 *
 * Dragging the slider updates the number input; typing in the number input
 * clamps the value to [0, 2] and updates the slider position.
 *
 * @private — called only by wireSyncEvents().
 */
function wireTempSync() {
  dom.tempRange.addEventListener("input", () => {
    dom.tempInput.value = dom.tempRange.value;
  });

  dom.tempInput.addEventListener("input", () => {
    const v = clamp(parseFloat(dom.tempInput.value) || 0, 0, 2);
    dom.tempRange.value = v;
  });
}

/**
 * Wire sync-related event listeners.
 *
 * Registers handlers for:
 *   - JSON textarea input (debounced 280ms) → parse → rebuild sliders
 *   - System prompt textarea input → update override badge
 *   - Temperature range ↔ number input sync
 *
 * Called once during startup by the mod-events coordinator.
 */
export function wireSyncEvents() {
  // ── JSON textarea (debounced) → parse → rebuild sliders ────────── //
  // Debounced to 280ms so we don't re-parse on every keystroke while
  // the user is editing the raw JSON.
  dom.jsonTextarea.addEventListener(
    "input",
    debounce(() => {
      const obj = safeParse(dom.jsonTextarea.value);
      if (!obj) {
        setJsonBadge(false);
        setStatus("Invalid JSON.");
        return;
      }
      setJsonBadge(true);
      state.payload = obj;
      buildSlidersFromJson();
      setStatus("JSON updated.");
    }, 280)
  );

  // ── System prompt textarea → update override badge ──────────────── //
  dom.systemPromptTextarea.addEventListener("input", () => {
    updateSystemPromptBadge();
  });

  // ── Temperature range ↔ number input sync ──────────────────────── //
  wireTempSync();
}

/**
 * app/static/app.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Axis Descriptor Lab – all client-side interactive behaviour.
 *
 * Architecture
 * ────────────
 * The single canonical source of truth is `state.payload`, a plain JS object
 * mirroring the AxisPayload schema.  Everything else (sliders, JSON textarea,
 * output, diff) is derived from or writes back to that object.
 *
 * Data flow
 * ─────────
 *   JSON textarea edit  ──→  safeParse  ──→  state.payload  ──→  buildSliders
 *   Slider change       ──→  state.payload  ──→  syncJsonTextarea
 *   Label input change  ──→  state.payload  ──→  syncJsonTextarea
 *   Generate button     ──→  POST /api/generate  ──→  output box + diff B
 *   "Set as A" button   ──→  state.baseline stored
 *   "Recompute" button  ──→  POST /api/relabel  ──→  state.payload  ──→  rebuild
 *
 * Diff algorithm
 * ──────────────
 * A simple word-level LCS diff is computed client-side on the two output
 * strings.  Words added in B are highlighted green; words removed from A
 * are highlighted red with strikethrough.
 */

"use strict";

/* ═══════════════════════════════════════════════════════════════════════════
   STATE
════════════════════════════════════════════════════════════════════════════ */

/**
 * Application-level mutable state.
 * @type {{
 *   payload:  object|null,   // current AxisPayload JS object
 *   baseline: string|null,   // baseline (A) generated text for diffing
 *   current:  string|null,   // latest (B) generated text
 *   busy:     boolean        // true while a generate request is in-flight
 * }}
 */
const state = {
  payload:  null,
  baseline: null,
  current:  null,
  busy:     false,
};

/* ═══════════════════════════════════════════════════════════════════════════
   DOM REFERENCES
   All queried once at startup – avoids repeated querySelector calls.
════════════════════════════════════════════════════════════════════════════ */

const $ = (id) => document.getElementById(id);

const exampleSelect       = $("example-select");
const btnLoadExample      = $("btn-load-example");
const jsonTextarea        = $("json-textarea");
const jsonStatusBadge     = $("json-status-badge");
const systemPromptTextarea = $("system-prompt-textarea");
const sliderPanel         = $("slider-panel");
const autoLabelToggle     = $("auto-label-toggle");
const btnRelabel          = $("btn-relabel");
const modelSelect         = $("model-select");
const modelInput          = $("model-input");
const tempRange           = $("temp-range");
const tempInput           = $("temp-input");
const tokensInput         = $("tokens-input");
const seedInput           = $("seed-input");
const btnGenerate         = $("btn-generate");
const outputBox           = $("output-box");
const outputMeta          = $("output-meta");
const btnSetBaseline      = $("btn-set-baseline");
const btnClearOutput      = $("btn-clear-output");
const diffA               = $("diff-a");
const diffB               = $("diff-b");
const diffDelta           = $("diff-delta");
const statusText          = $("status-text");
const spinner             = $("spinner");

/* ═══════════════════════════════════════════════════════════════════════════
   UTILITIES
════════════════════════════════════════════════════════════════════════════ */

/**
 * Resolve the seed value from the seed input field.
 *
 * If the input is -1 (or any negative value), generate a random 32-bit
 * unsigned integer using Math.random().  This does NOT pollute any global
 * RNG state — Math.random() is stateless from the caller's perspective and
 * the generated seed is used solely to populate the payload's `seed` field
 * for this single request.
 *
 * Positive values are passed through as-is, providing deterministic
 * reproducibility when the same seed is reused.
 *
 * @returns {number} A non-negative integer seed.
 */
function resolveSeed() {
  const raw = parseInt(seedInput.value, 10);
  if (isNaN(raw) || raw < 0) {
    // Generate a random 32-bit unsigned integer (0 to 4294967295).
    // Math.random() is not a CSPRNG but is perfectly adequate for
    // non-security seed generation in a local lab tool.
    return Math.floor(Math.random() * 0x100000000);
  }
  return raw;
}

/**
 * Debounce: delay invoking `fn` until `ms` milliseconds have passed since
 * the last call.  Used to avoid rebuilding sliders on every keystroke.
 *
 * @param {Function} fn   - Function to debounce.
 * @param {number}   ms   - Delay in milliseconds.
 * @returns {Function}
 */
function debounce(fn, ms) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

/**
 * Attempt to parse a JSON string.  Returns the parsed object on success or
 * null on failure.  Never throws.
 *
 * @param {string} str - Raw JSON string.
 * @returns {object|null}
 */
function safeParse(str) {
  try {
    return JSON.parse(str);
  } catch {
    return null;
  }
}

/**
 * Clamp a numeric value to [min, max].
 *
 * @param {number} val
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

/**
 * Return the effective model name: prefers the <select> if it has a
 * non-empty value, otherwise falls back to the manual text <input>.
 *
 * @returns {string}
 */
function getModelName() {
  const sel = modelSelect.value.trim();
  return sel || modelInput.value.trim();
}

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS BAR
════════════════════════════════════════════════════════════════════════════ */

/**
 * Update the status bar text and spinner visibility.
 *
 * @param {string}  msg     - Message to display.
 * @param {boolean} [busy]  - If true, shows the spinner.
 */
function setStatus(msg, busy = false) {
  statusText.textContent = msg;
  spinner.classList.toggle("hidden", !busy);
}

/* ═══════════════════════════════════════════════════════════════════════════
   JSON TEXTAREA  ↔  SLIDER SYNC
════════════════════════════════════════════════════════════════════════════ */

/**
 * Reflect `state.payload` back into the JSON textarea as pretty-printed JSON.
 * Called after every slider / label change so the textarea stays in sync.
 */
function syncJsonTextarea() {
  jsonTextarea.value = JSON.stringify(state.payload, null, 2);
}

/**
 * Mark the JSON status badge as valid or error.
 *
 * @param {boolean} valid
 */
function setJsonBadge(valid) {
  if (valid) {
    jsonStatusBadge.textContent = "OK";
    jsonStatusBadge.className   = "badge";
  } else {
    jsonStatusBadge.textContent = "ERR";
    jsonStatusBadge.className   = "badge badge--err";
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   SLIDER PANEL
════════════════════════════════════════════════════════════════════════════ */

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
 * `state.payload` and call `syncJsonTextarea()`.
 */
function buildSlidersFromJson() {
  const payload = state.payload;

  // Guard: nothing to render
  if (!payload || typeof payload.axes !== "object" || payload.axes === null) {
    sliderPanel.innerHTML = '<p class="placeholder-text">No axes found in payload.</p>';
    return;
  }

  const axes = payload.axes;
  const keys = Object.keys(axes);

  if (keys.length === 0) {
    sliderPanel.innerHTML = '<p class="placeholder-text">axes object is empty.</p>';
    return;
  }

  // Build DOM fragment for all rows
  const fragment = document.createDocumentFragment();

  for (const axisKey of keys) {
    const axisVal = axes[axisKey];

    // Normalise the value in case it came from malformed JSON
    const score  = clamp(parseFloat(axisVal.score)  || 0, 0, 1);
    const label  = String(axisVal.label || "");

    const row = document.createElement("div");
    row.className    = "axis-row";
    row.dataset.axis = axisKey;

    // ── Axis name ──────────────────────────────────────────────────────── //
    const nameEl = document.createElement("span");
    nameEl.className   = "axis-name";
    nameEl.textContent = axisKey;
    nameEl.title       = axisKey;

    // ── Centre column: slider + score display ──────────────────────────── //
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

    sliderWrap.appendChild(slider);
    sliderWrap.appendChild(scoreDisplay);

    // ── Label input ────────────────────────────────────────────────────── //
    const labelInput = document.createElement("input");
    labelInput.type      = "text";
    labelInput.className = "axis-label-input";
    labelInput.value     = label;
    labelInput.setAttribute("aria-label", `${axisKey} label`);

    // Disable label editing when auto-label mode is active
    labelInput.disabled = autoLabelToggle.checked;

    // ── Wire up events ─────────────────────────────────────────────────── //

    /**
     * Slider input: update score in state + textarea + score display.
     */
    slider.addEventListener("input", () => {
      const newScore = parseFloat(slider.value);
      scoreDisplay.textContent = newScore.toFixed(3);

      // Write back into state (mutate in place to preserve other fields)
      state.payload.axes[axisKey] = {
        ...state.payload.axes[axisKey],
        score: newScore,
      };

      syncJsonTextarea();
    });

    /**
     * Label input: update label in state + textarea.
     * No auto-relabel is triggered on manual edit – the user owns the label.
     */
    labelInput.addEventListener("input", () => {
      state.payload.axes[axisKey] = {
        ...state.payload.axes[axisKey],
        label: labelInput.value,
      };
      syncJsonTextarea();
    });

    // Assemble row
    row.appendChild(nameEl);
    row.appendChild(sliderWrap);
    row.appendChild(labelInput);

    fragment.appendChild(row);
  }

  sliderPanel.innerHTML = "";
  sliderPanel.appendChild(fragment);
}

/* ═══════════════════════════════════════════════════════════════════════════
   EXAMPLE LOADER
════════════════════════════════════════════════════════════════════════════ */

/**
 * Fetch the list of available examples from the server and populate the
 * example <select> dropdown.  Called once at startup.
 */
async function loadExampleList() {
  try {
    const res  = await fetch("/api/examples");
    const list = await res.json();
    exampleSelect.innerHTML = '<option value="">— choose —</option>';
    for (const name of list) {
      const opt   = document.createElement("option");
      opt.value   = name;
      opt.textContent = name;
      exampleSelect.appendChild(opt);
    }
  } catch (err) {
    setStatus(`Failed to load examples: ${err.message}`);
  }
}

/**
 * Load a named example from the server, update state, rebuild sliders and
 * sync the JSON textarea.
 *
 * @param {string} name - Example stem (e.g. "example_a").
 */
async function loadExample(name) {
  if (!name) return;

  setStatus(`Loading ${name}…`, true);
  try {
    const res     = await fetch(`/api/examples/${name}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();

    state.payload = payload;
    syncJsonTextarea();
    buildSlidersFromJson();
    setJsonBadge(true);
    setStatus(`Loaded ${name}.`);
  } catch (err) {
    setStatus(`Error loading example: ${err.message}`);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   GENERATE
════════════════════════════════════════════════════════════════════════════ */

/**
 * POST /api/generate with current state, then update the output box and
 * populate diff B.
 *
 * Disables the generate button while in-flight to prevent double-submissions.
 */
async function generate() {
  if (state.busy) return;

  if (!state.payload) {
    setStatus("No payload loaded – paste JSON or load an example.");
    return;
  }

  const model       = getModelName();
  const temperature = parseFloat(tempInput.value);
  const max_tokens  = parseInt(tokensInput.value, 10);
  const rawSeed     = parseInt(seedInput.value, 10);
  const wasRandom   = isNaN(rawSeed) || rawSeed < 0;
  const seed        = resolveSeed();

  if (!model) {
    setStatus("No model specified.");
    return;
  }

  // Apply the resolved seed to the payload so the request carries the
  // actual seed used (important for logging and reproducibility).
  state.payload.seed = seed;
  syncJsonTextarea();

  const systemPromptVal = systemPromptTextarea.value.trim();

  const reqBody = {
    payload:       state.payload,
    model,
    temperature,
    max_tokens,
    system_prompt: systemPromptVal || null,
  };

  state.busy = true;
  btnGenerate.disabled = true;
  setStatus(`Generating via ${model}…`, true);

  // Clear previous output
  outputBox.textContent   = "";
  outputMeta.textContent  = "";
  outputMeta.classList.add("hidden");

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

    // ── Update output box ─────────────────────────────────────────────── //
    outputBox.textContent = data.text;
    state.current = data.text;

    // ── Show meta info ────────────────────────────────────────────────── //
    const seedLabel = wasRandom ? `seed: ${seed} (random)` : `seed: ${seed}`;
    let metaStr = `model: ${data.model}  ·  temp: ${data.temperature}  ·  ${seedLabel}`;
    if (data.usage) {
      const p = data.usage.prompt_eval_count;
      const e = data.usage.eval_count;
      if (p !== null && p !== undefined) metaStr += `  ·  prompt tokens: ${p}`;
      if (e !== null && e !== undefined) metaStr += `  ·  gen tokens: ${e}`;
    }
    outputMeta.textContent = metaStr;
    outputMeta.classList.remove("hidden");

    // ── Update diff B ─────────────────────────────────────────────────── //
    diffB.textContent = data.text;
    updateDiff();

    setStatus(`Done (${data.model}).`);
  } catch (err) {
    outputBox.innerHTML = `<span style="color:var(--col-err);">Error: ${escapeHtml(err.message)}</span>`;
    setStatus(`Error: ${err.message}`);
  } finally {
    state.busy           = false;
    btnGenerate.disabled = false;
    spinner.classList.add("hidden");
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   DIFF
════════════════════════════════════════════════════════════════════════════ */

/**
 * Compute and render the word-level diff between `state.baseline` (A) and
 * `state.current` (B).
 *
 * Algorithm: patience-like LCS on word tokens.
 *   - Words only in A → shown red + strikethrough in delta view.
 *   - Words only in B → shown green in delta view.
 *   - Common words → shown as-is.
 *
 * The A and B boxes show the raw texts.
 * The Δ box shows the annotated diff.
 */
function updateDiff() {
  const textA = state.baseline || "";
  const textB = state.current  || "";

  // Update the raw side-by-side panels
  diffA.textContent = textA || "(no baseline)";
  diffB.textContent = textB || "(no output)";

  if (!textA || !textB) {
    diffDelta.innerHTML = '<span class="placeholder-text">Set a baseline and generate to compare.</span>';
    return;
  }

  // Tokenise: split on whitespace, keeping tokens
  const wordsA = tokenise(textA);
  const wordsB = tokenise(textB);

  const diff = lcsWordDiff(wordsA, wordsB);

  // Build annotated HTML
  const fragment = document.createDocumentFragment();

  for (const [op, word] of diff) {
    if (op === "=") {
      fragment.appendChild(document.createTextNode(word + " "));
    } else if (op === "+") {
      const span = document.createElement("span");
      span.className   = "diff-add";
      span.textContent = word;
      fragment.appendChild(span);
      fragment.appendChild(document.createTextNode(" "));
    } else if (op === "-") {
      const span = document.createElement("span");
      span.className   = "diff-del";
      span.textContent = word;
      fragment.appendChild(span);
      fragment.appendChild(document.createTextNode(" "));
    }
  }

  diffDelta.innerHTML = "";
  diffDelta.appendChild(fragment);

  // Open the diff <details> automatically if it isn't already
  const detailsEl = document.getElementById("diff-details");
  if (detailsEl && !detailsEl.open) {
    detailsEl.open = true;
  }
}

/**
 * Split text into word tokens, preserving punctuation attached to words.
 * Consecutive whitespace is collapsed.
 *
 * @param {string} text
 * @returns {string[]}
 */
function tokenise(text) {
  return text.trim().split(/\s+/).filter(Boolean);
}

/**
 * Compute a word-level LCS-based diff between two token arrays.
 * Returns an array of [operation, word] tuples where operation is:
 *   "=" – word is in both A and B (common)
 *   "+" – word is only in B (added)
 *   "-" – word is only in A (removed)
 *
 * Uses the standard dynamic-programming LCS algorithm (O(m*n)).
 * For very long texts this could be slow, but paragraph-length outputs
 * (≤ 200 words) are always fast enough for a local dev tool.
 *
 * @param {string[]} a - Token array for text A.
 * @param {string[]} b - Token array for text B.
 * @returns {Array<[string, string]>}
 */
function lcsWordDiff(a, b) {
  const m = a.length;
  const n = b.length;

  // Build LCS length table
  // dp[i][j] = length of LCS of a[0..i-1] and b[0..j-1]
  const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // Backtrack to reconstruct diff
  const result = [];
  let i = m, j = n;

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      result.push(["=", a[i - 1]]);
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      result.push(["+", b[j - 1]]);
      j--;
    } else {
      result.push(["-", a[i - 1]]);
      i--;
    }
  }

  result.reverse();
  return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
   AUTO-RELABEL
════════════════════════════════════════════════════════════════════════════ */

/**
 * POST /api/relabel with the current payload.
 * The server applies its policy table and returns an updated payload with
 * labels recomputed from scores.
 * Updates state, textarea, and sliders.
 */
async function relabel() {
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

/* ═══════════════════════════════════════════════════════════════════════════
   LOGGING
════════════════════════════════════════════════════════════════════════════ */

/**
 * Fire-and-forget call to POST /api/log.
 * Failures are logged to the browser console but do not surface to the user.
 *
 * @param {string} output      - Generated text to log.
 * @param {string} model       - Model name used.
 * @param {number} temperature - Temperature used.
 * @param {number} max_tokens  - Token budget used.
 */
async function logRun(output, model, temperature, max_tokens) {
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

/* ═══════════════════════════════════════════════════════════════════════════
   HTML ESCAPING (safety utility)
════════════════════════════════════════════════════════════════════════════ */

/**
 * Escape a string for safe insertion into HTML.
 * Prevents XSS when displaying error messages that may contain server text.
 *
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g,  "&amp;")
    .replace(/</g,  "&lt;")
    .replace(/>/g,  "&gt;")
    .replace(/"/g,  "&quot;")
    .replace(/'/g,  "&#39;");
}

/* ═══════════════════════════════════════════════════════════════════════════
   TEMPERATURE SYNC (range ↔ number)
════════════════════════════════════════════════════════════════════════════ */

/**
 * Keep the temperature range slider and number input in sync.
 * When the user drags the slider, the number input updates, and vice-versa.
 */
function wireTempSync() {
  tempRange.addEventListener("input", () => {
    tempInput.value = tempRange.value;
  });

  tempInput.addEventListener("input", () => {
    const v = clamp(parseFloat(tempInput.value) || 0, 0, 2);
    tempRange.value = v;
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   EVENT WIRING
════════════════════════════════════════════════════════════════════════════ */

/**
 * Wire all interactive elements.  Called once after DOMContentLoaded.
 */
function wireEvents() {
  // ── Load example button ───────────────────────────────────────────────── //
  btnLoadExample.addEventListener("click", () => {
    const name = exampleSelect.value;
    if (name) loadExample(name);
  });

  // Double-click the dropdown also loads (convenience)
  exampleSelect.addEventListener("dblclick", () => {
    const name = exampleSelect.value;
    if (name) loadExample(name);
  });

  // ── JSON textarea (debounced) → parse → rebuild sliders ──────────────── //
  jsonTextarea.addEventListener(
    "input",
    debounce(() => {
      const obj = safeParse(jsonTextarea.value);
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

  // ── Auto-label toggle: rebuild sliders to enable/disable label inputs ─── //
  autoLabelToggle.addEventListener("change", () => {
    // Just rebuild; buildSlidersFromJson reads the toggle state
    buildSlidersFromJson();
    // If turning on auto, immediately recompute from server policy
    if (autoLabelToggle.checked) {
      relabel();
    }
  });

  // ── Relabel button ────────────────────────────────────────────────────── //
  btnRelabel.addEventListener("click", () => {
    relabel();
  });

  // ── Temperature sync ──────────────────────────────────────────────────── //
  wireTempSync();

  // ── Generate ─────────────────────────────────────────────────────────── //
  btnGenerate.addEventListener("click", () => {
    generate();
  });

  // ── Set baseline (A) ─────────────────────────────────────────────────── //
  btnSetBaseline.addEventListener("click", () => {
    if (!state.current) {
      setStatus("Generate something first.");
      return;
    }
    state.baseline = state.current;
    diffA.textContent = state.baseline;
    btnSetBaseline.classList.add("is-active");
    setStatus("Baseline A set.");
  });

  // ── Clear output ─────────────────────────────────────────────────────── //
  btnClearOutput.addEventListener("click", () => {
    outputBox.innerHTML  = '<span class="placeholder-text">Click Generate to produce a description.</span>';
    outputMeta.textContent = "";
    outputMeta.classList.add("hidden");
    state.current        = null;
    state.baseline       = null;
    diffA.innerHTML      = '<span class="placeholder-text">No baseline set.</span>';
    diffB.innerHTML      = '<span class="placeholder-text">Generate to populate B.</span>';
    diffDelta.innerHTML  = '<span class="placeholder-text">—</span>';
    btnSetBaseline.classList.remove("is-active");
    setStatus("Output cleared.");
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   INIT
════════════════════════════════════════════════════════════════════════════ */

/**
 * Application entry point.  Runs once the DOM is ready.
 *
 * 1. Wires all event listeners.
 * 2. Fetches the example list from the server.
 * 3. Auto-loads the first example so the UI is not empty on first visit.
 */
/**
 * Initialise the theme toggle button and restore any saved preference.
 */
function wireThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;

  const applyTheme = (theme) => {
    document.documentElement.setAttribute("data-theme", theme);
    btn.textContent = theme === "light" ? "\u263E Dark" : "\u2600 Light";
    localStorage.setItem("padl-theme", theme);
  };

  // Restore saved preference, default to dark
  const saved = localStorage.getItem("padl-theme") || "dark";
  applyTheme(saved);

  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(current === "dark" ? "light" : "dark");
  });
}

async function init() {
  wireThemeToggle();
  wireEvents();
  await loadExampleList();

  // Auto-load the first example if any exist
  const firstOption = exampleSelect.options[1]; // [0] is the placeholder
  if (firstOption) {
    exampleSelect.value = firstOption.value;
    await loadExample(firstOption.value);
  }

  setStatus("Ready. Load an example or paste JSON to begin.");
}

// Boot when the DOM is fully parsed
document.addEventListener("DOMContentLoaded", init);

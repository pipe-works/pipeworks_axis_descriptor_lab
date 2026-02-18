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
const systemPromptBadge    = $("system-prompt-badge");
const promptSelect         = $("prompt-select");
const btnLoadPrompt        = $("btn-load-prompt");
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
const btnSave             = $("btn-save");
const btnClearOutput      = $("btn-clear-output");
const diffA               = $("diff-a");
const diffB               = $("diff-b");
const diffDelta           = $("diff-delta");
const signalPanel         = $("signal-panel");
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
 * Create a styled placeholder <span> element.
 *
 * Used instead of innerHTML to avoid any XSS surface when rendering
 * placeholder text inside output/diff panels.
 *
 * @param {string} text - The placeholder message to display.
 * @returns {HTMLSpanElement}
 */
function makePlaceholder(text) {
  const span = document.createElement("span");
  span.className = "placeholder-text";
  span.textContent = text;
  return span;
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

/**
 * Update the system prompt "override" badge to reflect whether a custom
 * prompt is active.
 *
 * When the textarea contains user text the badge switches to amber
 * (badge--active) to clearly signal that the server default is being
 * overridden.  When the textarea is empty (server default in effect),
 * the badge reverts to the muted grey style.
 */
function updateSystemPromptBadge() {
  const hasContent = systemPromptTextarea.value.trim().length > 0;
  if (hasContent) {
    systemPromptBadge.className = "badge badge--active";
  } else {
    systemPromptBadge.className = "badge badge--muted";
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
    sliderPanel.textContent = "";
    const noAxesP = document.createElement("p");
    noAxesP.className = "placeholder-text";
    noAxesP.textContent = "No axes found in payload.";
    sliderPanel.appendChild(noAxesP);
    return;
  }

  const axes = payload.axes;
  const keys = Object.keys(axes);

  if (keys.length === 0) {
    sliderPanel.textContent = "";
    const emptyP = document.createElement("p");
    emptyP.className = "placeholder-text";
    emptyP.textContent = "axes object is empty.";
    sliderPanel.appendChild(emptyP);
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

  sliderPanel.textContent = "";
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
    const defaultOpt = new Option("\u2014 choose \u2014", "");
    const opts = list.map((name) => new Option(name, name));
    exampleSelect.replaceChildren(defaultOpt, ...opts);
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
   PROMPT LIBRARY LOADER
   ─────────────────────────────────────────────────────────────────────────
   Mirrors the EXAMPLE LOADER pattern above, but for system prompt text
   files instead of JSON axis payloads.

   The prompt library lives in app/prompts/ as .txt files.  The API returns
   file stems (e.g. "system_prompt_v01") and the content as plain text.
   Loading a prompt populates the system prompt textarea, overriding any
   existing custom text.
════════════════════════════════════════════════════════════════════════════ */

/**
 * Fetch the list of available system prompts from the server and populate
 * the prompt <select> dropdown.  Called once at startup.
 *
 * Mirrors ``loadExampleList()`` but hits ``/api/prompts`` instead of
 * ``/api/examples``.  The list contains prompt file stems (e.g.
 * "system_prompt_v01", "system_prompt_v02_terse").
 */
async function loadPromptList() {
  try {
    const res  = await fetch("/api/prompts");
    const list = await res.json();

    // Replace the placeholder option with the full list from the server
    const defaultOpt = new Option("\u2014 choose \u2014", "");
    const opts = list.map((name) => new Option(name, name));
    promptSelect.replaceChildren(defaultOpt, ...opts);
  } catch (err) {
    setStatus(`Failed to load prompt list: ${err.message}`);
  }
}

/**
 * Load a named system prompt from the server and populate the system
 * prompt override textarea with its content.
 *
 * Fetches plain text from ``/api/prompts/{name}`` (not JSON — the endpoint
 * returns ``PlainTextResponse``) and sets it as the textarea value.  The
 * loaded text takes effect on the next generate call: the backend uses
 * whatever is in ``system_prompt``, falling back to the server default
 * only when the textarea is empty.
 *
 * After loading, the System Prompt ``<details>`` is opened so the user
 * can immediately see the loaded content.
 *
 * @param {string} name - Prompt stem (e.g. "system_prompt_v01").
 */
async function loadPrompt(name) {
  if (!name) return;

  setStatus(`Loading prompt ${name}…`, true);
  try {
    const res = await fetch(`/api/prompts/${name}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    // Use .text() because the endpoint returns PlainTextResponse, not JSON
    const text = await res.text();

    // Populate the system prompt textarea with the loaded content
    systemPromptTextarea.value = text;

    // Update the override badge to reflect that custom content is now active
    updateSystemPromptBadge();

    // Ensure the System Prompt <details> is open so the user can see it
    const details = document.getElementById("system-prompt-details");
    if (details && !details.open) details.open = true;

    setStatus(`Loaded prompt: ${name}`);
  } catch (err) {
    setStatus(`Error loading prompt: ${err.message}`);
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

    // ── Show meta info as a clean key-value table ─────────────────────── //
    // Builds a faint two-column table: labels on the left, values on the
    // right.  Much more scannable than the previous dot-separated string.

    const seedVal = wasRandom ? `${seed} (random)` : `${seed}`;

    // Collect rows as [key, value] pairs; skip any with missing values.
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

    // Build the <table> element
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
      tr.appendChild(tdVal);

      table.appendChild(tr);
    }

    outputMeta.textContent = "";
    outputMeta.appendChild(table);
    outputMeta.classList.remove("hidden");

    // ── Update diff B ─────────────────────────────────────────────────── //
    diffB.textContent = data.text;
    updateDiff();

    setStatus(`Done (${data.model}).`);
  } catch (err) {
    const errSpan = document.createElement("span");
    errSpan.style.color = "var(--col-err)";
    errSpan.textContent = `Error: ${err.message}`;
    outputBox.textContent = "";
    outputBox.appendChild(errSpan);
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
    diffDelta.textContent = "";
    diffDelta.appendChild(makePlaceholder("Set a baseline and generate to compare."));
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

  diffDelta.textContent = "";
  diffDelta.appendChild(fragment);

  // Open the diff <details> automatically if it isn't already
  const detailsEl = document.getElementById("diff-details");
  if (detailsEl && !detailsEl.open) {
    detailsEl.open = true;
  }

  // After building the word-level diff, also run signal isolation
  // to surface meaningful content-word pivots.
  updateSignalIsolation();
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIGNAL ISOLATION (content-word delta)
   ─────────────────────────────────────────────────────────────────────────
   Calls POST /api/analyze-delta after each diff computation to surface
   meaningful lexical pivots by filtering structural noise.

   The pipeline is server-side:
   1. Tokenise both texts (NLTK)
   2. Lemmatise to base forms (WordNet)
   3. Remove stopwords (English)
   4. Compute set difference

   The frontend simply renders the returned removed/added word lists
   as inline tags.
════════════════════════════════════════════════════════════════════════════ */

/**
 * Call POST /api/analyze-delta with the current baseline and output texts,
 * then render the results into the signal panel.
 *
 * Called at the end of updateDiff() when both baseline and current text
 * exist.  Failures are shown inline in the panel, not thrown.
 */
async function updateSignalIsolation() {
  const textA = state.baseline || "";
  const textB = state.current  || "";

  // If either text is missing, show a placeholder and bail.
  if (!textA || !textB) {
    signalPanel.textContent = "";
    signalPanel.appendChild(
      makePlaceholder("Set a baseline (A) and generate to analyze content-word changes.")
    );
    return;
  }

  try {
    const res = await fetch("/api/analyze-delta", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        baseline_text: textA,
        current_text:  textB,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Build the display as a document fragment (avoids reflows).
    const fragment = document.createDocumentFragment();

    // ── No changes ────────────────────────────────────────────────────── //
    if (data.removed.length === 0 && data.added.length === 0) {
      fragment.appendChild(makePlaceholder("No content-word differences detected."));
    } else {
      // ── Two-column "Semantic Pivot" layout ──────────────────────────── //
      // REMOVED on the left, ADDED on the right.  The side-by-side
      // arrangement lets the eye scan left→right to grasp the lexical
      // pivot at a glance.

      const grid = document.createElement("div");
      grid.className = "signal-columns";

      // ── Left column: REMOVED words ──────────────────────────────────── //
      const removedCol = document.createElement("div");

      const removedHeader = document.createElement("div");
      removedHeader.className = "signal-header";
      removedHeader.textContent = `REMOVED (${data.removed.length})`;
      removedCol.appendChild(removedHeader);

      const removedList = document.createElement("div");
      removedList.className = "signal-word-list";
      for (const word of data.removed) {
        const el = document.createElement("div");
        el.className = "signal-word signal-word--removed";
        el.textContent = word;
        removedList.appendChild(el);
      }
      removedCol.appendChild(removedList);

      // ── Right column: ADDED words ───────────────────────────────────── //
      const addedCol = document.createElement("div");

      const addedHeader = document.createElement("div");
      addedHeader.className = "signal-header";
      addedHeader.textContent = `ADDED (${data.added.length})`;
      addedCol.appendChild(addedHeader);

      const addedList = document.createElement("div");
      addedList.className = "signal-word-list";
      for (const word of data.added) {
        const el = document.createElement("div");
        el.className = "signal-word signal-word--added";
        el.textContent = word;
        addedList.appendChild(el);
      }
      addedCol.appendChild(addedList);

      grid.appendChild(removedCol);
      grid.appendChild(addedCol);
      fragment.appendChild(grid);
    }

    // Replace panel contents with the rendered fragment.
    signalPanel.textContent = "";
    signalPanel.appendChild(fragment);

    // Open the signal <details> automatically if it isn't already.
    const signalDetailsEl = document.getElementById("signal-details");
    if (signalDetailsEl && !signalDetailsEl.open) {
      signalDetailsEl.open = true;
    }

  } catch (err) {
    // Show error inline in the panel rather than throwing.
    signalPanel.textContent = "";
    const errSpan = document.createElement("span");
    errSpan.style.color = "var(--col-err)";
    errSpan.textContent = `Signal analysis error: ${err.message}`;
    signalPanel.appendChild(errSpan);
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
   SAVE
   ─────────────────────────────────────────────────────────────────────────
   Persist a complete session snapshot to a timestamped subfolder under
   data/ via POST /api/save.

   The backend writes individual files:
     metadata.json     – model, temperature, max_tokens, seed, timestamp, hash
     payload.json      – the full AxisPayload
     system_prompt.md  – the system prompt used
     output.md         – the generated text (if any)
     baseline.md       – the baseline text (if set)

   The frontend must resolve the effective system prompt before sending:
   if the user typed a custom prompt, send that; otherwise fetch the server
   default via GET /api/system-prompt so the saved file is always complete.
════════════════════════════════════════════════════════════════════════════ */

/**
 * Save the current session state to a timestamped folder under data/.
 *
 * Collects state.payload, state.current, state.baseline, the active model,
 * temperature, max_tokens, and the effective system prompt (custom override
 * if provided, otherwise fetched from the server default endpoint).
 *
 * On success, updates the status bar with the folder name and file list.
 * On failure, surfaces the error in the status bar without throwing.
 *
 * The system prompt must always be resolved and sent verbatim so the saved
 * file reflects exactly what the LLM received, not a null/placeholder.
 */
async function saveRun() {
  // Guard: require a payload to be loaded before saving
  if (!state.payload) {
    setStatus("Nothing to save \u2013 load a payload first.");
    return;
  }

  // ── Resolve the effective system prompt ────────────────────────────────
  // If the user has typed anything in the system prompt textarea, use that
  // verbatim.  Otherwise fetch the server default via GET /api/system-prompt
  // so the saved file is always complete and accurate.
  let systemPromptVal = systemPromptTextarea.value.trim();
  if (!systemPromptVal) {
    try {
      const defaultRes = await fetch("/api/system-prompt");
      if (defaultRes.ok) {
        systemPromptVal = await defaultRes.text();
      } else {
        // Fallback if the endpoint fails: note that it was the default
        systemPromptVal = "(server default \u2013 see system_prompt_v01.txt)";
      }
    } catch {
      systemPromptVal = "(server default \u2013 see system_prompt_v01.txt)";
    }
  }

  // ── Gather all session state into the request body ─────────────────────
  const model       = getModelName();
  const temperature = parseFloat(tempInput.value);
  const max_tokens  = parseInt(tokensInput.value, 10);

  const reqBody = {
    payload:       state.payload,
    output:        state.current,     // null if not generated yet
    baseline:      state.baseline,    // null if no baseline set
    model,
    temperature,
    max_tokens,
    system_prompt: systemPromptVal,
  };

  // ── POST to /api/save ──────────────────────────────────────────────────
  btnSave.disabled = true;
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

    // Show the folder name and file list so the user knows where to find
    // the saved files on disk.
    setStatus(`Saved \u2192 data/${data.folder_name}/ (${data.files.join(", ")})`);
  } catch (err) {
    setStatus(`Save error: ${err.message}`);
  } finally {
    btnSave.disabled = false;
    spinner.classList.add("hidden");
  }
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

  // ── Load prompt button ──────────────────────────────────────────────── //
  // Mirrors the example-loading pattern: click loads, dblclick is a
  // convenience shortcut.  The loaded text populates the system prompt
  // textarea and opens the System Prompt <details> collapsible.
  btnLoadPrompt.addEventListener("click", () => {
    const name = promptSelect.value;
    if (name) loadPrompt(name);
  });

  // Double-click the prompt dropdown also loads (convenience, mirrors examples)
  promptSelect.addEventListener("dblclick", () => {
    const name = promptSelect.value;
    if (name) loadPrompt(name);
  });

  // ── System prompt textarea → update override badge ──────────────────── //
  // When the user types in the system prompt textarea (or clears it), the
  // badge switches between muted (server default) and active (custom override).
  systemPromptTextarea.addEventListener("input", () => {
    updateSystemPromptBadge();
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

  // ── Save session state to data/ folder ──────────────────────────────── //
  btnSave.addEventListener("click", () => {
    saveRun();
  });

  // ── Clear output ─────────────────────────────────────────────────────── //
  btnClearOutput.addEventListener("click", () => {
    outputBox.textContent = "";
    outputBox.appendChild(makePlaceholder("Click Generate to produce a description."));
    outputMeta.textContent = "";
    outputMeta.classList.add("hidden");
    state.current        = null;
    state.baseline       = null;
    diffA.textContent = "";
    diffA.appendChild(makePlaceholder("No baseline set."));
    diffB.textContent = "";
    diffB.appendChild(makePlaceholder("Generate to populate B."));
    diffDelta.textContent = "";
    diffDelta.appendChild(makePlaceholder("\u2014"));
    signalPanel.textContent = "";
    signalPanel.appendChild(
      makePlaceholder("Set a baseline (A) and generate to analyze content-word changes.")
    );
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

/* ═══════════════════════════════════════════════════════════════════════════
   TOOLTIP SYSTEM
   ─────────────────────────────────────────────────────────────────────────
   Controls the global visibility and positioning of hover help tooltips.

   Why JS-positioned tooltips?
   ───────────────────────────
   The three main panels use `overflow-y: auto` for scrolling, which creates
   a clipping boundary.  CSS pseudo-element tooltips (::before / ::after)
   are children of their trigger element in the rendering tree, so they get
   clipped by the panel's overflow.  By creating real DOM nodes appended to
   <body>, the tooltip bubble and arrow sit outside all overflow containers
   and are never clipped.

   Architecture
   ────────────
   1. Tooltip text lives in `data-tooltip` attributes on trigger elements.
   2. On mouseenter, if tooltips are enabled (data-tooltips="on" on <html>),
      JS creates two elements: `.tooltip-bubble` (text) and `.tooltip-arrow`
      (triangle), both appended to <body> with `position: fixed`.
   3. Positioning uses getBoundingClientRect() for viewport-relative coords:
      - Default: below the element, horizontally centred on it.
      - If below would overflow the viewport bottom → flip above.
      - If the bubble extends past the left edge → clamp left to 8px.
      - If the bubble extends past the right edge → clamp right to 8px.
      - The arrow always points at the horizontal centre of the trigger.
   4. On mouseleave the elements are removed from the DOM.

   Toggle state
   ────────────
   Gated by `data-tooltips="on"|"off"` on <html>.  Persisted in
   localStorage under key "padl-tooltips".

   The toggle button uses the same "is-active" amber glow pattern as
   "Set as A" for visual consistency.
════════════════════════════════════════════════════════════════════════════ */

/**
 * Currently visible tooltip elements, or null if no tooltip is showing.
 * Stored so mouseleave can clean them up, and so we never create duplicates.
 *
 * @type {{ bubble: HTMLDivElement, arrow: HTMLDivElement } | null}
 */
let activeTooltip = null;

/**
 * Show a tooltip for the given trigger element.
 *
 * Creates two fixed-position DOM nodes (bubble + arrow) appended to <body>,
 * then positions them relative to the trigger's bounding rect with
 * viewport-edge clamping.
 *
 * @param {HTMLElement} trigger - The element with a `data-tooltip` attribute.
 */
function showTooltip(trigger) {
  // Remove any existing tooltip first (defensive; shouldn't happen normally)
  hideTooltip();

  const text = trigger.getAttribute("data-tooltip");
  if (!text) return;

  // ── Create the bubble (text container) ───────────────────────────────── //
  const bubble = document.createElement("div");
  bubble.className = "tooltip-bubble";
  bubble.textContent = text;

  // ── Create the arrow (triangle pointer) ──────────────────────────────── //
  const arrow = document.createElement("div");
  arrow.className = "tooltip-arrow";

  // Append to <body> so they are outside any overflow-clipping ancestor
  document.body.appendChild(bubble);
  document.body.appendChild(arrow);

  // Store references for cleanup on mouseleave
  activeTooltip = { bubble, arrow };

  // ── Measure trigger and bubble for positioning ───────────────────────── //
  const triggerRect = trigger.getBoundingClientRect();
  const bubbleRect  = bubble.getBoundingClientRect();

  // Viewport dimensions (used for edge clamping)
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // Gap between trigger edge and arrow tip (pixels)
  const GAP = 6;

  // Arrow size must match the CSS `border: 6px` on .tooltip-arrow
  const ARROW_SIZE = 6;

  // Margin from viewport edges (minimum padding)
  const EDGE_MARGIN = 8;

  // ── Vertical placement: prefer below, flip above if no room ──────────── //
  //
  // "Below" means the arrow's tip sits GAP pixels beneath the trigger's
  // bottom edge, and the bubble sits directly beneath the arrow.
  //
  // "Above" means the bubble's bottom edge sits GAP pixels above the
  // trigger's top edge, and the arrow points downward from the bubble.

  const spaceBelow = vh - triggerRect.bottom;  // px available below trigger
  const spaceAbove = triggerRect.top;           // px available above trigger
  const totalNeeded = GAP + ARROW_SIZE + bubbleRect.height;

  // Choose placement: below if there's room, otherwise above
  const placeAbove = spaceBelow < totalNeeded && spaceAbove > spaceBelow;

  let bubbleTop;
  let arrowTop;

  if (placeAbove) {
    // ── Place ABOVE the trigger ──────────────────────────────────────── //
    // Bubble bottom edge sits GAP + ARROW_SIZE above trigger top
    bubbleTop = triggerRect.top - GAP - ARROW_SIZE - bubbleRect.height;

    // Arrow points DOWN from bottom of bubble toward the trigger
    arrowTop = triggerRect.top - GAP - ARROW_SIZE;

    // Arrow triangle: border-top-color creates a downward-pointing triangle
    arrow.style.borderTopColor = "var(--tooltip-border)";
  } else {
    // ── Place BELOW the trigger (default) ────────────────────────────── //
    // Arrow tip sits GAP below trigger bottom
    arrowTop = triggerRect.bottom + GAP;

    // Bubble top sits just below the arrow
    bubbleTop = arrowTop + ARROW_SIZE;

    // Arrow triangle: border-bottom-color creates an upward-pointing triangle
    arrow.style.borderBottomColor = "var(--tooltip-border)";
  }

  // ── Horizontal placement: centre on trigger, clamp to viewport ───────── //
  //
  // Start by centring the bubble horizontally on the trigger element.
  // Then clamp so it doesn't extend past either viewport edge.

  const triggerCentreX = triggerRect.left + triggerRect.width / 2;
  let bubbleLeft = triggerCentreX - bubbleRect.width / 2;

  // Clamp left edge: ensure at least EDGE_MARGIN from viewport left
  if (bubbleLeft < EDGE_MARGIN) {
    bubbleLeft = EDGE_MARGIN;
  }

  // Clamp right edge: ensure at least EDGE_MARGIN from viewport right
  if (bubbleLeft + bubbleRect.width > vw - EDGE_MARGIN) {
    bubbleLeft = vw - EDGE_MARGIN - bubbleRect.width;
  }

  // Arrow horizontal position: always point at the trigger's centre,
  // offset by half the arrow width so the triangle tip is centred
  const arrowLeft = triggerCentreX - ARROW_SIZE;

  // ── Apply computed positions ─────────────────────────────────────────── //
  bubble.style.top  = `${bubbleTop}px`;
  bubble.style.left = `${bubbleLeft}px`;

  arrow.style.top  = `${arrowTop}px`;
  arrow.style.left = `${arrowLeft}px`;
}

/**
 * Remove the currently visible tooltip from the DOM.
 * Safe to call even if no tooltip is showing (no-op).
 */
function hideTooltip() {
  if (!activeTooltip) return;

  // Remove both elements from <body>
  activeTooltip.bubble.remove();
  activeTooltip.arrow.remove();
  activeTooltip = null;
}

/**
 * Wire up tooltip show/hide listeners on all `[data-tooltip]` elements.
 *
 * Uses event delegation on `document.body` via mouseenter/mouseleave
 * (with `capture: true` for mouseenter since it doesn't bubble).
 * This approach automatically handles dynamically-added elements
 * (e.g. sliders rebuilt after a JSON edit) without re-wiring.
 *
 * The listeners check `data-tooltips="on"` on <html> before showing,
 * so no tooltip appears when the toggle is off.
 */
function wireTooltipListeners() {
  /**
   * mouseenter handler (capture phase).
   *
   * Walks up from the event target to find the nearest ancestor (or self)
   * with a `data-tooltip` attribute.  This handles cases where the mouse
   * enters a child element inside a tooltip-bearing parent (e.g. the text
   * inside a <button>).
   */
  document.body.addEventListener("mouseenter", (e) => {
    // Only show tooltips when the global toggle is ON
    if (document.documentElement.getAttribute("data-tooltips") !== "on") return;

    // Find the nearest tooltip-bearing ancestor (or the target itself)
    const trigger = e.target.closest("[data-tooltip]");
    if (!trigger) return;

    showTooltip(trigger);
  }, true);  // capture: true — mouseenter doesn't bubble, so we capture

  /**
   * mouseleave handler (capture phase).
   *
   * Hides the tooltip when the mouse leaves a tooltip-bearing element.
   * Uses the same closest() walk as mouseenter for symmetry.
   */
  document.body.addEventListener("mouseleave", (e) => {
    const trigger = e.target.closest("[data-tooltip]");
    if (!trigger) return;

    hideTooltip();
  }, true);  // capture: true — mouseleave doesn't bubble
}

/**
 * Initialise the tooltip toggle button and restore any saved preference.
 *
 * The toggle sets `data-tooltips="on"|"off"` on the <html> element.
 * JS mouseenter/mouseleave listeners check this attribute before
 * creating tooltip DOM elements.
 *
 * The button's `is-active` class mirrors the pattern used by the
 * "Set as A" button (.btn--secondary.is-active), providing a consistent
 * visual language for active/inactive toggle states.
 */
function wireTooltipToggle() {
  const btn = document.getElementById("tooltip-toggle");
  if (!btn) return;

  /**
   * Apply the tooltip state to the DOM and persist to localStorage.
   *
   * @param {string} tooltipState - Either "on" or "off".
   */
  const applyTooltips = (tooltipState) => {
    // Set the gating attribute checked by tooltip event listeners
    document.documentElement.setAttribute("data-tooltips", tooltipState);

    // Update the button text to reflect the current state
    btn.textContent = tooltipState === "on" ? "? Tips \u2713" : "? Tips";

    // Apply the is-active class for the amber glow visual (same as "Set as A")
    btn.classList.toggle("is-active", tooltipState === "on");

    // Persist the preference so it survives page reloads
    localStorage.setItem("padl-tooltips", tooltipState);

    // If turning off, immediately hide any visible tooltip
    if (tooltipState === "off") {
      hideTooltip();
    }
  };

  // Restore saved preference on page load, default to "off"
  const saved = localStorage.getItem("padl-tooltips") || "off";
  applyTooltips(saved);

  // Toggle between "on" and "off" on each click
  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-tooltips") || "off";
    applyTooltips(current === "on" ? "off" : "on");
  });

  // Wire the delegated mouseenter/mouseleave listeners (runs once)
  wireTooltipListeners();
}

async function init() {
  wireThemeToggle();
  wireTooltipToggle();
  wireEvents();
  await loadExampleList();
  await loadPromptList();

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

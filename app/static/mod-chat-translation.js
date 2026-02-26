/**
 * mod-chat-translation.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Chat Translation page — all state, slider construction, and API interaction.
 *
 * Architecture
 * ────────────
 * This module is self-contained: it maintains its own `chatState` object for
 * Characters A and B and does not share mutable state with the Character
 * Description page (`mod-state.js` / `state`).  DOM references *are* shared
 * via `mod-state.js#dom` because the elements are already queried at load
 * time; only the per-page logic is isolated here.
 *
 * Features
 * ────────
 *   • Example loading  — populates both character dropdowns from /api/examples
 *   • Slider panel     — builds axis rows with per-axis enable checkboxes,
 *                        score sliders, and editable label inputs; all slider
 *                        and label changes are written back to chatState and
 *                        sync'd to the JSON textarea in real time.
 *   • Relabel          — POST /api/relabel per character; preserves activeAxes
 *   • Randomise        — cryptoRandomFloat per axis score; optional auto-relabel
 *   • IC prompt loader — filters /api/prompts to names starting with "ic_"
 *   • Translate        — POST /api/translate_chat → renders IC text + IPC meta
 *   • Model refresh    — pulls model list from /api/models at the chat Ollama host
 *   • Live mode        — toggle that reveals per-character Send buttons and the
 *                        in-game output panel; batch Translate is disabled while
 *                        live mode is active.
 *   • In-game log      — appendGameEntry() builds MUD-style entries; chatState.gameLog
 *                        tracks the full history so clipboard and save functions can
 *                        access it without scraping the DOM.  Each entry stores the
 *                        original OOC message alongside the translated IC text.
 *   • OOC toggle       — header button that shows/hides the OOC column in the log
 *                        panel via .chat-game-output--hide-ooc + CSS display:none.
 *                        is-active (amber) = OOC shown; inactive = hidden.
 *   • Copy TXT         — formats log as ``CH | OOC | [channel]: IC text`` lines.
 *   • Copy MD          — formats log as a 5-column Markdown table; pipe chars are escaped.
 *   • Save all data    — POSTs to /api/save_chat then GETs /api/save/{folder}/export
 *                        to write files server-side and trigger a browser zip download
 *                        in one click.
 *
 * Data flow
 * ─────────
 *   1. User loads an example → chatState[ch].payload populated
 *   2. Sliders/checkboxes mutate chatState[ch].payload in real time
 *   3. Translate: buildAxesForRequest() filters by activeAxes, sends POST
 *   4. Result rendered into output boxes with IPC meta table
 *
 * Character key
 * ─────────────
 * Throughout this module the parameter `ch` is always either the string
 * `"a"` (Character A) or `"b"` (Character B).  The `charDom(ch)` helper
 * maps it to the relevant set of DOM refs.
 *
 * Imports: mod-state (dom only), mod-utils, mod-status
 */

import { dom } from "./mod-state.js";
import { clamp, debounce, safeParse, cryptoRandomFloat } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";

// ─────────────────────────────────────────────────────────────────────────────
// Internal state (chat page only)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-character mutable state for the Chat Translation page.
 *
 * Kept separate from `mod-state.js#state` so that switching pages does not
 * accidentally corrupt the Character Description page's state.
 *
 * @type {{
 *   a: {
 *     payload: Object|null,
 *     originalAxes: Object|null,
 *     activeAxes: Set<string>|null
 *   },
 *   b: {
 *     payload: Object|null,
 *     originalAxes: Object|null,
 *     activeAxes: Set<string>|null
 *   },
 *   busy:     boolean,
 *   liveMode: boolean,
 *   logSeq:   number,
 *   gameLog:  Array<{ch: string, channel: string, oocMessage: string, icText: string, model: string, ipcId: string|null, inputHash: string|null, systemPromptHash: string|null, outputHash: string|null}>
 * }}
 *
 * @property {Object|null}      [a|b].payload      - Parsed AxisPayload for the character,
 *                                                    or null before an example is loaded.
 * @property {Object|null}      [a|b].originalAxes - Deep-copy of axes at load time, used
 *                                                    to highlight modified sliders.
 * @property {Set<string>|null} [a|b].activeAxes   - Axis names currently enabled via the
 *                                                    checkbox column.  null = all enabled
 *                                                    (resolved lazily on first slider build).
 * @property {boolean}          busy               - True while a /api/translate_chat request
 *                                                    is in-flight; prevents double-submission.
 * @property {boolean}          liveMode           - True when the Live toggle is checked;
 *                                                    controls visibility of Send buttons and
 *                                                    the in-game output section.
 * @property {number}           logSeq             - Monotonically increasing counter, incremented
 *                                                    by appendGameEntry() on each new entry.
 *                                                    Currently used for debugging; reserved for
 *                                                    future row-numbering UI.
 * @property {Array}            gameLog            - Ordered list of in-game log entries.  Each
 *                                                    entry is ``{ ch, channel, oocMessage,
 *                                                    icText, model, ipcId, inputHash,
 *                                                    systemPromptHash, outputHash }``.
 *                                                    All four IPC hashes are stored so that
 *                                                    clipboard copy and save include the full
 *                                                    provenance record matching the in-browser
 *                                                    IPC meta table.  Populated by
 *                                                    appendGameEntry(), cleared by the Clear
 *                                                    button handler.
 */
const chatState = {
  a: { payload: null, originalAxes: null, activeAxes: null },
  b: { payload: null, originalAxes: null, activeAxes: null },
  busy: false,
  liveMode: false,
  logSeq: 0,
  /**
   * @type {{
   *   ch: string,
   *   channel: string,
   *   oocMessage: string,
   *   icText: string,
   *   model: string,
   *   ipcId: string|null,
   *   inputHash: string|null,
   *   systemPromptHash: string|null,
   *   outputHash: string|null
   * }[]}
   */
  gameLog: [],
};

// ─────────────────────────────────────────────────────────────────────────────
// DOM helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Return the DOM ref bundle for character `ch`.
 *
 * Abstracts away the `chatA*` / `chatB*` naming difference so that all
 * character-agnostic functions can operate on a single object regardless
 * of which character they are processing.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {{
 *   exampleSelect:  HTMLSelectElement,
 *   btnLoadExample: HTMLButtonElement,
 *   jsonTextarea:   HTMLTextAreaElement,
 *   jsonBadge:      HTMLElement,
 *   sliderPanel:    HTMLElement,
 *   btnRandomise:   HTMLButtonElement,
 *   autoLabel:      HTMLInputElement,
 *   btnRelabel:     HTMLButtonElement,
 *   oocTextarea:    HTMLTextAreaElement,
 *   channelSelect:  HTMLSelectElement,
 *   btnSend:        HTMLButtonElement
 * }} The DOM refs for the requested character panel.
 */
function charDom(ch) {
  return ch === "a"
    ? {
        exampleSelect:  dom.chatAExampleSelect,
        btnLoadExample: dom.chatABtnLoadExample,
        jsonTextarea:   dom.chatAJson,
        jsonBadge:      dom.chatAJsonBadge,
        sliderPanel:    dom.chatASliderPanel,
        btnRandomise:   dom.chatABtnRandomise,
        autoLabel:      dom.chatAAutoLabel,
        btnRelabel:     dom.chatABtnRelabel,
        oocTextarea:    dom.chatAOoc,
        channelSelect:  dom.chatAChannel,
        btnSend:        dom.chatABtnSend,
      }
    : {
        exampleSelect:  dom.chatBExampleSelect,
        btnLoadExample: dom.chatBBtnLoadExample,
        jsonTextarea:   dom.chatBJson,
        jsonBadge:      dom.chatBJsonBadge,
        sliderPanel:    dom.chatBSliderPanel,
        btnRandomise:   dom.chatBBtnRandomise,
        autoLabel:      dom.chatBAutoLabel,
        btnRelabel:     dom.chatBBtnRelabel,
        oocTextarea:    dom.chatBOoc,
        channelSelect:  dom.chatBChannel,
        btnSend:        dom.chatBBtnSend,
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON textarea sync
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Serialise `chatState[ch].payload` back into the JSON textarea.
 *
 * Called whenever the payload is mutated programmatically (slider drag,
 * label edit, relabel, randomise) so that the textarea always reflects
 * the current in-memory state.  If payload is null, the textarea is left
 * unchanged to avoid overwriting the user's in-progress manual edits.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {void}
 */
function syncJsonTextarea(ch) {
  const cd = charDom(ch);
  const payload = chatState[ch].payload;
  if (payload) cd.jsonTextarea.value = JSON.stringify(payload, null, 2);
}

/**
 * Update the JSON badge to indicate parse status.
 *
 * Shown as a small inline badge next to the "Axis JSON" collapsible header.
 * - Valid parse   → green "OK" badge  (.badge)
 * - Invalid parse → red "ERR" badge   (.badge.badge--err)
 *
 * @param {"a"|"b"} ch    - Character identifier.
 * @param {boolean}  valid - True if the textarea's JSON is currently valid.
 * @returns {void}
 */
function setJsonBadge(ch, valid) {
  const cd = charDom(ch);
  if (valid) {
    cd.jsonBadge.textContent = "OK";
    cd.jsonBadge.className = "badge";
  } else {
    cd.jsonBadge.textContent = "ERR";
    cd.jsonBadge.className = "badge badge--err";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Slider panel builder
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Rebuild the axis slider panel for character `ch` from `chatState[ch].payload`.
 *
 * Clears the slider panel DOM and constructs one axis row per entry in
 * `payload.axes`.  Each row contains four columns:
 *
 *   1. **Enable checkbox** — controls whether the axis is included in the
 *      profile sent to the backend (`chatState[ch].activeAxes`).
 *   2. **Axis name** — non-interactive label.
 *   3. **Score slider + display** — range 0–1, step 0.005.  Score display
 *      gains the `axis-modified` class when the value differs from the
 *      original loaded value.
 *   4. **Label input** — editable text field.  Disabled while auto-label
 *      is active.  Gains `axis-modified` when the text differs from the
 *      original.
 *
 * State persistence
 * -----------------
 * `chatState[ch].activeAxes` is preserved across rebuilds (relabel,
 * randomise) so that the user's checkbox selections survive a recompute.
 * It is only reset to all-enabled when a new example is loaded via
 * `loadChatExample()`.
 *
 * All slider and label `input` events write through to
 * `chatState[ch].payload.axes` and call `syncJsonTextarea()`.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {void}
 */
function buildChatSliders(ch) {
  const cd = charDom(ch);
  const payload = chatState[ch].payload;
  const panel = cd.sliderPanel;

  if (!payload || typeof payload.axes !== "object" || payload.axes === null) {
    panel.textContent = "";
    const p = document.createElement("p");
    p.className = "placeholder-text";
    p.textContent = "No axes found in payload.";
    panel.appendChild(p);
    return;
  }

  const axes = payload.axes;
  const keys = Object.keys(axes);

  if (keys.length === 0) {
    panel.textContent = "";
    const p = document.createElement("p");
    p.className = "placeholder-text";
    p.textContent = "axes object is empty.";
    panel.appendChild(p);
    return;
  }

  // Initialise activeAxes to all keys on first load; preserve on rebuilds.
  if (chatState[ch].activeAxes === null) {
    chatState[ch].activeAxes = new Set(keys);
  }

  const fragment = document.createDocumentFragment();

  for (const axisKey of keys) {
    const axisVal = axes[axisKey];
    const score = clamp(parseFloat(axisVal.score) || 0, 0, 1);
    const label = String(axisVal.label || "");
    const isActive = chatState[ch].activeAxes.has(axisKey);
    const orig = chatState[ch].originalAxes && chatState[ch].originalAxes[axisKey];

    const row = document.createElement("div");
    row.className = "axis-row chat-axis-row";
    row.dataset.axis = axisKey;

    // ── Column 1: Enable checkbox ──────────────────────────────────────── //
    // Toggling the checkbox adds/removes the axis key from activeAxes and
    // visually dims the row via the axis-row--disabled class.
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "axis-enable-checkbox";
    checkbox.checked = isActive;
    checkbox.title = "Include this axis in the character profile";
    checkbox.setAttribute("aria-label", `Enable ${axisKey} in profile`);

    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        chatState[ch].activeAxes.add(axisKey);
      } else {
        chatState[ch].activeAxes.delete(axisKey);
      }
      row.classList.toggle("axis-row--disabled", !checkbox.checked);
    });

    row.classList.toggle("axis-row--disabled", !isActive);

    // ── Column 2: Axis name label ──────────────────────────────────────── //
    const nameEl = document.createElement("span");
    nameEl.className = "axis-name";
    nameEl.textContent = axisKey;
    nameEl.title = axisKey;

    // ── Row 1, col 3: Score display ────────────────────────────────────── //
    // Placed as a direct grid child (col 3, row 1) so the slider below can
    // occupy the full 1fr column without sharing it with the score span.
    const scoreDisplay = document.createElement("span");
    scoreDisplay.className = "axis-score";
    scoreDisplay.textContent = score.toFixed(3);

    // Mark score as modified if it differs from the originally loaded value.
    if (orig && Math.abs(score - orig.score) > 0.0001) {
      scoreDisplay.classList.add("axis-modified");
    }

    // ── Row 2, col 2: Range slider ─────────────────────────────────────── //
    // Direct grid child (no wrapper div) so CSS can place it at row 2, col 2
    // and give it the full 1fr width (~210px on a 384px panel).
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "range-input";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.005";
    slider.value = score.toFixed(3);
    slider.setAttribute("aria-label", `${axisKey} score`);

    // ── Row 2, col 3: Label input ──────────────────────────────────────── //
    const labelInput = document.createElement("input");
    labelInput.type = "text";
    labelInput.className = "axis-label-input";
    labelInput.value = label;
    labelInput.setAttribute("aria-label", `${axisKey} label`);
    // Disable manual editing when auto-label is on (server computes labels).
    labelInput.disabled = cd.autoLabel.checked;

    if (orig && label !== orig.label) {
      labelInput.classList.add("axis-modified");
    }

    // ── Per-axis event listeners ───────────────────────────────────────── //
    // Slider drag → update in-memory payload + badge + sync textarea.
    slider.addEventListener("input", () => {
      const newScore = parseFloat(slider.value);
      scoreDisplay.textContent = newScore.toFixed(3);
      const origAxis = chatState[ch].originalAxes && chatState[ch].originalAxes[axisKey];
      if (origAxis) {
        scoreDisplay.classList.toggle("axis-modified", Math.abs(newScore - origAxis.score) > 0.0001);
      }
      chatState[ch].payload.axes[axisKey] = { ...chatState[ch].payload.axes[axisKey], score: newScore };
      syncJsonTextarea(ch);
    });

    // Label edit → update in-memory payload + badge + sync textarea.
    labelInput.addEventListener("input", () => {
      const origAxis = chatState[ch].originalAxes && chatState[ch].originalAxes[axisKey];
      if (origAxis) {
        labelInput.classList.toggle("axis-modified", labelInput.value !== origAxis.label);
      }
      chatState[ch].payload.axes[axisKey] = { ...chatState[ch].payload.axes[axisKey], label: labelInput.value };
      syncJsonTextarea(ch);
    });

    // Grid children order: checkbox (r1-2/c1), nameEl (r1/c2), scoreDisplay (r1/c3),
    // slider (r2/c2), labelInput (r2/c3). Explicit CSS grid-column/row on each class
    // handles placement regardless of DOM order, but keep it logical here.
    row.appendChild(checkbox);
    row.appendChild(nameEl);
    row.appendChild(scoreDisplay);
    row.appendChild(slider);
    row.appendChild(labelInput);
    fragment.appendChild(row);
  }

  panel.textContent = "";
  panel.appendChild(fragment);
}

// ─────────────────────────────────────────────────────────────────────────────
// Example loading
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch and load an example payload for character `ch`.
 *
 * GETs `/api/examples/{name}`, populates `chatState[ch].payload`,
 * deep-copies the axes into `originalAxes` for modification tracking,
 * resets `activeAxes` to null (all-enabled) for the fresh example, and
 * rebuilds the slider panel.
 *
 * @param {"a"|"b"} ch   - Character identifier.
 * @param {string}  name - Example name as returned by `/api/examples`.
 * @returns {Promise<void>}
 */
async function loadChatExample(ch, name) {
  if (!name) return;
  try {
    const res = await fetch(`/api/examples/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    chatState[ch].payload = data;
    // Deep-copy the axes at load time so we can compare later for modification
    // highlighting without the original being mutated by slider changes.
    chatState[ch].originalAxes = JSON.parse(JSON.stringify(data.axes || {}));
    // Reset activeAxes so all axes are enabled for a fresh example.
    chatState[ch].activeAxes = null;
    syncJsonTextarea(ch);
    setJsonBadge(ch, true);
    buildChatSliders(ch);
    setStatus(`Character ${ch.toUpperCase()} — loaded "${name}".`);
  } catch (err) {
    setStatus(`Error loading example for ${ch.toUpperCase()}: ${err.message}`);
  }
}

/**
 * Fetch the list of available example names and populate both character
 * example `<select>` dropdowns.
 *
 * GETs `/api/examples` and appends one `<option>` per name to each
 * character's dropdown.  The placeholder "— choose —" option (always
 * `options[0]`) is preserved; any previously loaded options are removed
 * before adding new ones.
 *
 * Silently ignores network errors — the server may not be running at
 * init time (e.g. during dev with Ollama down).
 *
 * @returns {Promise<void>}
 */
async function loadChatExampleList() {
  try {
    const res = await fetch("/api/examples");
    if (!res.ok) return;
    const names = await res.json();
    for (const ch of ["a", "b"]) {
      const cd = charDom(ch);
      const sel = cd.exampleSelect;
      // Keep the placeholder option, remove any previously loaded options.
      while (sel.options.length > 1) sel.remove(1);
      for (const name of names) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      }
    }
  } catch {
    // Silently ignore — Ollama/server may not be running at init
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Relabel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * POST the character's current payload to `/api/relabel` and update state.
 *
 * The relabel endpoint applies the server-side `RELABEL_POLICY` table to
 * map current axis scores to canonical labels.  On success the payload is
 * replaced with the server's response and the slider panel is rebuilt.
 * `activeAxes` is preserved across the rebuild so that the user's checkbox
 * selections survive a recompute.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Promise<void>}
 */
async function relabelChatChar(ch) {
  if (!chatState[ch].payload) {
    setStatus(`No payload to relabel for Character ${ch.toUpperCase()}.`);
    return;
  }
  setStatus(`Recomputing labels for Character ${ch.toUpperCase()}…`, true);
  try {
    const res = await fetch("/api/relabel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatState[ch].payload),
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(e.detail || `HTTP ${res.status}`);
    }
    chatState[ch].payload = await res.json();
    syncJsonTextarea(ch);
    buildChatSliders(ch);
    setStatus(`Character ${ch.toUpperCase()} — labels recomputed.`);
  } catch (err) {
    setStatus(`Relabel error (${ch.toUpperCase()}): ${err.message}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Randomise
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Assign cryptographically random scores (0–1) to all axes for character `ch`.
 *
 * Uses `cryptoRandomFloat()` (backed by `crypto.getRandomValues`) for
 * uniform distribution.  Scores are rounded to 3 decimal places to match
 * the slider step size.
 *
 * After randomising:
 * - If "Auto labels" is checked, triggers a relabel immediately so labels
 *   stay consistent with the new scores.
 * - Otherwise, rebuilds the slider panel with the new scores and updates
 *   the status bar.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Promise<void>}
 */
async function randomiseChatChar(ch) {
  const payload = chatState[ch].payload;
  if (!payload || !payload.axes) {
    setStatus(`No payload to randomise for Character ${ch.toUpperCase()}.`);
    return;
  }
  for (const axisKey of Object.keys(payload.axes)) {
    const newScore = Math.round(cryptoRandomFloat() * 1000) / 1000;
    payload.axes[axisKey] = { ...payload.axes[axisKey], score: newScore };
  }
  syncJsonTextarea(ch);
  buildChatSliders(ch);
  if (charDom(ch).autoLabel.checked) {
    await relabelChatChar(ch);
  } else {
    setStatus(`Character ${ch.toUpperCase()} — scores randomised.`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// IC Prompt loader
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch available IC prompt names and populate the IC prompt `<select>`.
 *
 * GETs `/api/prompts` (all prompt names) and filters to names that start
 * with `"ic_"`, which is the naming convention for in-character translation
 * prompts stored in `app/prompts/`.  The `"— default —"` placeholder option
 * is preserved; previously loaded options are removed first.
 *
 * Silently ignores network errors to avoid blocking startup.
 *
 * @returns {Promise<void>}
 */
async function loadChatIcPromptList() {
  try {
    const res = await fetch("/api/prompts");
    if (!res.ok) return;
    const names = await res.json();
    // Only show IC prompts (naming convention: prefix "ic_").
    const icNames = names.filter((n) => n.startsWith("ic_"));
    const sel = dom.chatPromptSelect;
    while (sel.options.length > 1) sel.remove(1);
    for (const name of icNames) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
  } catch {
    // Silently ignore
  }
}

/**
 * Update the IC prompt badge to reflect whether a custom prompt is loaded.
 *
 * - Empty textarea → "default" badge (muted style): server will use its
 *   built-in default IC prompt.
 * - Non-empty textarea → "custom" badge (active style): the inline text
 *   will be sent as `system_prompt` in the translate request.
 *
 * @returns {void}
 */
function updateChatPromptBadge() {
  const hasContent = dom.chatSystemPrompt.value.trim().length > 0;
  dom.chatPromptBadge.textContent = hasContent ? "custom" : "default";
  dom.chatPromptBadge.className = hasContent ? "badge badge--active" : "badge badge--muted";
}

// ─────────────────────────────────────────────────────────────────────────────
// Model helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Return the currently selected/entered model name.
 *
 * Prefers the `<select>` value (shown when Ollama is reachable and has
 * returned a model list); falls back to the free-text `<input>` value
 * (shown when Ollama is unreachable or has no models).
 *
 * @returns {string} The model tag, e.g. `"gemma2:2b"`, or `""` if empty.
 */
function getChatModelName() {
  const sel = dom.chatModelSelect.value.trim();
  return sel || dom.chatModelInput.value.trim();
}

/**
 * Return the current Ollama host URL from the chat settings field.
 *
 * Returns an empty string if the field is blank (the server will fall
 * back to the `OLLAMA_HOST` environment variable in that case).
 *
 * @returns {string} The host URL, e.g. `"http://localhost:11434"`, or `""`.
 */
function getChatOllamaHost() {
  return dom.chatOllamaHost.value.trim();
}

/**
 * Resolve the seed value from the chat seed input.
 *
 * The UI convention (mirroring the Character Description page) is:
 * - Positive integer → use as-is.
 * - `-1`, blank, or invalid → generate a random 32-bit unsigned integer.
 *
 * @returns {number} A non-negative integer to send as `seed` in the request.
 */
function resolveChatSeed() {
  const raw = parseInt(dom.chatSeedInput.value, 10);
  if (isNaN(raw) || raw < 0) {
    // Generate a random 32-bit unsigned int (matches the main page behaviour).
    return Math.floor(Math.random() * 0x100000000);
  }
  return raw;
}

/**
 * Fetch the available Ollama model list and populate the model `<select>`.
 *
 * GETs `/api/models?host=<host>` (or `/api/models` if host is empty).  On
 * success, replaces the `<select>` options and re-selects the previously
 * chosen model if it still appears in the list.  On failure (Ollama
 * unreachable), hides the `<select>` and shows the free-text `<input>`.
 *
 * @param {string} [host] - Ollama host override.  If omitted, uses the
 *                           current value of the Ollama host field.
 * @returns {Promise<void>}
 */
async function refreshChatModels(host) {
  const h = host || getChatOllamaHost();
  const url = h ? `/api/models?host=${encodeURIComponent(h)}` : "/api/models";
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const models = await res.json();
    const prev = getChatModelName();
    if (models.length > 0) {
      dom.chatModelSelect.innerHTML = "";
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === prev) opt.selected = true;
        dom.chatModelSelect.appendChild(opt);
      }
      dom.chatModelSelect.classList.remove("hidden");
      dom.chatModelInput.classList.add("hidden");
    } else {
      dom.chatModelSelect.classList.add("hidden");
      dom.chatModelInput.classList.remove("hidden");
    }
  } catch {
    dom.chatModelSelect.classList.add("hidden");
    dom.chatModelInput.classList.remove("hidden");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Translate helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the axes dict for the translate API request from the character's
 * current payload, keeping only axes that are enabled in the slider panel.
 *
 * Disabled axes (unchecked in the slider panel) are intentionally excluded
 * so that the backend renders the profile summary without those axes.  This
 * lets the user test how the LLM responds when, say, the `wealth` axis is
 * omitted from the character profile.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Object|null} A `{ axisName: { label, score } }` dict containing
 *                         only the active axes, or `null` if no payload is
 *                         loaded.
 */
function buildAxesForRequest(ch) {
  const payload = chatState[ch].payload;
  if (!payload || !payload.axes) return null;
  const active = chatState[ch].activeAxes;
  const axes = {};
  for (const [k, v] of Object.entries(payload.axes)) {
    // When activeAxes is null (no example loaded), include everything.
    if (active === null || active.has(k)) {
      axes[k] = v;
    }
  }
  return axes;
}

/**
 * Build a compact IPC meta table element from a `ChatTranslationResult`.
 *
 * Renders four rows (input, prompt, output, ipc) showing the first 16
 * characters of each hash followed by an ellipsis.  The table uses the
 * existing `.meta-table` / `.meta-key` / `.meta-val` styles from the
 * Character Description page.
 *
 * Rows for hashes that are `null` or `undefined` (e.g. `output_hash` and
 * `ipc_id` on a failed translation) are omitted entirely.
 *
 * @param {Object} result - `ChatTranslationResult` object from the API,
 *                           containing `input_hash`, `system_prompt_hash`,
 *                           `output_hash`, and `ipc_id`.
 * @returns {HTMLTableElement} A populated `<table class="meta-table">`.
 */
function buildIpcMetaTable(result) {
  const rows = [];
  if (result.input_hash)         rows.push(["input",  result.input_hash.slice(0, 16) + "\u2026"]);
  if (result.system_prompt_hash) rows.push(["prompt", result.system_prompt_hash.slice(0, 16) + "\u2026"]);
  if (result.output_hash)        rows.push(["output", result.output_hash.slice(0, 16) + "\u2026"]);
  if (result.ipc_id)             rows.push(["ipc",    result.ipc_id.slice(0, 16) + "\u2026"]);

  const table = document.createElement("table");
  table.className = "meta-table";
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    const tdKey = document.createElement("td");
    tdKey.className = "meta-key";
    tdKey.textContent = k;
    const tdVal = document.createElement("td");
    tdVal.className = "meta-val";
    tdVal.textContent = v;
    tr.appendChild(tdKey);
    tr.appendChild(tdVal);
    table.appendChild(tr);
  }
  return table;
}

/**
 * Render a `ChatTranslationResult` into the character's output box and meta div.
 *
 * Handles all three result states:
 * - `result === null`  — Character was not requested (B when only A was sent).
 *   Shows "Not requested." placeholder and a neutral `"—"` badge.
 * - `status === "success"` — Shows IC text and a green "ok" badge.
 *   Builds the IPC meta table and makes the meta div visible.
 * - `status === "fallback.*"` — Shows an error message in `--col-err` and a
 *   red badge with a human-readable description of the failure mode.
 *
 * @param {HTMLElement}     outputBox   - The `.output-box` element to write into.
 * @param {HTMLElement}     metaDiv     - The meta div below the output box.
 * @param {HTMLElement}     statusBadge - The inline status badge element.
 * @param {Object|null}     result      - `ChatTranslationResult` from the API,
 *                                         or `null` if not requested.
 * @returns {void}
 */
function renderTranslationResult(outputBox, metaDiv, statusBadge, result) {
  outputBox.textContent = "";
  metaDiv.textContent = "";
  metaDiv.classList.add("hidden");

  if (result === null) {
    outputBox.innerHTML = '<span class="placeholder-text">Not requested.</span>';
    statusBadge.textContent = "—";
    statusBadge.className = "badge badge--muted";
    return;
  }

  // ── Status badge ────────────────────────────────────────────────────── //
  if (result.status === "success") {
    statusBadge.textContent = "ok";
    statusBadge.className = "badge";
  } else if (result.status === "fallback.api_error") {
    statusBadge.textContent = "api error";
    statusBadge.className = "badge badge--err";
  } else {
    // fallback.validation_failed
    statusBadge.textContent = "rejected";
    statusBadge.className = "badge badge--err";
  }

  if (result.ic_text) {
    outputBox.textContent = result.ic_text;
    const metaTable = buildIpcMetaTable(result);
    metaDiv.appendChild(metaTable);
    metaDiv.classList.remove("hidden");
  } else {
    const errSpan = document.createElement("span");
    errSpan.style.color = "var(--col-err)";
    errSpan.textContent =
      result.status === "fallback.api_error"
        ? "Ollama unreachable or timed out — check host and model."
        : "Output rejected by validator (PASSTHROUGH or constraint violation).";
    outputBox.appendChild(errSpan);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Live mode
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sync UI elements to the current `chatState.liveMode` value.
 *
 * When live mode is on:
 * - Per-character Send buttons become visible.
 * - The batch Translate button is disabled (only individual sends work).
 * - The in-game output section is revealed.
 *
 * When live mode is off:
 * - Send buttons are hidden.
 * - Translate button is re-enabled.
 * - Game section is hidden.
 *
 * @returns {void}
 */
function updateLiveModeUI() {
  const live = chatState.liveMode;
  for (const ch of ["a", "b"]) {
    charDom(ch).btnSend.classList.toggle("hidden", !live);
  }
  dom.btnTranslate.disabled = live;
  dom.chatGameSection.classList.toggle("hidden", !live);
}

/**
 * Append a single in-game log entry to the game output panel.
 *
 * Removes the placeholder text on first call.  Each entry shows the
 * character letter, the original OOC message, the channel tag, and the
 * translated IC text in a flex row:
 *
 *   ``[A]  "she glances around cautiously"  [say]  She surveys the chamber…``
 *
 * The OOC span is toggled visible/hidden by the "OOC" button in the game
 * log header (which adds/removes `.chat-game-output--hide-ooc` on the
 * parent container).  The panel auto-scrolls to the latest entry.
 *
 * The entry is also pushed to `chatState.gameLog` so that clipboard copy
 * and save functions can access the full log without scraping the DOM.
 *
 * @param {"a"|"b"} ch                     - Character identifier.
 * @param {string}  channel                - Channel name (e.g. "say", "yell", "whisper").
 * @param {string}  oocMessage             - Original out-of-character text typed by the player.
 *                                           Displayed in the OOC column; included in copy/save.
 * @param {string}  icText                 - The translated IC text to display.
 * @param {string}  model                  - Ollama model tag used for this translation.
 * @param {string|null} [ipcId]            - IPC ID from the translation result (the 'ipc' row
 *                                           in the in-browser IPC meta table), or null if not
 *                                           available.
 * @param {string|null} [inputHash]        - SHA-256 of the canonical input dict (the 'input'
 *                                           row in the IPC meta table), or null if unavailable.
 * @param {string|null} [systemPromptHash] - SHA-256 of the fully-rendered IC system prompt
 *                                           (the 'prompt' row in the IPC meta table).
 * @param {string|null} [outputHash]       - SHA-256 of the normalised IC output text (the
 *                                           'output' row in the IPC meta table).
 * @returns {void}
 */
function appendGameEntry(
  ch, channel, oocMessage, icText, model,
  ipcId = null, inputHash = null, systemPromptHash = null, outputHash = null,
) {
  chatState.logSeq++;
  chatState.gameLog.push({
    ch, channel, oocMessage, icText, model,
    ipcId, inputHash, systemPromptHash, outputHash,
  });
  const placeholder = dom.chatGameOutput.querySelector(".placeholder-text");
  if (placeholder) placeholder.remove();

  const entry = document.createElement("div");
  entry.className = `game-entry game-entry--${ch}`;

  // ── Column 1: character letter (A / B) ─────────────────────────────── //
  const charEl = document.createElement("span");
  charEl.className = "game-entry__char";
  charEl.textContent = ch.toUpperCase();

  // ── Column 2: original OOC message ─────────────────────────────────── //
  // Visibility controlled by the "OOC" toggle button via the
  // .chat-game-output--hide-ooc class on the parent container.
  const oocEl = document.createElement("span");
  oocEl.className = "game-entry__ooc";
  // Display in quotes so it reads as a citation of the player's words.
  oocEl.textContent = `"${oocMessage}"`;
  oocEl.title = oocMessage; // full text on hover in case it's truncated

  // ── Column 3: channel tag ([say] / [yell] / [whisper]) ─────────────── //
  const channelEl = document.createElement("span");
  channelEl.className = "game-entry__channel";
  channelEl.textContent = `[${channel}]`;

  // ── Column 4: translated IC text ───────────────────────────────────── //
  const textEl = document.createElement("span");
  textEl.className = "game-entry__text";
  textEl.textContent = icText;

  entry.appendChild(charEl);
  entry.appendChild(oocEl);
  entry.appendChild(channelEl);
  entry.appendChild(textEl);
  dom.chatGameOutput.appendChild(entry);
  dom.chatGameOutput.scrollTop = dom.chatGameOutput.scrollHeight;
}

/**
 * Translate a single character's OOC message in live mode.
 *
 * Builds a minimal `ChatTranslationRequest` with only the specified
 * character set (the other is explicitly null), POSTs to
 * `/api/translate_chat`, renders the result into the character's output
 * box, and on success appends an entry to the in-game log.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Promise<void>}
 */
async function sendForChar(ch) {
  if (chatState.busy) return;
  const model = getChatModelName();
  if (!model) { setStatus("No model specified."); return; }

  const axes = buildAxesForRequest(ch);
  if (!axes || Object.keys(axes).length === 0) {
    setStatus(`Load an example for Character ${ch.toUpperCase()} before sending.`);
    return;
  }
  const ooc = charDom(ch).oocTextarea.value.trim();
  if (!ooc) { setStatus(`Enter an OOC message for Character ${ch.toUpperCase()}.`); return; }

  const channel = charDom(ch).channelSelect.value;
  const charInput = {
    axes,
    ooc_message: ooc,
    channel,
    active_axes: chatState[ch].activeAxes ? [...chatState[ch].activeAxes] : null,
  };

  const reqBody = {
    character_a: ch === "a" ? charInput : null,
    character_b: ch === "b" ? charInput : null,
    model,
    temperature: parseFloat(dom.chatTempInput.value),
    max_tokens: parseInt(dom.chatTokensInput.value, 10),
    seed: resolveChatSeed(),
    strict_mode: dom.chatStrictMode.checked,
    max_output_chars: parseInt(dom.chatMaxChars.value, 10),
    system_prompt: dom.chatSystemPrompt.value.trim() || null,
  };

  chatState.busy = true;
  charDom(ch).btnSend.disabled = true;
  setStatus(`Sending ${ch.toUpperCase()} via ${model}…`, true);

  const outputBox   = ch === "a" ? dom.chatAOutput : dom.chatBOutput;
  const metaDiv     = ch === "a" ? dom.chatAMeta   : dom.chatBMeta;
  const badge       = ch === "a" ? dom.chatAStatusBadge : dom.chatBStatusBadge;
  outputBox.innerHTML = '<span class="placeholder-text">Translating…</span>';

  try {
    const res = await fetch("/api/translate_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const result = ch === "a" ? data.character_a : data.character_b;
    renderTranslationResult(outputBox, metaDiv, badge, result);

    if (result && result.status === "success" && result.ic_text) {
      // Pass all four IPC provenance hashes from the translation result so they
      // are stored in chatState.gameLog and included in copy/save output.
      // These match the four rows shown in the in-browser IPC meta table.
      appendGameEntry(
        ch, channel, ooc, result.ic_text, model,
        result.ipc_id            ?? null,
        result.input_hash        ?? null,
        result.system_prompt_hash ?? null,
        result.output_hash        ?? null,
      );
    }
    setStatus(`${ch.toUpperCase()} sent (${model}) — ${result ? result.status : "no result"}.`);
  } catch (err) {
    outputBox.innerHTML = `<span style="color:var(--col-err)">Error: ${err.message}</span>`;
    setStatus(`Send error (${ch.toUpperCase()}): ${err.message}`);
  } finally {
    chatState.busy = false;
    charDom(ch).btnSend.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Translate (main action)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Collect form state, POST to `/api/translate_chat`, and render results.
 *
 * Performs pre-flight validation:
 * - A model must be selected.
 * - Character A must have a loaded payload with at least one active axis.
 * - Character A must have a non-empty OOC message.
 *
 * Character B is optional: it is included in the request only when a
 * payload is loaded **and** an OOC message has been entered.  If either is
 * missing, `character_b` is sent as `null` and the B output box shows
 * "Not requested.".
 *
 * On success, calls `renderTranslationResult()` for both characters.
 * On error (network failure or non-2xx response), shows the error message
 * in Character A's output box.
 *
 * Uses `chatState.busy` as a re-entrancy guard to prevent double-submission.
 *
 * @returns {Promise<void>}
 */
export async function translate() {
  if (chatState.busy) return;

  const model = getChatModelName();
  if (!model) {
    setStatus("No model specified for translation.");
    return;
  }

  const temperature = parseFloat(dom.chatTempInput.value);
  const max_tokens  = parseInt(dom.chatTokensInput.value, 10);
  const seed        = resolveChatSeed();
  const strict_mode = dom.chatStrictMode.checked;
  const max_output_chars = parseInt(dom.chatMaxChars.value, 10);
  // Empty prompt textarea → send null → server uses default IC prompt.
  const system_prompt = dom.chatSystemPrompt.value.trim() || null;

  // ── Validate Character A (required) ─────────────────────────────────── //
  const axesA = buildAxesForRequest("a");
  if (!axesA || Object.keys(axesA).length === 0) {
    setStatus("Load an example for Character A before translating.");
    return;
  }
  const oocA = dom.chatAOoc.value.trim();
  if (!oocA) {
    setStatus("Enter an OOC message for Character A.");
    return;
  }

  const character_a = {
    axes: axesA,
    ooc_message: oocA,
    channel: dom.chatAChannel.value,
    // Convert Set to Array for JSON serialisation.
    active_axes: chatState.a.activeAxes ? [...chatState.a.activeAxes] : null,
  };

  // ── Build Character B input (optional) ──────────────────────────────── //
  // Only included when both a payload and an OOC message exist.  The user
  // can use the page for A-only testing without filling in B.
  let character_b = null;
  const axesB = buildAxesForRequest("b");
  const oocB = dom.chatBOoc.value.trim();
  if (axesB && Object.keys(axesB).length > 0 && oocB) {
    character_b = {
      axes: axesB,
      ooc_message: oocB,
      channel: dom.chatBChannel.value,
      active_axes: chatState.b.activeAxes ? [...chatState.b.activeAxes] : null,
    };
  }

  const reqBody = {
    character_a,
    character_b,
    model,
    temperature,
    max_tokens,
    seed,
    strict_mode,
    max_output_chars,
    system_prompt,
  };

  // ── Pre-request UI state ─────────────────────────────────────────────── //
  chatState.busy = true;
  dom.btnTranslate.disabled = true;
  setStatus(`Translating via ${model}…`, true);

  // Show "Translating…" placeholder in output boxes while in-flight.
  dom.chatAOutput.innerHTML = '<span class="placeholder-text">Translating…</span>';
  dom.chatBOutput.innerHTML = character_b
    ? '<span class="placeholder-text">Translating…</span>'
    : '<span class="placeholder-text">Not requested.</span>';
  dom.chatAMeta.textContent = "";
  dom.chatAMeta.classList.add("hidden");
  dom.chatBMeta.textContent = "";
  dom.chatBMeta.classList.add("hidden");

  try {
    const res = await fetch("/api/translate_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();

    renderTranslationResult(
      dom.chatAOutput, dom.chatAMeta, dom.chatAStatusBadge,
      data.character_a,
    );
    renderTranslationResult(
      dom.chatBOutput, dom.chatBMeta, dom.chatBStatusBadge,
      data.character_b,
    );

    // Summarise both status fields in the global status bar.
    const statusParts = [`A: ${data.character_a.status}`];
    if (data.character_b) statusParts.push(`B: ${data.character_b.status}`);
    setStatus(`Done (${model}) — ${statusParts.join(", ")}.`);

  } catch (err) {
    // Show the error message in Character A's output box and status bar.
    const errSpan = document.createElement("span");
    errSpan.style.color = "var(--col-err)";
    errSpan.textContent = `Error: ${err.message}`;
    dom.chatAOutput.textContent = "";
    dom.chatAOutput.appendChild(errSpan.cloneNode(true));
    setStatus(`Translation error: ${err.message}`);
  } finally {
    chatState.busy = false;
    dom.btnTranslate.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Game log clipboard + save
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Copy the in-game log to the clipboard as plain text.
 *
 * Each entry is formatted as:
 *
 *   ``CH | OOC message | [channel]: IC text``
 *
 * where ``CH`` is the character letter (A or B).  The pipe-separated OOC
 * field is always included so the plain-text output mirrors the MD table
 * column order (Char | OOC | Channel | IC Text).
 *
 * Updates the button label to "Copied!" for 1200 ms to provide tactile
 * feedback.
 *
 * @returns {void}
 */
function copyGameLogTxt() {
  if (chatState.gameLog.length === 0) { setStatus("No entries to copy."); return; }
  const text = chatState.gameLog
    .map(e => `${e.ch.toUpperCase()} | ${e.oocMessage} | [${e.channel}]: ${e.icText}`)
    .join("\n");
  navigator.clipboard.writeText(text).then(() => {
    dom.chatCopyLogTxt.textContent = "Copied!";
    setTimeout(() => { dom.chatCopyLogTxt.textContent = "Copy TXT"; }, 1200);
  });
}

/**
 * Copy the in-game log to the clipboard as a Markdown table.
 *
 * Produces a five-column table:
 *
 *   | # | Char | OOC | Channel | IC Text |
 *
 * Pipe characters in both OOC and IC text are backslash-escaped so they
 * do not break the table structure.  Updates the button label to "Copied!"
 * for 1200 ms to provide tactile feedback.
 *
 * @returns {void}
 */
function copyGameLogMd() {
  if (chatState.gameLog.length === 0) { setStatus("No entries to copy."); return; }
  const lines = [
    "| # | Char | OOC | Channel | IC Text |",
    "| --- | --- | --- | --- | --- |",
  ];
  chatState.gameLog.forEach((e, i) => {
    // Escape pipe characters in both fields so the MD table stays well-formed.
    const oocEscaped = e.oocMessage.replace(/\|/g, "\\|");
    const icEscaped  = e.icText.replace(/\|/g, "\\|");
    lines.push(`| ${i + 1} | ${e.ch.toUpperCase()} | ${oocEscaped} | ${e.channel} | ${icEscaped} |`);
  });
  navigator.clipboard.writeText(lines.join("\n")).then(() => {
    dom.chatCopyLogMd.textContent = "Copied!";
    setTimeout(() => { dom.chatCopyLogMd.textContent = "Copy MD"; }, 1200);
  });
}

/**
 * Save the in-game log to disk and immediately download it as a zip.
 *
 * Two-step flow:
 * 1. POST ``/api/save_chat`` → server writes files, returns ``folder_name``.
 * 2. GET ``/api/save/{folder}/export`` → server streams zip → browser download.
 *
 * Includes the currently loaded character payloads and system prompt so the
 * save package is self-contained.
 *
 * @returns {Promise<void>}
 */
async function saveChatLog() {
  if (chatState.gameLog.length === 0) { setStatus("No game log entries to save."); return; }

  const model = getChatModelName();
  const reqBody = {
    entries: chatState.gameLog.map(e => ({
      ch: e.ch,
      channel: e.channel,
      // ooc_message serialises to the ChatLogEntry schema field of the same name.
      ooc_message: e.oocMessage ?? "",
      ic_text: e.icText,
      model: e.model,
      // All four IPC provenance hashes are included so the server can write a
      // complete provenance record to metadata.json without re-computing them.
      // These match the four rows displayed in the in-browser IPC meta table.
      ipc_id:             e.ipcId            ?? null,
      input_hash:         e.inputHash        ?? null,
      system_prompt_hash: e.systemPromptHash ?? null,
      output_hash:        e.outputHash       ?? null,
    })),
    character_a: chatState.a.payload ? chatState.a.payload.axes : null,
    character_b: chatState.b.payload ? chatState.b.payload.axes : null,
    model,
    temperature: parseFloat(dom.chatTempInput.value),
    max_tokens: parseInt(dom.chatTokensInput.value, 10),
    seed: resolveChatSeed(),
    system_prompt: dom.chatSystemPrompt.value.trim() || null,
  };

  dom.chatSaveLog.disabled = true;
  setStatus("Saving game log…", true);

  try {
    // Step 1: POST save
    const saveRes = await fetch("/api/save_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
    });
    if (!saveRes.ok) {
      const errData = await saveRes.json().catch(() => ({ detail: saveRes.statusText }));
      throw new Error(errData.detail || `HTTP ${saveRes.status}`);
    }
    const saveData = await saveRes.json();

    // Step 2: GET zip → trigger browser download
    const exportRes = await fetch(
      `/api/save/${encodeURIComponent(saveData.folder_name)}/export`
    );
    if (!exportRes.ok) throw new Error(`Export HTTP ${exportRes.status}`);
    const blob = await exportRes.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${saveData.folder_name}_chat.zip`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);

    setStatus(`Chat log saved — ${saveData.files.length} files in ${saveData.folder_name}.`);
  } catch (err) {
    setStatus(`Save error: ${err.message}`);
  } finally {
    dom.chatSaveLog.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Perform all startup data-loading for the Chat Translation page.
 *
 * Runs in parallel:
 * - `loadChatExampleList()` — populates both example `<select>` dropdowns.
 * - `loadChatIcPromptList()` — populates the IC prompt `<select>`.
 *
 * Called once from `init()` in `mod-init.js` during Phase 3 (async data
 * loading), after all event listeners have been wired.
 *
 * @returns {Promise<void>} Resolves when both lists have been fetched.
 */
export async function initChatTranslation() {
  await Promise.all([
    loadChatExampleList(),
    loadChatIcPromptList(),
  ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Event wiring
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Wire all interactive event listeners for the Chat Translation page.
 *
 * Attaches handlers for:
 * - Example load buttons and dropdowns (A and B)
 * - JSON textarea changes (debounced 280 ms) → parse and rebuild sliders
 * - Relabel and Randomise buttons (A and B)
 * - Auto-label toggle (A and B) — rebuilds sliders and triggers relabel
 * - Temperature slider ↔ number input sync
 * - Ollama host field (debounced 600 ms) → refresh model list
 * - IC prompt load button → fetch and populate the system prompt textarea
 * - System prompt textarea → update the prompt badge
 * - Translate button → `translate()`
 * - Live toggle → `updateLiveModeUI()` (shows Send buttons + game section)
 * - Per-character Send buttons (A and B) → `sendForChar(ch)` in live mode
 * - Clear log button → empties game output panel and resets `chatState.gameLog`
 * - OOC toggle button → toggles `.chat-game-output--hide-ooc` on the game output
 *                       container; shows/hides every `.game-entry__ooc` span via CSS
 * - Copy TXT button → `copyGameLogTxt()` (plain-text clipboard; includes OOC)
 * - Copy MD button  → `copyGameLogMd()` (5-column Markdown table; includes OOC)
 * - Save all data button → `saveChatLog()` (server write + zip download; includes OOC)
 *
 * Called once during startup by the mod-events coordinator
 * ({@link module:mod-events~wireEvents}).
 *
 * @returns {void}
 */
export function wireChatTranslationEvents() {
  // ── Example load (A and B) ────────────────────────────────────────── //
  for (const ch of ["a", "b"]) {
    const cd = charDom(ch);

    // Load button: fetch the selected example name.
    cd.btnLoadExample.addEventListener("click", () => {
      loadChatExample(ch, cd.exampleSelect.value);
    });

    // JSON textarea (debounced) → parse → rebuild sliders.
    // Active axes are preserved so manual JSON edits don't reset checkboxes.
    cd.jsonTextarea.addEventListener(
      "input",
      debounce(() => {
        const obj = safeParse(cd.jsonTextarea.value);
        if (!obj) {
          setJsonBadge(ch, false);
          setStatus("Invalid JSON.");
          return;
        }
        setJsonBadge(ch, true);
        chatState[ch].payload = obj;
        buildChatSliders(ch);
        setStatus("JSON updated.");
      }, 280)
    );

    // Relabel button.
    cd.btnRelabel.addEventListener("click", () => relabelChatChar(ch));

    // Randomise button.
    cd.btnRandomise.addEventListener("click", () => randomiseChatChar(ch));

    // Auto-label toggle: rebuild sliders (to enable/disable label inputs)
    // and immediately trigger a relabel if now checked.
    cd.autoLabel.addEventListener("change", () => {
      buildChatSliders(ch);
      if (cd.autoLabel.checked) relabelChatChar(ch);
    });
  }

  // ── Temperature range ↔ number input sync ────────────────────────── //
  dom.chatTempRange.addEventListener("input", () => {
    dom.chatTempInput.value = dom.chatTempRange.value;
  });
  dom.chatTempInput.addEventListener("input", () => {
    const v = clamp(parseFloat(dom.chatTempInput.value) || 0, 0, 2);
    dom.chatTempRange.value = v;
  });

  // ── Ollama host (debounced 600 ms) → refresh model list ──────────── //
  dom.chatOllamaHost.addEventListener(
    "input",
    debounce(() => refreshChatModels(), 600)
  );

  // ── IC prompt loader ──────────────────────────────────────────────── //
  dom.chatBtnLoadPrompt.addEventListener("click", async () => {
    const name = dom.chatPromptSelect.value;
    if (!name) return;
    try {
      const res = await fetch(`/api/prompts/${encodeURIComponent(name)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      dom.chatSystemPrompt.value = text;
      updateChatPromptBadge();
      setStatus(`IC prompt "${name}" loaded.`);
    } catch (err) {
      setStatus(`Prompt load error: ${err.message}`);
    }
  });

  // Update badge whenever the system prompt textarea changes.
  dom.chatSystemPrompt.addEventListener("input", () => updateChatPromptBadge());

  // ── Translate button ──────────────────────────────────────────────── //
  dom.btnTranslate.addEventListener("click", () => translate());

  // ── Live toggle ───────────────────────────────────────────────────── //
  dom.chatLiveToggle.addEventListener("change", () => {
    chatState.liveMode = dom.chatLiveToggle.checked;
    updateLiveModeUI();
  });

  // ── Per-character Send buttons ────────────────────────────────────── //
  for (const ch of ["a", "b"]) {
    charDom(ch).btnSend.addEventListener("click", () => sendForChar(ch));
  }

  // ── Clear game log ────────────────────────────────────────────────── //
  dom.chatClearLog.addEventListener("click", () => {
    dom.chatGameOutput.innerHTML = '<span class="placeholder-text">Send a message to see in-game output.</span>';
    chatState.logSeq = 0;
    chatState.gameLog = [];
  });

  // ── OOC visibility toggle ────────────────────────────────────────── //
  // Toggles the .chat-game-output--hide-ooc modifier class on the game
  // output container.  When the class is present, CSS hides every
  // .game-entry__ooc span without reflowing the other columns.
  // The button itself uses is-active (amber border/glow) to reflect the
  // current state: is-active = OOC visible, no is-active = OOC hidden.
  dom.chatToggleOoc.addEventListener("click", () => {
    const nowVisible = dom.chatGameOutput.classList.toggle("chat-game-output--hide-ooc");
    // classList.toggle returns true when the class was ADDED (i.e. OOC is now HIDDEN).
    // is-active should be present when OOC is VISIBLE (class absent), so invert.
    dom.chatToggleOoc.classList.toggle("is-active", !nowVisible);
  });

  // ── Game log copy + save ──────────────────────────────────────────── //
  dom.chatCopyLogTxt.addEventListener("click", () => copyGameLogTxt());
  dom.chatCopyLogMd.addEventListener("click",  () => copyGameLogMd());
  dom.chatSaveLog.addEventListener("click",    () => saveChatLog());
}

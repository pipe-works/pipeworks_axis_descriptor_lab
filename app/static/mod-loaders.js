/**
 * mod-loaders.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Example and system prompt loaders.
 *
 * Handles two parallel data-loading flows that share the same pattern:
 *
 *   1. **Examples** — JSON axis payloads from `/api/examples`
 *      loadExampleList()  → populate example <select>
 *      loadExample(name)  → fetch payload → state.payload → sliders
 *
 *   2. **Prompts** — system prompt text files from `/api/prompts`
 *      loadPromptList()   → populate prompt <select>
 *      loadPrompt(name)   → fetch text → system prompt textarea
 *
 * Both flows populate a `<select>` dropdown at startup and load selected
 * items on button click or double-click.
 *
 * Imports: mod-state, mod-status, mod-sync
 */

import { state, dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import { syncJsonTextarea, buildSlidersFromJson, setJsonBadge, updateSystemPromptBadge } from "./mod-sync.js";

/**
 * Fetch the list of available examples from the server and populate the
 * example `<select>` dropdown.
 *
 * Called once at startup from `init()`.  The first option is always a
 * placeholder ("— choose —"); real example names follow.
 */
export async function loadExampleList() {
  try {
    const res  = await fetch("/api/examples");
    const list = await res.json();
    const defaultOpt = new Option("\u2014 choose \u2014", "");
    const opts = list.map((name) => new Option(name, name));
    dom.exampleSelect.replaceChildren(defaultOpt, ...opts);
  } catch (err) {
    setStatus(`Failed to load examples: ${err.message}`);
  }
}

/**
 * Load a named example from the server, update state, rebuild sliders and
 * sync the JSON textarea.
 *
 * Also snapshots the original axes via a JSON round-trip deep copy, so
 * the slider panel can highlight any user modifications relative to the
 * loaded example.
 *
 * @param {string} name - Example stem (e.g. "example_a").  No-op if empty.
 */
export async function loadExample(name) {
  if (!name) return;

  setStatus(`Loading ${name}…`, true);
  try {
    const res     = await fetch(`/api/examples/${name}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();

    state.payload = payload;

    // Deep copy via JSON round-trip to avoid shared references
    state.originalAxes = JSON.parse(JSON.stringify(payload.axes || {}));

    syncJsonTextarea();
    buildSlidersFromJson();
    setJsonBadge(true);
    setStatus(`Loaded ${name}.`);
  } catch (err) {
    setStatus(`Error loading example: ${err.message}`);
  }
}

/**
 * Fetch the list of available system prompts from the server and populate
 * the prompt `<select>` dropdown.
 *
 * Called once at startup from `init()`.  Mirrors `loadExampleList()` but
 * hits `/api/prompts` instead of `/api/examples`.
 */
export async function loadPromptList() {
  try {
    const res  = await fetch("/api/prompts");
    const list = await res.json();

    const defaultOpt = new Option("\u2014 choose \u2014", "");
    const opts = list.map((name) => new Option(name, name));
    dom.promptSelect.replaceChildren(defaultOpt, ...opts);
  } catch (err) {
    setStatus(`Failed to load prompt list: ${err.message}`);
  }
}

/**
 * Load a named system prompt from the server and populate the system
 * prompt override textarea with its content.
 *
 * Fetches plain text from `/api/prompts/{name}` (not JSON — the endpoint
 * returns `PlainTextResponse`) and sets it as the textarea value.  The
 * loaded text takes effect on the next generate call.
 *
 * After loading, the System Prompt `<details>` is opened so the user
 * can immediately see the loaded content, and the override badge is
 * updated to reflect that custom content is now active.
 *
 * @param {string} name - Prompt stem (e.g. "system_prompt_v01").  No-op if empty.
 */
export async function loadPrompt(name) {
  if (!name) return;

  setStatus(`Loading prompt ${name}…`, true);
  try {
    const res = await fetch(`/api/prompts/${name}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    // Use .text() because the endpoint returns PlainTextResponse, not JSON
    const text = await res.text();

    dom.systemPromptTextarea.value = text;
    updateSystemPromptBadge();

    // Clear the "pending load" highlight now that the prompt is loaded
    dom.btnLoadPrompt.classList.remove("is-active");

    // Ensure the System Prompt <details> is open so the user can see it
    const details = document.getElementById("system-prompt-details");
    if (details && !details.open) details.open = true;

    setStatus(`Loaded prompt: ${name}`);
  } catch (err) {
    setStatus(`Error loading prompt: ${err.message}`);
  }
}

/**
 * Wire loader-related event listeners.
 *
 * Registers handlers for:
 *   - Load Example button click + example dropdown double-click
 *   - Load Prompt button click + prompt dropdown double-click
 *   - Prompt dropdown change → highlight Load button as "pending"
 *
 * Called once during startup by the mod-events coordinator.
 */
export function wireLoaderEvents() {
  // ── Load example button ────────────────────────────────────────── //
  dom.btnLoadExample.addEventListener("click", () => {
    const name = dom.exampleSelect.value;
    if (name) loadExample(name);
  });

  // Double-click the dropdown also loads (convenience shortcut)
  dom.exampleSelect.addEventListener("dblclick", () => {
    const name = dom.exampleSelect.value;
    if (name) loadExample(name);
  });

  // ── Load prompt button ─────────────────────────────────────────── //
  dom.btnLoadPrompt.addEventListener("click", () => {
    const name = dom.promptSelect.value;
    if (name) loadPrompt(name);
  });

  // Double-click the prompt dropdown also loads (mirrors examples)
  dom.promptSelect.addEventListener("dblclick", () => {
    const name = dom.promptSelect.value;
    if (name) loadPrompt(name);
  });

  // Highlight the Load button when the user selects a different prompt
  // to signal "you've picked something but not loaded it yet"
  dom.promptSelect.addEventListener("change", () => {
    if (dom.promptSelect.value) {
      dom.btnLoadPrompt.classList.add("is-active");
    } else {
      dom.btnLoadPrompt.classList.remove("is-active");
    }
  });
}

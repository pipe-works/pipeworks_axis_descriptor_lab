/**
 * mod-init.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Application entry point — orchestrates startup.
 *
 * This is the single `<script type="module">` loaded by `index.html`.
 * ES modules with `type="module"` are **deferred by default**, meaning
 * the DOM is guaranteed to be fully parsed before any module code runs.
 * The `DOMContentLoaded` listener is retained as a defensive safeguard.
 *
 * Startup sequence
 * ────────────────
 *   1. `wireThemeToggle()`   — restore saved theme preference
 *   2. `wireTooltipToggle()` — restore saved tooltip preference + delegation
 *   3. `wireEvents()`        — register all interactive event listeners
 *   4. `loadExampleList()`   — populate the example `<select>` from server
 *   5. `loadPromptList()`    — populate the prompt `<select>` from server
 *   6. Auto-load first example (if any exist in the dropdown)
 *   7. `setStatus("Ready.")` — signal successful initialisation
 *
 * Theme and tooltip toggles are wired first (before `wireEvents`) so
 * their saved preferences are applied before any other UI interaction
 * is possible.
 *
 * Import graph (this module is the root):
 *   mod-init.js
 *     ├── mod-state.js     (dom refs)
 *     ├── mod-status.js    (setStatus)
 *     ├── mod-theme.js     (wireThemeToggle)
 *     ├── mod-tooltip.js   (wireTooltipToggle)
 *     ├── mod-events.js    (wireEvents → all wire*Events)
 *     └── mod-loaders.js   (loadExampleList, loadPromptList, loadExample)
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import { wireThemeToggle } from "./mod-theme.js";
import { wireTooltipToggle } from "./mod-tooltip.js";
import { wireEvents } from "./mod-events.js";
import { loadExampleList, loadPromptList, loadExample } from "./mod-loaders.js";

/**
 * Application entry point.  Runs once the DOM is ready.
 *
 * Performs all initialisation in a defined sequence:
 *   1. Standalone UI preferences (theme, tooltips)
 *   2. All interactive event wiring
 *   3. Async data loading (examples, prompts)
 *   4. Auto-load first example for immediate usability
 *
 * @returns {Promise<void>} Resolves when startup is complete.
 */
async function init() {
  // Phase 1: Standalone UI preferences (no deps on other modules)
  wireThemeToggle();
  wireTooltipToggle();

  // Phase 2: Wire all interactive event listeners
  wireEvents();

  // Phase 3: Populate dropdowns from server
  await loadExampleList();
  await loadPromptList();

  // Phase 4: Auto-load the first example if any exist
  // options[0] is the "— choose —" placeholder; options[1] is the first real example
  const firstOption = dom.exampleSelect.options[1];
  if (firstOption) {
    dom.exampleSelect.value = firstOption.value;
    await loadExample(firstOption.value);
  }

  setStatus("Ready. Load an example or paste JSON to begin.");
}

// Boot when the DOM is fully parsed (defensive — modules are already deferred)
document.addEventListener("DOMContentLoaded", init);

/**
 * mod-events.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Thin event-wiring coordinator.
 *
 * This module replaces the original monolithic `wireEvents()` function
 * (277 lines, 41 event listeners in app.js) with a simple dispatcher
 * that calls each feature module's `wire*Events()` function in turn.
 *
 * The call order mirrors the DOM layout (left panel → centre panel →
 * right panel → footer actions), but is functionally independent —
 * event handlers are self-contained within their feature modules.
 *
 * Wiring sequence
 * ───────────────
 *   1. wireSyncEvents()             — JSON textarea, system prompt, temp slider
 *   2. wireLoaderEvents()           — example + prompt load buttons / dropdowns
 *   3. wireGenerateEvents()         — generate button, set-baseline button
 *   4. wireDiffEvents()             — tmap mode toggle, copy TSV, copy MD
 *   5. wireAxisEvents()             — relabel, randomise, auto-label, Ollama host
 *   6. wirePersistenceEvents()      — save, export, import, clear output
 *   7. wireIndicatorModalEvents()   — indicator tag click → modal
 *
 * Called once during startup by `init()` in `mod-init.js`.
 *
 * Imports: mod-sync, mod-loaders, mod-generate, mod-diff,
 *          mod-axis-actions, mod-persistence, mod-indicator-modal
 */

import { wireSyncEvents } from "./mod-sync.js";
import { wireLoaderEvents } from "./mod-loaders.js";
import { wireGenerateEvents } from "./mod-generate.js";
import { wireDiffEvents } from "./mod-diff.js";
import { wireAxisEvents } from "./mod-axis-actions.js";
import { wirePersistenceEvents } from "./mod-persistence.js";
import { wireIndicatorModalEvents } from "./mod-indicator-modal.js";

/**
 * Wire all interactive event listeners across the application.
 *
 * Delegates to each feature module's dedicated wiring function.
 * Called once after DOMContentLoaded by `init()` in `mod-init.js`.
 *
 * This is the single coordination point for all user interaction
 * setup.  Adding a new feature module's events requires only adding
 * one import and one function call here.
 */
export function wireEvents() {
  wireSyncEvents();
  wireLoaderEvents();
  wireGenerateEvents();
  wireDiffEvents();
  wireAxisEvents();
  wirePersistenceEvents();
  wireIndicatorModalEvents();
}

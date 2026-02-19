/**
 * mod-status.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Status bar updates.
 *
 * Provides a single function that controls the footer status bar: message
 * text and spinner visibility.  Extracted into its own module because it
 * is imported by nearly every feature module (sync, loaders, generate,
 * diff, axis-actions, persistence) and keeping it separate avoids
 * circular imports.
 *
 * Imports: mod-state (dom refs only)
 */

import { dom } from "./mod-state.js";

/**
 * Update the status bar text and spinner visibility.
 *
 * The status bar sits in the `<footer>` and provides feedback to the
 * user on the outcome of every action (load, generate, save, error, etc.).
 * When `busy` is true the CSS spinner is shown; when false it is hidden.
 *
 * @param {string}  msg     - Message to display in the status bar.
 * @param {boolean} [busy]  - If true, shows the loading spinner.
 *                             Defaults to false (spinner hidden).
 */
export function setStatus(msg, busy = false) {
  dom.statusText.textContent = msg;
  dom.spinner.classList.toggle("hidden", !busy);
}

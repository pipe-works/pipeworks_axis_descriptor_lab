/**
 * mod-theme.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Dark/light theme toggle with localStorage persistence.
 *
 * Standalone module — no imports from other modules.
 *
 * The theme is controlled by a `data-theme` attribute on `<html>`:
 *   - `"dark"` (default) — dark industrial palette with amber accents
 *   - `"light"` — lighter variant for daytime / screenshot use
 *
 * CSS variables in `styles.css` respond to `[data-theme="light"]` to
 * swap the full colour palette.  This module only manages the attribute,
 * the button label, and the localStorage persistence (key: `padl-theme`).
 *
 * Called once during startup by `init()` in `mod-init.js`.
 */

/**
 * Initialise the theme toggle button and restore any saved preference.
 *
 * Looks up the `#theme-toggle` button in the DOM and:
 *   1. Restores the saved theme from `localStorage` (key: `padl-theme`,
 *      default: `"dark"`).
 *   2. Applies the theme: sets `data-theme` on `<html>`, updates button
 *      text (moon/sun icon), persists to localStorage.
 *   3. Wires the click handler to toggle between dark and light.
 *
 * If the `#theme-toggle` button doesn't exist in the DOM, the function
 * returns silently (graceful degradation).
 */
export function wireThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;

  /**
   * Apply the theme to the DOM, button label, and localStorage.
   * @param {string} theme - Either "dark" or "light".
   */
  const applyTheme = (theme) => {
    document.documentElement.setAttribute("data-theme", theme);
    btn.textContent = theme === "light" ? "\u263E Dark" : "\u2600 Light";
    localStorage.setItem("padl-theme", theme);
  };

  // Restore saved preference (default: dark)
  const saved = localStorage.getItem("padl-theme") || "dark";
  applyTheme(saved);

  // Toggle on click
  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(current === "dark" ? "light" : "dark");
  });
}

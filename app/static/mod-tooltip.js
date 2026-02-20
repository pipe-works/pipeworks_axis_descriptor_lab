/**
 * mod-tooltip.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Tooltip system (self-contained, no imports).
 *
 * Provides JS-positioned tooltips for elements with `data-tooltip`
 * attributes.  The tooltips can be globally toggled on/off via a
 * toolbar button, and the user's preference persists in localStorage.
 *
 * Why JS-positioned instead of CSS pseudo-elements?
 * ─────────────────────────────────────────────────
 * The three main panels use `overflow-y: auto`, which clips any
 * CSS-only tooltip (::before/::after pseudo-elements) that extends
 * beyond the panel boundary.  By appending real DOM nodes to `<body>`,
 * the tooltip renders in the viewport layer and sits outside all
 * overflow containers.
 *
 * Architecture
 * ────────────
 *   - `showTooltip(trigger)` — creates bubble + arrow, positions them
 *     relative to the trigger element with viewport-edge clamping
 *   - `hideTooltip()` — removes the active tooltip from the DOM
 *   - `wireTooltipListeners()` — event delegation on document.body
 *     using capture-phase mouseenter/mouseleave
 *   - `wireTooltipToggle()` — (exported) toolbar button + localStorage
 *
 * The tooltip state is gated by a `data-tooltips` attribute on
 * `<html>`: when set to `"on"`, hover events trigger tooltips; when
 * `"off"`, they are ignored.  This avoids any per-element bookkeeping.
 *
 * Standalone module — no imports from other modules.
 */

/**
 * The currently visible tooltip, or null if none is showing.
 *
 * Stored as a pair of DOM references so `hideTooltip()` can remove
 * both the text bubble and the directional arrow in one operation.
 *
 * @type {{ bubble: HTMLDivElement, arrow: HTMLDivElement } | null}
 */
let activeTooltip = null;

/**
 * Show a tooltip for the given trigger element.
 *
 * Creates a `.tooltip-bubble` (text container) and a `.tooltip-arrow`
 * (CSS-drawn directional indicator), appends both to `<body>`, then
 * positions them relative to the trigger's bounding rect.
 *
 * Positioning logic:
 *   1. Default placement is **below** the trigger.
 *   2. If insufficient viewport space below, flip to **above**.
 *   3. Horizontal position is centred on the trigger, clamped to
 *      `EDGE_MARGIN` pixels from the viewport edges.
 *
 * Any previously visible tooltip is dismissed before showing a new one.
 *
 * @param {HTMLElement} trigger - The element with a `data-tooltip`
 *   attribute whose text becomes the tooltip content.
 */
function showTooltip(trigger) {
  hideTooltip();

  const text = trigger.getAttribute("data-tooltip");
  if (!text) return;

  // Create the tooltip DOM nodes
  const bubble = document.createElement("div");
  bubble.className = "tooltip-bubble";
  bubble.textContent = text;

  const arrow = document.createElement("div");
  arrow.className = "tooltip-arrow";

  document.body.appendChild(bubble);
  document.body.appendChild(arrow);

  activeTooltip = { bubble, arrow };

  // ── Compute position ──────────────────────────────────────────────── //
  const triggerRect = trigger.getBoundingClientRect();
  const bubbleRect  = bubble.getBoundingClientRect();

  const vw = window.innerWidth;
  const vh = window.innerHeight;

  const GAP = 6;           // Space between trigger and arrow
  const ARROW_SIZE = 6;    // Arrow CSS border size
  const EDGE_MARGIN = 8;   // Minimum distance from viewport edge

  // Decide above vs below placement
  const spaceBelow = vh - triggerRect.bottom;
  const spaceAbove = triggerRect.top;
  const totalNeeded = GAP + ARROW_SIZE + bubbleRect.height;

  const placeAbove = spaceBelow < totalNeeded && spaceAbove > spaceBelow;

  let bubbleTop;
  let arrowTop;

  if (placeAbove) {
    // Position above the trigger
    bubbleTop = triggerRect.top - GAP - ARROW_SIZE - bubbleRect.height;
    arrowTop = triggerRect.top - GAP - ARROW_SIZE;
    arrow.style.borderTopColor = "var(--col-tooltip-border)";
  } else {
    // Position below the trigger (default)
    arrowTop = triggerRect.bottom + GAP;
    bubbleTop = arrowTop + ARROW_SIZE;
    arrow.style.borderBottomColor = "var(--col-tooltip-border)";
  }

  // Centre horizontally on the trigger, clamped to viewport edges
  const triggerCentreX = triggerRect.left + triggerRect.width / 2;
  let bubbleLeft = triggerCentreX - bubbleRect.width / 2;

  if (bubbleLeft < EDGE_MARGIN) {
    bubbleLeft = EDGE_MARGIN;
  }

  if (bubbleLeft + bubbleRect.width > vw - EDGE_MARGIN) {
    bubbleLeft = vw - EDGE_MARGIN - bubbleRect.width;
  }

  const arrowLeft = triggerCentreX - ARROW_SIZE;

  // Apply computed positions
  bubble.style.top  = `${bubbleTop}px`;
  bubble.style.left = `${bubbleLeft}px`;

  arrow.style.top  = `${arrowTop}px`;
  arrow.style.left = `${arrowLeft}px`;
}

/**
 * Remove the currently visible tooltip from the DOM.
 *
 * Removes both the bubble and arrow elements and resets `activeTooltip`
 * to null.  Safe to call when no tooltip is visible (no-op).
 */
function hideTooltip() {
  if (!activeTooltip) return;

  activeTooltip.bubble.remove();
  activeTooltip.arrow.remove();
  activeTooltip = null;
}

/**
 * Wire up tooltip show/hide listeners using event delegation on
 * `document.body`.
 *
 * Uses **capture-phase** listeners (`true` as the third argument to
 * `addEventListener`) because `mouseenter` does not bubble — capture
 * phase is required for delegation to work.
 *
 * The tooltip system is gated by the `data-tooltips` attribute on
 * `<html>`: if it's not `"on"`, hover events are silently ignored.
 * This means toggling tooltips off requires no per-element cleanup.
 *
 * @private — called only by `wireTooltipToggle()`.
 */
function wireTooltipListeners() {
  document.body.addEventListener("mouseenter", (e) => {
    // Bail if tooltips are globally disabled
    if (document.documentElement.getAttribute("data-tooltips") !== "on") return;

    const trigger = e.target.closest("[data-tooltip]");
    if (!trigger) return;

    showTooltip(trigger);
  }, true);

  document.body.addEventListener("mouseleave", (e) => {
    const trigger = e.target.closest("[data-tooltip]");
    if (!trigger) return;

    hideTooltip();
  }, true);
}

/**
 * Initialise the tooltip toggle button and restore any saved preference.
 *
 * Looks up the `#tooltip-toggle` button in the DOM and:
 *   1. Restores the saved tooltip state from `localStorage` (key:
 *      `padl-tooltips`, default: `"off"`).
 *   2. Applies the state: sets `data-tooltips` on `<html>`, updates
 *      button text and active class, persists to localStorage.
 *   3. Wires the click handler to toggle between "on" and "off".
 *   4. Calls `wireTooltipListeners()` to set up the hover delegation.
 *
 * If the `#tooltip-toggle` button doesn't exist in the DOM, the
 * function returns silently (graceful degradation).
 *
 * Called once during startup by `init()` in `mod-init.js`.
 */
export function wireTooltipToggle() {
  const btn = document.getElementById("tooltip-toggle");
  if (!btn) return;

  /**
   * Apply the tooltip state to the DOM, button text, and localStorage.
   * @param {string} tooltipState - Either "on" or "off".
   */
  const applyTooltips = (tooltipState) => {
    document.documentElement.setAttribute("data-tooltips", tooltipState);
    btn.textContent = tooltipState === "on" ? "? Tips \u2713" : "? Tips";
    btn.classList.toggle("is-active", tooltipState === "on");
    localStorage.setItem("padl-tooltips", tooltipState);

    // Dismiss any visible tooltip when turning tooltips off
    if (tooltipState === "off") {
      hideTooltip();
    }
  };

  // Restore saved preference (default: off)
  const saved = localStorage.getItem("padl-tooltips") || "off";
  applyTooltips(saved);

  // Toggle on click
  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-tooltips") || "off";
    applyTooltips(current === "on" ? "off" : "on");
  });

  // Set up the hover delegation listeners
  wireTooltipListeners();
}

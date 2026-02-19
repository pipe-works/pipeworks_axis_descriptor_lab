/**
 * mod-indicator-modal.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Indicator tooltip text and click-to-open modal for micro-indicator tags.
 *
 * Purpose
 * ───────
 * Each micro-indicator tag in the Transformation Map supports two layers
 * of progressive disclosure:
 *
 *   1. **Hover tooltip** — a one-line structural definition, displayed by
 *      the existing tooltip system in mod-tooltip.js.  This module provides
 *      the tooltip text via `getIndicatorTooltip()`.
 *
 *   2. **Click modal** — a richer panel showing the indicator's title,
 *      full definition, heuristic basis, a removed/added example pair,
 *      and a link to the Sphinx documentation.  The modal is opened by
 *      `openIndicatorModal()`.
 *
 * Design philosophy (from the Structural Learning Layer spec)
 * ───────────────────────────────────────────────────────────
 * Indicators are **structural pattern labels** — deterministic heuristics,
 * educational cues, and analysis aids.  They describe structure, not intent.
 * The tooltip says "A shift of this type appears to have occurred."  It
 * never says "The model performed X because of Y."
 *
 * The modal exists to **encourage curiosity** and **introduce conceptual
 * vocabulary** — a beginner sees "[modality shift]", looks it up, and
 * learns.  An expert sees "[compression + intensity ↑]" and understands
 * instantly.  Same tool, different depth.
 *
 * DOM strategy
 * ────────────
 * The modal overlay is created dynamically on first use and appended to
 * `<body>`, matching the pattern used by mod-tooltip.js.  This avoids
 * polluting the HTML template and ensures the modal escapes any
 * `overflow: hidden` containers.  Once created, the overlay is shown /
 * hidden by toggling an `.is-visible` CSS class — no DOM churn on
 * repeated opens.
 *
 * Imports: none (standalone leaf module)
 */

// ─────────────────────────────────────────────────────────────────────────────
// Indicator definitions
// ─────────────────────────────────────────────────────────────────────────────
//
// Static vocabulary sourced from the Structural Learning Layer design
// document (§12).  Keys match the exact strings returned by the server
// (and defined in ALL_INDICATORS in app/micro_indicators.py).
//
// Each entry provides:
//   - title       Title-case display name for the modal header.
//   - tooltip     One-line definition shown on hover (via data-tooltip).
//   - definition  Fuller structural definition for the modal body.
//   - heuristic   How the indicator is determined (rule-based, never AI).
//   - example     A { removed, added } pair illustrating the shift.
// ─────────────────────────────────────────────────────────────────────────────

/** @type {Object<string, {title: string, tooltip: string, definition: string, heuristic: string, example: {removed: string, added: string}}>} */
const INDICATOR_DEFS = {
  "compression": {
    title: "Compression",
    tooltip: "Multiple descriptive elements condensed into fewer tokens.",
    definition:
      "Multiple descriptive elements condensed into fewer tokens. " +
      "The replacement clause is structurally shorter while preserving " +
      "the core meaning of the original.",
    heuristic: "Removed token count \u2265 2\u00d7 added token count.",
    example: {
      removed: "etched with lines that speak of hardship",
      added: "suggesting",
    },
  },

  "expansion": {
    title: "Expansion",
    tooltip: "Short phrase rewritten into longer descriptive clause.",
    definition:
      "A short phrase is rewritten into a longer descriptive clause. " +
      "The replacement elaborates on the original with additional " +
      "structural detail.",
    heuristic: "Added token count \u2265 2\u00d7 removed token count.",
    example: {
      removed: "burden",
      added: "a heavy burden weighing on him",
    },
  },

  "embodiment shift": {
    title: "Embodiment Shift",
    tooltip:
      "Abstract emotional state expressed through physical description.",
    definition:
      "Abstract or atmospheric language becomes physicalised. " +
      "An emotional or conceptual state is rewritten as a bodily " +
      "or sensory description.",
    heuristic: "Abstract lexicon \u2192 Physical lexicon match.",
    example: {
      removed: "tension hangs",
      added: "trembling hands",
    },
  },

  "abstraction \u2191": {
    title: "Abstraction \u2191",
    tooltip: "Concrete imagery replaced with abstract framing.",
    definition:
      "Concrete, physical imagery is replaced with abstract framing. " +
      "Sensory detail gives way to conceptual or evaluative language.",
    heuristic: "Physical \u2192 Abstract lexicon match.",
    example: {
      removed: "threadbare coat",
      added: "limited resources",
    },
  },

  "intensity \u2191": {
    title: "Intensity \u2191",
    tooltip: "Rhetorical force increases on a known scale.",
    definition:
      "Rhetorical force increases. A word is replaced by one that " +
      "sits higher on a known intensity scale, amplifying the " +
      "descriptive weight.",
    heuristic:
      "Replacement word at a higher position in a known intensity scale.",
    example: {
      removed: "uneasy",
      added: "perilous",
    },
  },

  "intensity \u2193": {
    title: "Intensity \u2193",
    tooltip: "Rhetorical force decreases on a known scale.",
    definition:
      "Rhetorical force decreases. A word is replaced by one that " +
      "sits lower on a known intensity scale, softening the " +
      "descriptive weight.",
    heuristic:
      "Replacement word at a lower position in a known intensity scale.",
    example: {
      removed: "perilous",
      added: "uneasy",
    },
  },

  "consolidation": {
    title: "Consolidation",
    tooltip: "Two or more clauses merged into one.",
    definition:
      "Two or more clauses are merged into a single clause. " +
      "Sentence boundaries decrease, producing denser prose.",
    heuristic: "Sentence count decreases between removed and added text.",
    example: {
      removed: "He appeared weary. His face was gaunt.",
      added: "He appeared weary, gaunt-faced.",
    },
  },

  "fragmentation": {
    title: "Fragmentation",
    tooltip: "Single clause split into multiple clauses.",
    definition:
      "A single clause is split into multiple clauses. " +
      "Sentence boundaries increase, producing more segmented prose.",
    heuristic: "Sentence count increases between removed and added text.",
    example: {
      removed: "He appeared weary and gaunt-faced.",
      added: "He appeared weary. His face was gaunt.",
    },
  },

  "tone reframing": {
    title: "Tone Reframing",
    tooltip: "Lexical substitution without structural shift.",
    definition:
      "The same structural content is expressed with different " +
      "rhetorical framing. Words change, but no other structural " +
      "indicator (compression, intensity, embodiment, etc.) applies.",
    heuristic:
      "Lexical substitution detected without intensity or embodiment change.",
    example: {
      removed: "a silent threat",
      added: "an unspoken intensity",
    },
  },

  "modality shift": {
    title: "Modality Shift",
    tooltip: "Change in descriptive mode (verb/adjective density).",
    definition:
      "The descriptive mode changes \u2014 for example, from narrative " +
      "to observational, or from atmospheric to psychological. " +
      "Detected via a significant shift in verb/adjective density.",
    heuristic:
      "Significant change in verb/adjective density between removed " +
      "and added text.",
    example: {
      removed: "the figure moved slowly through the corridor",
      added: "slow, deliberate, heavy with purpose",
    },
  },

  "lexical pivot": {
    title: "Lexical Pivot",
    tooltip: "Rare content word replaced by another rare content word.",
    definition:
      "A high-information content word is replaced by another " +
      "high-information content word. Neither word appears in the " +
      "standard lexicon sets, suggesting a meaningful but " +
      "structurally unclassified substitution.",
    heuristic:
      "Rare content word (not a stopword, not in any lexicon) replaced " +
      "by another rare content word.",
    example: {
      removed: "crevice",
      added: "fracture",
    },
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// ReadTheDocs URL for the micro-indicators API page
// ─────────────────────────────────────────────────────────────────────────────

/** @type {string} */
const DOCS_URL =
  "https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/api/micro_indicators.html";

// ─────────────────────────────────────────────────────────────────────────────
// Module-level references for the lazily-created modal DOM
// ─────────────────────────────────────────────────────────────────────────────

/** @type {HTMLDivElement|null} Overlay backdrop — created once on first open. */
let _overlay = null;

/** @type {HTMLHeadingElement|null} Title element inside the modal panel. */
let _titleEl = null;

/** @type {HTMLElement|null} Definition <dd> element. */
let _defDD = null;

/** @type {HTMLElement|null} Heuristic <dd> element. */
let _heuristicDD = null;

/** @type {HTMLElement|null} Example removed <span> element. */
let _exRemovedEl = null;

/** @type {HTMLElement|null} Example added <span> element. */
let _exAddedEl = null;

// ─────────────────────────────────────────────────────────────────────────────
// Modal DOM construction (lazy — built once on first openIndicatorModal call)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the modal overlay and panel DOM, append to `<body>`.
 *
 * The structure is:
 *
 *   div.indicator-modal-overlay          (backdrop)
 *     div.indicator-modal                (panel)
 *       button.indicator-modal__close    (× close button)
 *       h3.indicator-modal__title        (indicator name)
 *       dl.indicator-modal__dl           (definition list)
 *         dt "Definition"  → dd          (full definition text)
 *         dt "Heuristic"   → dd          (detection rule)
 *         dt "Example"     → dd          (removed/added pair)
 *       a.indicator-modal__docs-link     (Read the docs →)
 *
 * Called once; subsequent opens reuse the same DOM nodes.
 *
 * @private
 */
function _buildModal() {
  // ── Overlay (backdrop) ────────────────────────────────────────────────
  _overlay = document.createElement("div");
  _overlay.className = "indicator-modal-overlay";

  // ── Panel ─────────────────────────────────────────────────────────────
  const panel = document.createElement("div");
  panel.className = "indicator-modal";

  // ── Close button (top-right ×) ────────────────────────────────────────
  const closeBtn = document.createElement("button");
  closeBtn.className = "indicator-modal__close";
  closeBtn.setAttribute("aria-label", "Close");
  closeBtn.textContent = "\u00d7"; // ×
  panel.appendChild(closeBtn);

  // ── Title ─────────────────────────────────────────────────────────────
  _titleEl = document.createElement("h3");
  _titleEl.className = "indicator-modal__title";
  panel.appendChild(_titleEl);

  // ── Definition list (dt/dd pairs) ─────────────────────────────────────
  const dl = document.createElement("dl");
  dl.className = "indicator-modal__dl";

  // Definition row
  const dtDef = document.createElement("dt");
  dtDef.textContent = "Definition";
  _defDD = document.createElement("dd");
  dl.appendChild(dtDef);
  dl.appendChild(_defDD);

  // Heuristic row
  const dtHeur = document.createElement("dt");
  dtHeur.textContent = "Heuristic";
  _heuristicDD = document.createElement("dd");
  dl.appendChild(dtHeur);
  dl.appendChild(_heuristicDD);

  // Example row
  const dtEx = document.createElement("dt");
  dtEx.textContent = "Example";
  const ddEx = document.createElement("dd");
  ddEx.className = "indicator-modal__example";

  _exRemovedEl = document.createElement("span");
  _exRemovedEl.className = "indicator-modal__example-removed";
  ddEx.appendChild(_exRemovedEl);

  _exAddedEl = document.createElement("span");
  _exAddedEl.className = "indicator-modal__example-added";
  ddEx.appendChild(_exAddedEl);

  dl.appendChild(dtEx);
  dl.appendChild(ddEx);

  panel.appendChild(dl);

  // ── Docs link ─────────────────────────────────────────────────────────
  const docsLink = document.createElement("a");
  docsLink.className = "indicator-modal__docs-link";
  docsLink.href = DOCS_URL;
  docsLink.target = "_blank";
  docsLink.rel = "noopener noreferrer";
  docsLink.textContent = "Read the docs \u2192";
  panel.appendChild(docsLink);

  // ── Assemble and attach ───────────────────────────────────────────────
  _overlay.appendChild(panel);
  document.body.appendChild(_overlay);

  // ── Close handlers on the modal DOM ───────────────────────────────────

  // Close button click
  closeBtn.addEventListener("click", _closeModal);

  // Backdrop click (only when clicking the overlay itself, not the panel)
  _overlay.addEventListener("click", (e) => {
    if (e.target === _overlay) {
      _closeModal();
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Show / hide helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hide the modal overlay by removing the `.is-visible` class.
 *
 * Safe to call when the modal is already hidden (no-op).
 *
 * @private
 */
function _closeModal() {
  if (_overlay) {
    _overlay.classList.remove("is-visible");
  }
}

/**
 * Populate the modal with an indicator's definition and show it.
 *
 * @param {{title: string, definition: string, heuristic: string, example: {removed: string, added: string}}} def
 * @private
 */
function _showModal(def) {
  _titleEl.textContent = def.title;
  _defDD.textContent = def.definition;
  _heuristicDD.textContent = def.heuristic;
  _exRemovedEl.textContent = def.example.removed;
  _exAddedEl.textContent = def.example.added;
  _overlay.classList.add("is-visible");
}

// ═════════════════════════════════════════════════════════════════════════════
// Public API
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Get the one-line tooltip text for a micro-indicator.
 *
 * Called by `mod-diff.js` when rendering `.tmap-indicator` spans to set
 * the `data-tooltip` attribute.  The existing tooltip system in
 * `mod-tooltip.js` handles hover display automatically.
 *
 * @param {string} name — Indicator name exactly as returned by the server
 *                        (e.g. `"compression"`, `"intensity ↑"`).
 * @returns {string|null} Tooltip text, or `null` if the indicator name
 *                        is not in the vocabulary.
 */
export function getIndicatorTooltip(name) {
  const def = INDICATOR_DEFS[name];
  return def ? def.tooltip : null;
}

/**
 * Open the indicator modal for the given indicator name.
 *
 * On the first call, the modal DOM is lazily constructed and appended
 * to `<body>`.  Subsequent calls reuse the existing DOM, updating the
 * content and toggling visibility.
 *
 * If the indicator name is not recognised, this function is a no-op.
 *
 * @param {string} name — Indicator name exactly as returned by the server.
 */
export function openIndicatorModal(name) {
  const def = INDICATOR_DEFS[name];
  if (!def) return;

  // Lazy-create the modal DOM on first use.
  if (!_overlay) {
    _buildModal();
  }

  _showModal(def);
}

/**
 * Wire click and keyboard event listeners for indicator modal interaction.
 *
 * Sets up:
 *   - **Click delegation** on `#tmap-panel` — when a `.tmap-indicator`
 *     element with a `data-indicator` attribute is clicked, the modal
 *     opens with that indicator's definition.
 *   - **Escape key** — closes the modal when it is visible.
 *
 * Called once during startup by `wireEvents()` in `mod-events.js`.
 */
export function wireIndicatorModalEvents() {
  // ── Click delegation on the tmap panel ────────────────────────────────
  // Using delegation (single listener on the container) rather than
  // per-element listeners because the table DOM is rebuilt on every
  // generation / mode toggle.  Delegation survives re-renders.
  const tmapPanel = document.getElementById("tmap-panel");
  if (tmapPanel) {
    tmapPanel.addEventListener("click", (e) => {
      // Walk up from the click target to find a .tmap-indicator with
      // a data-indicator attribute (handles clicks on text nodes inside
      // the span).
      const tag = e.target.closest(".tmap-indicator[data-indicator]");
      if (tag) {
        openIndicatorModal(tag.getAttribute("data-indicator"));
      }
    });
  }

  // ── Escape key closes the modal ──────────────────────────────────────
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && _overlay && _overlay.classList.contains("is-visible")) {
      _closeModal();
    }
  });
}

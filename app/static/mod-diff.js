/**
 * mod-diff.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Word-level diff visualisation, signal isolation, and transformation map
 * with micro-indicators.
 *
 * This module handles all three layers of A/B text comparison:
 *
 *   1. **Word diff** (updateDiff) — client-side LCS-based word-level diff
 *      between state.baseline (A) and state.current (B).  Renders
 *      additions in green and deletions in red+strikethrough in the
 *      Δ Changes panel.
 *
 *   2. **Signal isolation** (updateSignalIsolation) — server-side NLP
 *      pipeline (tokenise → lemmatise → filter stopwords → set delta)
 *      via POST /api/analyze-delta.  Surfaces meaningful vocabulary pivots
 *      by filtering structural noise.
 *
 *   3. **Transformation map** (updateTransformationMap) — server-side
 *      clause-level alignment via POST /api/transformation-map, returning
 *      REMOVED | ADDED | INDICATORS rows.  Each row is annotated with
 *      micro-indicator labels (compression, embodiment shift, etc.).
 *      Falls back to client-side extraction if the server is unreachable.
 *
 * Data flow
 * ─────────
 *   generate() → state.current updated → updateDiff()
 *                                          ├─ updateSignalIsolation()
 *                                          └─ updateTransformationMap()
 *
 * Imports: mod-state, mod-utils, mod-status, mod-indicator-modal
 */

import { state, dom } from "./mod-state.js";
import { tokenise, lcsWordDiff, extractTransformationRows, makePlaceholder } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";
import { getIndicatorTooltip } from "./mod-indicator-modal.js";

/**
 * Compute and render the word-level diff between `state.baseline` (A) and
 * `state.current` (B).
 *
 * Algorithm: patience-like LCS on word tokens.
 *   - Words only in A → shown red + strikethrough in delta view.
 *   - Words only in B → shown green in delta view.
 *   - Common words → shown as-is (plain text).
 *
 * The A and B boxes show the raw texts.
 * The Δ box shows the annotated diff.
 *
 * After rendering the word diff, this function also:
 *   - Computes and displays the change percentage badge
 *   - Caches the diff in state.lastDiff for tmap toggle re-renders
 *   - Calls updateSignalIsolation() for server-side NLP analysis
 *   - Calls updateTransformationMap() for clause-level grouping
 *   - Auto-opens the diff `<details>` panel if not already open
 */
export function updateDiff() {
  const textA = state.baseline || "";
  const textB = state.current  || "";

  // Update the raw side-by-side A/B panels
  dom.diffA.textContent = textA || "(no baseline)";
  dom.diffB.textContent = textB || "(no output)";

  // Guard: both texts required for a meaningful diff
  if (!textA || !textB) {
    dom.diffDelta.textContent = "";
    dom.diffDelta.appendChild(makePlaceholder("Set a baseline and generate to compare."));
    dom.diffPct.style.display = "none";
    return;
  }

  // Tokenise and compute word-level LCS diff
  const wordsA = tokenise(textA);
  const wordsB = tokenise(textB);
  const diff = lcsWordDiff(wordsA, wordsB);

  // Build annotated diff as a DocumentFragment (avoids reflows)
  const fragment = document.createDocumentFragment();

  for (const [op, word] of diff) {
    if (op === "=") {
      // Common word — plain text
      fragment.appendChild(document.createTextNode(word + " "));
    } else if (op === "+") {
      // Added in B — green highlight
      const span = document.createElement("span");
      span.className   = "diff-add";
      span.textContent = word;
      fragment.appendChild(span);
      fragment.appendChild(document.createTextNode(" "));
    } else if (op === "-") {
      // Removed from A — red highlight + strikethrough
      const span = document.createElement("span");
      span.className   = "diff-del";
      span.textContent = word;
      fragment.appendChild(span);
      fragment.appendChild(document.createTextNode(" "));
    }
  }

  dom.diffDelta.textContent = "";
  dom.diffDelta.appendChild(fragment);

  // ── Change percentage badge ────────────────────────────────────── //
  // (insertions + deletions) / total words
  const eqCount  = diff.filter(([op]) => op === "=").length;
  const addCount = diff.filter(([op]) => op === "+").length;
  const delCount = diff.filter(([op]) => op === "-").length;
  const total    = eqCount + addCount + delCount;
  if (total > 0) {
    const pct = Math.round(((addCount + delCount) / total) * 100);
    dom.diffPct.textContent = `${pct}% changed`;
    dom.diffPct.style.display = "";
  } else {
    dom.diffPct.style.display = "none";
  }

  // Auto-open the diff <details> if it isn't already
  const detailsEl = document.getElementById("diff-details");
  if (detailsEl && !detailsEl.open) {
    detailsEl.open = true;
  }

  // Cache the diff for tmap toggle re-renders without recomputation
  state.lastDiff = diff;

  // Trigger downstream analyses
  updateSignalIsolation();
  updateTransformationMap(diff);
}

/**
 * Call POST /api/analyze-delta with the current baseline and output texts,
 * then render the results into the signal panel.
 *
 * The server-side pipeline:
 *   1. Tokenise both texts (NLTK)
 *   2. Lemmatise to base forms (WordNet)
 *   3. Remove stopwords (English)
 *   4. Compute set difference → removed / added word lists
 *
 * The frontend renders the returned word lists as a two-column
 * REMOVED | ADDED layout using inline tags.
 *
 * Called at the end of updateDiff() when both baseline and current exist.
 * Failures are shown inline in the panel, not thrown.
 */
export async function updateSignalIsolation() {
  const textA = state.baseline || "";
  const textB = state.current  || "";

  // Guard: both texts required
  if (!textA || !textB) {
    dom.signalPanel.textContent = "";
    dom.signalPanel.appendChild(
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

    const fragment = document.createDocumentFragment();

    if (data.removed.length === 0 && data.added.length === 0) {
      fragment.appendChild(makePlaceholder("No content-word differences detected."));
    } else {
      // ── Two-column "Semantic Pivot" layout ─────────────────────── //
      // REMOVED on the left, ADDED on the right.
      const grid = document.createElement("div");
      grid.className = "signal-columns";

      // ── Left column: REMOVED words ─────────────────────────────── //
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

      // ── Right column: ADDED words ──────────────────────────────── //
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

    dom.signalPanel.textContent = "";
    dom.signalPanel.appendChild(fragment);

    // Auto-open the signal <details> on first population
    const signalDetailsEl = document.getElementById("signal-details");
    if (signalDetailsEl && !signalDetailsEl.open) {
      signalDetailsEl.open = true;
    }

  } catch (err) {
    // Show error inline in the panel rather than throwing
    dom.signalPanel.textContent = "";
    const errSpan = document.createElement("span");
    errSpan.style.color = "var(--col-err)";
    errSpan.textContent = `Signal analysis error: ${err.message}`;
    dom.signalPanel.appendChild(errSpan);
  }
}

/**
 * Fetch transformation map rows (with micro-indicators) from the server
 * and render as a three-column table: REMOVED | ADDED | INDICATORS.
 *
 * The server endpoint (POST /api/transformation-map) runs sentence-aware
 * alignment, token-level diffing, and micro-indicator classification.
 * Results are cached in `state.lastTmapResponse` so the mode toggle can
 * re-render without a server round-trip.
 *
 * Falls back to client-side extraction via `extractTransformationRows()`
 * if the server is unreachable (renders a 2-column table without indicators).
 *
 * @param {Array<[string, string]>} diff - Output of lcsWordDiff(): used
 *   only as a guard (presence means a diff was computed) and as the
 *   fallback data source when the server call fails.
 */
export async function updateTransformationMap(diff) {
  // Guard: no diff data available
  if (!diff || diff.length === 0) {
    dom.tmapPanel.textContent = "";
    dom.tmapPanel.appendChild(
      makePlaceholder("Set a baseline (A) and generate to see clause-level substitutions.")
    );
    state.lastTmapResponse = null;
    return;
  }

  const textA = state.baseline || "";
  const textB = state.current  || "";

  // Need both texts for server-side analysis
  if (!textA || !textB) {
    _renderTmapClientSide(diff);
    return;
  }

  try {
    // Load indicator config from localStorage (if user has tuned thresholds)
    const storedConfig = localStorage.getItem("adl_indicator_config");
    const indicatorConfig = storedConfig ? JSON.parse(storedConfig) : null;

    const reqBody = {
      baseline_text:    textA,
      current_text:     textB,
      include_all:      true,   // Always fetch all rows; filter client-side
      indicator_config: indicatorConfig,
    };

    const res = await fetch("/api/transformation-map", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(reqBody),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();

    // Cache the full response for toggle re-renders and copy handlers
    state.lastTmapResponse = data.rows;

    _renderTmapFromServer(data.rows);
  } catch (err) {
    console.warn("[ADL] Server tmap failed, falling back to client-side:", err);
    state.lastTmapResponse = null;
    _renderTmapClientSide(diff);
  }

  // Auto-open the Transformation Map <details> on first population
  const tmapDetailsEl = document.getElementById("tmap-details");
  if (tmapDetailsEl && !tmapDetailsEl.open) {
    tmapDetailsEl.open = true;
  }
}


/**
 * Render a three-column transformation map table from server response data.
 *
 * Columns: REMOVED (red) | ADDED (green) | INDICATORS (amber tags).
 * Rows are filtered client-side based on `state.tmapIncludeAll`:
 *   - false: only rows where both removed and added are non-empty
 *   - true: all rows (inserts/deletes shown with em dashes)
 *
 * @param {Array<{removed: string, added: string, indicators: string[]}>} allRows
 *   Full server response rows (fetched with include_all=true).
 */
function _renderTmapFromServer(allRows) {
  // Filter based on mode toggle
  const rows = state.tmapIncludeAll
    ? allRows
    : allRows.filter(r => r.removed && r.added);

  const fragment = document.createDocumentFragment();

  if (rows.length === 0) {
    fragment.appendChild(makePlaceholder("No clause-level substitutions detected."));
  } else {
    // ── Three-column table: REMOVED | ADDED | INDICATORS ──────── //
    const table = document.createElement("table");
    table.className = "tmap-table tmap-table--3col";

    // Header row
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const label of ["REMOVED", "ADDED", "INDICATORS"]) {
      const th = document.createElement("th");
      th.className = "tmap-header";
      th.textContent = label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Data rows
    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");

      // REMOVED cell
      const tdR = document.createElement("td");
      tdR.className = "tmap-cell tmap-cell--removed";
      if (row.removed) {
        tdR.textContent = row.removed;
      } else {
        tdR.textContent = "\u2014";
        tdR.classList.add("tmap-cell--empty");
      }
      tr.appendChild(tdR);

      // ADDED cell
      const tdA = document.createElement("td");
      tdA.className = "tmap-cell tmap-cell--added";
      if (row.added) {
        tdA.textContent = row.added;
      } else {
        tdA.textContent = "\u2014";
        tdA.classList.add("tmap-cell--empty");
      }
      tr.appendChild(tdA);

      // INDICATORS cell — render as inline tags
      const tdI = document.createElement("td");
      tdI.className = "tmap-cell tmap-cell--indicators";
      if (row.indicators && row.indicators.length > 0) {
        for (const ind of row.indicators) {
          const tag = document.createElement("span");
          tag.className = "tmap-indicator";
          tag.textContent = ind;

          // Set data-indicator for click-to-open modal delegation
          tag.setAttribute("data-indicator", ind);

          // Set data-tooltip for hover tooltip (handled by mod-tooltip.js)
          const tip = getIndicatorTooltip(ind);
          if (tip) tag.setAttribute("data-tooltip", tip);

          tdI.appendChild(tag);
        }
      } else {
        tdI.textContent = "\u2014";
        tdI.classList.add("tmap-cell--empty");
      }
      tr.appendChild(tdI);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    fragment.appendChild(table);
  }

  dom.tmapPanel.textContent = "";
  dom.tmapPanel.appendChild(fragment);
}


/**
 * Fallback: render a two-column transformation map from client-side LCS diff.
 *
 * Used when the server endpoint is unreachable.  No indicators column.
 *
 * @param {Array<[string, string]>} diff - LCS word diff tuples.
 */
function _renderTmapClientSide(diff) {
  const rows = extractTransformationRows(diff, state.tmapIncludeAll);

  const fragment = document.createDocumentFragment();

  if (rows.length === 0) {
    fragment.appendChild(makePlaceholder("No clause-level substitutions detected."));
  } else {
    const table = document.createElement("table");
    table.className = "tmap-table";

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const label of ["REMOVED", "ADDED"]) {
      const th = document.createElement("th");
      th.className = "tmap-header";
      th.textContent = label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");

      const tdR = document.createElement("td");
      tdR.className = "tmap-cell tmap-cell--removed";
      tdR.textContent = row.removed || "\u2014";
      if (!row.removed) tdR.classList.add("tmap-cell--empty");
      tr.appendChild(tdR);

      const tdA = document.createElement("td");
      tdA.className = "tmap-cell tmap-cell--added";
      tdA.textContent = row.added || "\u2014";
      if (!row.added) tdA.classList.add("tmap-cell--empty");
      tr.appendChild(tdA);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    fragment.appendChild(table);
  }

  dom.tmapPanel.textContent = "";
  dom.tmapPanel.appendChild(fragment);
}

/**
 * Get the currently displayable transformation map rows.
 *
 * Prefers server-computed rows (with indicators) when available.
 * Falls back to client-side extraction from the cached LCS diff.
 * Applies the include-all / replacements-only filter in both cases.
 *
 * @returns {Array<{removed: string, added: string, indicators?: string[]}>}
 *   Filtered rows for display or copy, or empty array if nothing available.
 */
function _getTmapRows() {
  if (state.lastTmapResponse) {
    return state.tmapIncludeAll
      ? state.lastTmapResponse
      : state.lastTmapResponse.filter(r => r.removed && r.added);
  }
  if (state.lastDiff) {
    return extractTransformationRows(state.lastDiff, state.tmapIncludeAll);
  }
  return [];
}


/**
 * Wire diff-related event listeners.
 *
 * Registers handlers for:
 *   - Transformation Map mode toggle (replacements only ↔ all changes)
 *   - Copy TSV button (tab-separated clipboard copy)
 *   - Copy MD button (GitHub-Flavoured Markdown table clipboard copy)
 *
 * The toggle re-renders from cached data (server response or LCS diff)
 * without making a server call.  Copy handlers produce 3-column output
 * when server data (with indicators) is available, 2-column otherwise.
 *
 * Called once during startup by the mod-events coordinator.
 */
export function wireDiffEvents() {
  // ── Transformation Map mode toggle ─────────────────────────────── //
  // Switches between "Replacements only" (default) and "All changes"
  // (includes pure inserts and pure deletes).  Re-renders instantly
  // from cached data without any server call.
  dom.btnTmapMode.addEventListener("click", () => {
    state.tmapIncludeAll = !state.tmapIncludeAll;
    dom.btnTmapMode.textContent = state.tmapIncludeAll ? "All changes" : "Replacements only";
    dom.btnTmapMode.classList.toggle("is-active", state.tmapIncludeAll);

    // Re-render from cached server response or client-side diff
    if (state.lastTmapResponse) {
      _renderTmapFromServer(state.lastTmapResponse);
    } else if (state.lastDiff) {
      _renderTmapClientSide(state.lastDiff);
    }
  });

  // ── Copy as TSV ────────────────────────────────────────────────── //
  // Copies the tmap table as tab-separated values.  Includes the
  // INDICATORS column when server data is available (3-column output).
  dom.btnTmapCopy.addEventListener("click", () => {
    const rows = _getTmapRows();
    if (rows.length === 0) {
      setStatus("No transformation rows to copy.");
      return;
    }

    const hasIndicators = state.lastTmapResponse !== null;
    const header = hasIndicators ? "REMOVED\tADDED\tINDICATORS" : "REMOVED\tADDED";
    const lines = [header];
    for (const row of rows) {
      let line = `${row.removed || "\u2014"}\t${row.added || "\u2014"}`;
      if (hasIndicators) {
        const indicators = (row.indicators || []).join(", ") || "\u2014";
        line += `\t${indicators}`;
      }
      lines.push(line);
    }

    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      dom.btnTmapCopy.textContent = "Copied";
      setTimeout(() => { dom.btnTmapCopy.textContent = "Copy TSV"; }, 1200);
    });
  });

  // ── Copy as Markdown ───────────────────────────────────────────── //
  // Copies the tmap table as a GFM Markdown table.  Includes the
  // Indicators column when server data is available (3-column output).
  dom.btnTmapCopyMd.addEventListener("click", () => {
    const rows = _getTmapRows();
    if (rows.length === 0) {
      setStatus("No transformation rows to copy.");
      return;
    }

    const hasIndicators = state.lastTmapResponse !== null;
    const lines = hasIndicators
      ? ["| Removed | Added | Indicators |", "| --- | --- | --- |"]
      : ["| Removed | Added |", "| --- | --- |"];

    for (const row of rows) {
      const removed = (row.removed || "\u2014").replace(/\|/g, "\\|");
      const added   = (row.added   || "\u2014").replace(/\|/g, "\\|");
      if (hasIndicators) {
        const indicators = ((row.indicators || []).join(", ") || "\u2014").replace(/\|/g, "\\|");
        lines.push(`| ${removed} | ${added} | ${indicators} |`);
      } else {
        lines.push(`| ${removed} | ${added} |`);
      }
    }

    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      dom.btnTmapCopyMd.textContent = "Copied";
      setTimeout(() => { dom.btnTmapCopyMd.textContent = "Copy MD"; }, 1200);
    });
  });
}

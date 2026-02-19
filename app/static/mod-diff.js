/**
 * mod-diff.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Word-level diff visualisation, signal isolation, and transformation map.
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
 *   3. **Transformation map** (updateTransformationMap) — client-side
 *      clause-level grouping of the LCS diff into a REMOVED | ADDED table.
 *      Uses extractTransformationRows() from mod-utils.
 *
 * Data flow
 * ─────────
 *   generate() → state.current updated → updateDiff()
 *                                          ├─ updateSignalIsolation()
 *                                          └─ updateTransformationMap()
 *
 * Imports: mod-state, mod-utils, mod-status
 */

import { state, dom } from "./mod-state.js";
import { tokenise, lcsWordDiff, extractTransformationRows, makePlaceholder } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";

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
 * Build and render the Transformation Map table from a word-level LCS diff.
 *
 * Extracts clause-level rows via `extractTransformationRows()` and renders
 * them as a two-column HTML table (REMOVED | ADDED) inside the
 * `#tmap-panel` element.  Purely client-side — no server API call needed
 * because the diff is already computed by `lcsWordDiff()` in `updateDiff()`.
 *
 * Called in two contexts:
 *   1. At the end of `updateDiff()` after each generation (with the
 *      freshly computed diff).
 *   2. When the user clicks the mode toggle button (with `state.lastDiff`,
 *      the cached diff from the last `updateDiff()` call).
 *
 * Table structure:
 *   - REMOVED column: text from baseline A (red), or em dash for pure inserts.
 *   - ADDED column: text from current B (green), or em dash for pure deletes.
 *
 * @param {Array<[string, string]>} diff - Output of lcsWordDiff(): an array
 *   of [operation, word] tuples.
 */
export function updateTransformationMap(diff) {
  // Guard: no diff data available
  if (!diff || diff.length === 0) {
    dom.tmapPanel.textContent = "";
    dom.tmapPanel.appendChild(
      makePlaceholder("Set a baseline (A) and generate to see clause-level substitutions.")
    );
    return;
  }

  const rows = extractTransformationRows(diff, state.tmapIncludeAll);

  const fragment = document.createDocumentFragment();

  if (rows.length === 0) {
    fragment.appendChild(makePlaceholder("No clause-level substitutions detected."));
  } else {
    // ── Two-column table: REMOVED | ADDED ────────────────────────── //
    const table = document.createElement("table");
    table.className = "tmap-table";

    // Header row
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    const thRemoved = document.createElement("th");
    thRemoved.className = "tmap-header";
    thRemoved.textContent = "REMOVED";
    headerRow.appendChild(thRemoved);

    const thAdded = document.createElement("th");
    thAdded.className = "tmap-header";
    thAdded.textContent = "ADDED";
    headerRow.appendChild(thAdded);

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Data rows
    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");

      // REMOVED cell: text from A, or em dash for pure inserts
      const tdRemoved = document.createElement("td");
      tdRemoved.className = "tmap-cell tmap-cell--removed";
      if (row.removed) {
        tdRemoved.textContent = row.removed;
      } else {
        tdRemoved.textContent = "\u2014";  // em dash
        tdRemoved.classList.add("tmap-cell--empty");
      }
      tr.appendChild(tdRemoved);

      // ADDED cell: text from B, or em dash for pure deletes
      const tdAdded = document.createElement("td");
      tdAdded.className = "tmap-cell tmap-cell--added";
      if (row.added) {
        tdAdded.textContent = row.added;
      } else {
        tdAdded.textContent = "\u2014";  // em dash
        tdAdded.classList.add("tmap-cell--empty");
      }
      tr.appendChild(tdAdded);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    fragment.appendChild(table);
  }

  dom.tmapPanel.textContent = "";
  dom.tmapPanel.appendChild(fragment);

  // Auto-open the Transformation Map <details> on first population
  const tmapDetailsEl = document.getElementById("tmap-details");
  if (tmapDetailsEl && !tmapDetailsEl.open) {
    tmapDetailsEl.open = true;
  }
}

/**
 * Wire diff-related event listeners.
 *
 * Registers handlers for:
 *   - Transformation Map mode toggle (replacements only ↔ all changes)
 *   - Copy TSV button (tab-separated clipboard copy)
 *   - Copy MD button (GitHub-Flavoured Markdown table clipboard copy)
 *
 * All three handlers operate on `state.lastDiff` (the cached LCS diff)
 * and re-render or copy without making any API call.
 *
 * Called once during startup by the mod-events coordinator.
 */
export function wireDiffEvents() {
  // ── Transformation Map mode toggle ─────────────────────────────── //
  // Switches between "Replacements only" (default) and "All changes"
  // (includes pure inserts and pure deletes).  Re-renders instantly
  // from the cached diff without any server call.
  dom.btnTmapMode.addEventListener("click", () => {
    state.tmapIncludeAll = !state.tmapIncludeAll;
    dom.btnTmapMode.textContent = state.tmapIncludeAll ? "All changes" : "Replacements only";
    dom.btnTmapMode.classList.toggle("is-active", state.tmapIncludeAll);
    if (state.lastDiff) {
      updateTransformationMap(state.lastDiff);
    }
  });

  // ── Copy as TSV ────────────────────────────────────────────────── //
  // Copies the tmap table as tab-separated values.  TSV is chosen over
  // CSV because it pastes cleanly into spreadsheets and avoids quoting
  // issues with commas in descriptive text.
  dom.btnTmapCopy.addEventListener("click", () => {
    if (!state.lastDiff) {
      setStatus("Nothing to copy \u2013 generate a diff first.");
      return;
    }
    const rows = extractTransformationRows(state.lastDiff, state.tmapIncludeAll);
    if (rows.length === 0) {
      setStatus("No transformation rows to copy.");
      return;
    }

    const lines = ["REMOVED\tADDED"];
    for (const row of rows) {
      lines.push(`${row.removed || "\u2014"}\t${row.added || "\u2014"}`);
    }

    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      dom.btnTmapCopy.textContent = "Copied";
      setTimeout(() => { dom.btnTmapCopy.textContent = "Copy TSV"; }, 1200);
    });
  });

  // ── Copy as Markdown ───────────────────────────────────────────── //
  // Copies the tmap table as a GFM Markdown table.  Pipe characters
  // inside cell text are escaped as \| to prevent column separator
  // misinterpretation.
  dom.btnTmapCopyMd.addEventListener("click", () => {
    if (!state.lastDiff) {
      setStatus("Nothing to copy \u2013 generate a diff first.");
      return;
    }
    const rows = extractTransformationRows(state.lastDiff, state.tmapIncludeAll);
    if (rows.length === 0) {
      setStatus("No transformation rows to copy.");
      return;
    }

    const lines = [
      "| Removed | Added |",
      "| --- | --- |",
    ];
    for (const row of rows) {
      const removed = (row.removed || "\u2014").replace(/\|/g, "\\|");
      const added   = (row.added   || "\u2014").replace(/\|/g, "\\|");
      lines.push(`| ${removed} | ${added} |`);
    }

    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      dom.btnTmapCopyMd.textContent = "Copied";
      setTimeout(() => { dom.btnTmapCopyMd.textContent = "Copy MD"; }, 1200);
    });
  });
}

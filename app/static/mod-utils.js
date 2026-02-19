/**
 * mod-utils.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Pure utility functions — zero DOM or state dependencies.
 *
 * This module is a dependency leaf: it imports nothing and is imported by
 * nearly every other module.  All functions are side-effect-free and operate
 * only on their arguments, making them straightforward to reason about and
 * to test in isolation.
 *
 * Exports
 * ───────
 *   clamp              – Numeric range clamping
 *   debounce           – Delay-based call coalescing
 *   safeParse          – Non-throwing JSON.parse wrapper
 *   tokenise           – Whitespace-based word tokeniser
 *   lcsWordDiff        – Word-level LCS diff (O(m*n) DP)
 *   extractTransformationRows – Clause-level grouping of diff ops
 *   cryptoRandomFloat  – CSPRNG float in [0, 1]
 *   makePlaceholder    – Safe placeholder <span> factory
 */

/**
 * Clamp a numeric value to the closed interval [min, max].
 *
 * Used by slider normalisation (axis scores must stay in [0, 1]) and by
 * temperature sync (clamped to [0, 2]).
 *
 * @param {number} val - The value to clamp.
 * @param {number} min - Lower bound (inclusive).
 * @param {number} max - Upper bound (inclusive).
 * @returns {number} The clamped value.
 */
export function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

/**
 * Debounce: delay invoking `fn` until `ms` milliseconds have passed since
 * the last call.  Used to avoid rebuilding sliders on every keystroke in
 * the JSON textarea and to throttle Ollama host URL refreshes.
 *
 * Returns a wrapper function that resets its internal timer on each
 * invocation.  The original `this` context and arguments are forwarded
 * to `fn` when it finally fires.
 *
 * @param {Function} fn   - Function to debounce.
 * @param {number}   ms   - Delay in milliseconds.
 * @returns {Function} Debounced wrapper.
 */
export function debounce(fn, ms) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

/**
 * Attempt to parse a JSON string.  Returns the parsed object on success or
 * null on failure.  Never throws.
 *
 * Used by the JSON textarea input handler to silently detect syntax errors
 * without disrupting the user's typing flow.
 *
 * @param {string} str - Raw JSON string.
 * @returns {object|null} Parsed object, or null if the string is not valid JSON.
 */
export function safeParse(str) {
  try {
    return JSON.parse(str);
  } catch {
    return null;
  }
}

/**
 * Split text into word tokens by whitespace.  Punctuation stays attached
 * to the word it adjoins (e.g. "hello," remains one token).  Consecutive
 * whitespace is collapsed; leading/trailing whitespace is stripped.
 *
 * This is the tokeniser shared by the diff algorithm and the
 * transformation map extraction — it must stay consistent across both
 * so that word indices align.
 *
 * @param {string} text - Input text to tokenise.
 * @returns {string[]} Array of non-empty word tokens.
 */
export function tokenise(text) {
  return text.trim().split(/\s+/).filter(Boolean);
}

/**
 * Compute a word-level LCS-based diff between two token arrays.
 *
 * Returns an array of [operation, word] tuples where operation is:
 *   "=" – word is in both A and B (common / unchanged)
 *   "+" – word is only in B (added)
 *   "-" – word is only in A (removed)
 *
 * Algorithm
 * ─────────
 * Uses the standard dynamic-programming Longest Common Subsequence (LCS)
 * approach in two phases:
 *
 *   1. **Forward pass** — Build an (m+1)×(n+1) table where dp[i][j]
 *      holds the LCS length of a[0..i-1] and b[0..j-1].  Time: O(m*n).
 *      Space: O(m*n) via Int32Array rows (more memory-efficient than
 *      nested plain arrays).
 *
 *   2. **Backtrack** — Walk from dp[m][n] to dp[0][0], emitting "=",
 *      "+", or "-" ops depending on which cell we came from.  The
 *      result is built in reverse and then flipped.
 *
 * For paragraph-length LLM outputs (≤ 200 words) this is always
 * sub-millisecond.  For very long texts it could be slow, but that is
 * outside the tool's intended use case.
 *
 * @param {string[]} a - Token array for text A (baseline).
 * @param {string[]} b - Token array for text B (current).
 * @returns {Array<[string, string]>} Ordered diff tuples.
 */
export function lcsWordDiff(a, b) {
  const m = a.length;
  const n = b.length;

  // ── Phase 1: Build LCS length table ────────────────────────────────── //
  // dp[i][j] = length of LCS of a[0..i-1] and b[0..j-1].
  // Int32Array gives a contiguous typed-array row, reducing GC pressure
  // compared to nested Array-of-Array.
  const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;     // match → extend LCS
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);  // skip one side
      }
    }
  }

  // ── Phase 2: Backtrack to reconstruct diff ─────────────────────────── //
  // Walk from (m, n) to (0, 0).  At each step:
  //   - diagonal move with match → "=" (common word)
  //   - move left (j--)         → "+" (word added in B)
  //   - move up   (i--)         → "-" (word removed from A)
  const result = [];
  let i = m, j = n;

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      result.push(["=", a[i - 1]]);
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      result.push(["+", b[j - 1]]);
      j--;
    } else {
      result.push(["-", a[i - 1]]);
      i--;
    }
  }

  result.reverse();
  return result;
}

/**
 * Extract clause-level replacement rows from a word-level LCS diff.
 *
 * Groups consecutive removed ("-") and added ("+") words between runs of
 * equal ("=") words into clause-level rows.  Each row represents a single
 * contiguous region of change — a clause-scale substitution, insertion,
 * or deletion.
 *
 * Algorithm
 * ─────────
 * Walk the diff tuples linearly.  Two buffers accumulate "-" and "+"
 * words respectively.  When an "=" word is encountered (or the end of
 * the diff is reached), the buffers are flushed into a row and cleared.
 *
 * In "replacements only" mode (`includeAll=false`), rows where one side
 * is empty are silently discarded — these represent pure insertions or
 * deletions with no counterpart text.  In "all changes" mode, every
 * non-empty flush is emitted.
 *
 * @param {Array<[string, string]>} diff - Output of lcsWordDiff(): an
 *   array of [operation, word] tuples where operation is "=", "+", or "-".
 * @param {boolean} includeAll - When true, include pure inserts (removed="")
 *   and pure deletes (added="") as rows.  When false, only rows where both
 *   the removed and added sides are non-empty are returned.
 * @returns {Array<{removed: string, added: string}>} Ordered list of
 *   clause-level change rows.
 */
export function extractTransformationRows(diff, includeAll) {
  const rows = [];
  let removedBuf = [];   // accumulates consecutive "-" words
  let addedBuf   = [];   // accumulates consecutive "+" words

  /** Flush current buffers into a row (if non-empty). */
  function flush() {
    if (removedBuf.length === 0 && addedBuf.length === 0) return;

    const removed = removedBuf.join(" ");
    const added   = addedBuf.join(" ");

    // In "replacements only" mode, skip rows where one side is empty.
    if (includeAll || (removed && added)) {
      rows.push({ removed, added });
    }

    removedBuf = [];
    addedBuf   = [];
  }

  for (const [op, word] of diff) {
    if (op === "=") {
      flush();               // equal word = boundary between change regions
    } else if (op === "-") {
      removedBuf.push(word); // word present in A but absent from B
    } else if (op === "+") {
      addedBuf.push(word);   // word present in B but absent from A
    }
  }

  flush();  // handle any trailing change region not terminated by "="

  return rows;
}

/**
 * Generate a single random float in [0, 1] using the Web Crypto API.
 *
 * Uses a 32-bit unsigned integer from `crypto.getRandomValues()` divided
 * by 2^32 - 1 (4294967295) to produce a uniformly distributed float.
 *
 * This avoids `Math.random()` entirely, ensuring no shared RNG state with
 * any other part of the system.  The payload seed (used for Ollama's token
 * sampling) is completely unaffected by calls to this function.
 *
 * @returns {number} A random float in the closed interval [0, 1].
 */
export function cryptoRandomFloat() {
  const buf = new Uint32Array(1);
  crypto.getRandomValues(buf);
  return buf[0] / 4294967295;
}

/**
 * Create a styled placeholder `<span>` element.
 *
 * Used instead of `innerHTML` to avoid any XSS surface when rendering
 * placeholder text inside output/diff panels.  Sets `textContent` (not
 * `innerHTML`) so the message is always treated as plain text.
 *
 * @param {string} text - The placeholder message to display.
 * @returns {HTMLSpanElement} A `<span class="placeholder-text">` element.
 */
export function makePlaceholder(text) {
  const span = document.createElement("span");
  span.className = "placeholder-text";
  span.textContent = text;
  return span;
}

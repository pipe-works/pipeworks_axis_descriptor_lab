/**
 * mod-state.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Application state singleton and cached DOM references.
 *
 * Design decision: ES module singleton
 * ─────────────────────────────────────
 * ES modules export **live bindings** — every module that imports `state`
 * or `dom` receives the same object reference.  Mutations performed by
 * one module (e.g. `state.current = data.text` in mod-generate.js) are
 * immediately visible in every other module that holds a reference.
 *
 * This preserves the exact same shared-mutable-state semantics as the
 * original monolithic app.js, without introducing a pub/sub layer or
 * event bus.
 *
 * Design decision: `dom` object
 * ─────────────────────────────
 * All 40 cached DOM element references are properties of a single
 * exported `dom` object.  Consuming modules access them as `dom.outputBox`
 * rather than bare `outputBox`, making the DOM dependency explicit at
 * every call site.  `type="module"` scripts are deferred by default, so
 * the DOM is guaranteed to be fully parsed when these refs are captured.
 *
 * Imports: none (leaf module)
 */

/**
 * Application-level mutable state.
 *
 * All properties start as null/false and are populated during the
 * application lifecycle.  The object is mutated in place — never
 * reassigned — so all importers always see the latest values.
 *
 * @type {{
 *   payload:           object|null,
 *   baseline:          string|null,
 *   current:           string|null,
 *   busy:              boolean,
 *   originalAxes:      object|null,
 *   lastMeta:          object|null,
 *   baselineMeta:      object|null,
 *   tmapIncludeAll:    boolean,
 *   lastDiff:          Array|null,
 *   lastSaveFolderName: string|null
 * }}
 */
export const state = {
  /** Current AxisPayload JS object (mirrors the JSON textarea). */
  payload:  null,

  /** Baseline (A) generated text for diffing.  Set by "Set as A". */
  baseline: null,

  /** Latest (B) generated text from the most recent /api/generate call. */
  current:  null,

  /** True while a generate request is in-flight (prevents double-submit). */
  busy:     false,

  /**
   * Deep copy of the axes from the most recently loaded example.
   * Used to detect which scores/labels the user has modified since loading.
   * Null until an example is loaded; reset on each loadExample() call.
   */
  originalAxes: null,

  /**
   * Metadata from the most recent generation.  Plain object mapping
   * meta row keys to display values (e.g. { "input": "a88b…", "ipc": "2ee9…" }).
   * Snapshotted into baselineMeta when "Set as A" is clicked.
   * Null before the first generation.
   */
  lastMeta: null,

  /**
   * Snapshot of lastMeta from the generation that was set as baseline (A).
   * Used to highlight which meta rows have changed in subsequent generations.
   * Null until "Set as A" is clicked.
   */
  baselineMeta: null,

  /** Transformation Map mode: false = replacements only, true = all changes. */
  tmapIncludeAll: false,

  /**
   * Cached LCS diff from the last updateDiff() call, so the tmap toggle
   * can re-render without recomputing the diff.  Null before first diff.
   */
  lastDiff: null,

  /**
   * Folder name returned by the most recent successful save.  Used by
   * exportSave() to construct the export URL.  Null before the first save;
   * set in saveRun() on success.  Enables the Export Zip button.
   */
  lastSaveFolderName: null,
};

/**
 * Short-hand for `document.getElementById`.
 *
 * Used only during module initialisation to populate the `dom` cache.
 * Not intended for general use — prefer `dom.<name>` for cached lookups.
 *
 * @param {string} id - The element ID to look up.
 * @returns {HTMLElement|null} The element, or null if not found.
 */
export const $ = (id) => document.getElementById(id);

/**
 * Cached DOM element references, queried once at module load time.
 *
 * Because `<script type="module">` is deferred by default, the DOM is
 * guaranteed to be fully parsed when this code executes.  Every other
 * module imports `dom` to access these references, avoiding repeated
 * `document.getElementById` calls throughout the application.
 *
 * @type {Object<string, HTMLElement|null>}
 */
export const dom = {
  // ── Left panel: payload editor ────────────────────────────────────── //
  exampleSelect:        $("example-select"),
  btnLoadExample:       $("btn-load-example"),
  jsonTextarea:         $("json-textarea"),
  jsonStatusBadge:      $("json-status-badge"),
  systemPromptTextarea: $("system-prompt-textarea"),
  systemPromptBadge:    $("system-prompt-badge"),
  promptSelect:         $("prompt-select"),
  btnLoadPrompt:        $("btn-load-prompt"),

  // ── Centre panel: axis controls ───────────────────────────────────── //
  sliderPanel:          $("slider-panel"),
  autoLabelToggle:      $("auto-label-toggle"),
  btnRelabel:           $("btn-relabel"),
  btnRandomise:         $("btn-randomise"),
  ollamaHostInput:      $("ollama-host-input"),
  modelSelect:          $("model-select"),
  modelInput:           $("model-input"),
  tempRange:            $("temp-range"),
  tempInput:            $("temp-input"),
  tokensInput:          $("tokens-input"),
  seedInput:            $("seed-input"),
  btnGenerate:          $("btn-generate"),
  btnSetBaseline:       $("btn-set-baseline"),

  // ── Centre panel: A/B diff boxes ──────────────────────────────────── //
  diffA:                $("diff-a"),
  diffB:                $("diff-b"),
  tmapPanel:            $("tmap-panel"),
  btnTmapMode:          $("btn-tmap-mode"),
  btnTmapCopy:          $("btn-tmap-copy"),
  btnTmapCopyMd:        $("btn-tmap-copy-md"),

  // ── Right panel: output + diff ────────────────────────────────────── //
  outputBox:            $("output-box"),
  outputMeta:           $("output-meta"),
  btnSave:              $("btn-save"),
  btnClearOutput:       $("btn-clear-output"),
  btnExport:            $("btn-export"),
  btnImport:            $("btn-import"),
  importFileInput:      $("import-file-input"),
  diffDelta:            $("diff-delta"),
  diffPct:              $("diff-pct"),
  signalPanel:          $("signal-panel"),

  // ── Status bar ────────────────────────────────────────────────────── //
  statusText:           $("status-text"),
  spinner:              $("spinner"),
};

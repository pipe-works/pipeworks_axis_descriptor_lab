/**
 * mod-navigation.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Page switching between Character Description, Chat Translation, and the
 * Artifact Editor, and Pipeline Build.
 *
 * Layout model
 * ────────────
 * The application has three top-level "pages" that share a single HTML document.
 * They live as sibling <div> elements below the shared header + nav bar:
 *
 *   <div id="page-char-description">   ← Character Description page
 *   <div id="page-chat-translation">   ← Chat Translation page
 *   <div id="page-artifact-editor">    ← Artifact Editor page
 *   <div id="page-pipeline-build">     ← Pipeline Build page
 *
 * Only one page is visible at a time.  Switching is done by toggling the
 * "hidden" CSS class on the page divs (display:none / display:contents) and
 * the "is-active" class on the corresponding nav button.
 *
 * No state is persisted across page switches — each page retains whatever
 * DOM state it had when the user left it, so in-progress work is not lost.
 *
 * Imports: mod-state (dom refs for page divs and nav buttons)
 */

import { dom } from "./mod-state.js";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Activate one page and deactivate the rest.
 *
 * Shows the requested page and hides every other page by toggling the
 * `"hidden"` class.  Marks the matching nav button as active and clears
 * the active state from the others.
 *
 * @param {string} activeKey - Page key to activate.
 */
function switchPage(activeKey) {
  const pages = {
    char: dom.pageCharDescription,
    chat: dom.pageChatTranslation,
    artifact: dom.pageArtifactEditor,
    pipeline: dom.pagePipelineBuild,
  };
  const buttons = {
    char: dom.navCharDesc,
    chat: dom.navChatTrans,
    artifact: dom.navArtifactEditor,
    pipeline: dom.navPipelineBuild,
  };

  for (const [key, page] of Object.entries(pages)) {
    if (!page) continue;
    page.classList.toggle("hidden", key !== activeKey);
  }
  for (const [key, button] of Object.entries(buttons)) {
    if (!button) continue;
    button.classList.toggle("is-active", key === activeKey);
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Wire the navigation bar button event listeners.
 *
 * Attaches "click" handlers to the three nav buttons:
 *   - "Character Description" (#nav-char-desc) → shows #page-char-description
 *   - "Chat Translation"      (#nav-chat-trans) → shows #page-chat-translation
 *   - "Artifact Editor"       (#nav-artifact-editor) → shows #page-artifact-editor
 *   - "Pipeline Build"        (#nav-pipeline-build) → shows #page-pipeline-build
 *
 * Called once during startup by the mod-events coordinator
 * ({@link module:mod-events~wireEvents}).
 *
 * @returns {void}
 */
export function wireNavigationEvents() {
  // ── Character Description nav button ────────────────────────────────── //
  dom.navCharDesc.addEventListener("click", () => {
    switchPage("char");
  });

  // ── Chat Translation nav button ──────────────────────────────────────── //
  dom.navChatTrans.addEventListener("click", () => {
    switchPage("chat");
    // Notify the chat translation module so it can re-check session state
    // (prevents stale auth after returning from the Character Description page).
    document.dispatchEvent(new CustomEvent("chat-translation-activated"));
  });

  // ── Artifact Editor nav button ───────────────────────────────────────── //
  dom.navArtifactEditor.addEventListener("click", () => {
    switchPage("artifact");
    document.dispatchEvent(new CustomEvent("artifact-editor-activated"));
  });

  // ── Pipeline Build nav button ───────────────────────────────────────── //
  dom.navPipelineBuild.addEventListener("click", () => {
    switchPage("pipeline");
    document.dispatchEvent(new CustomEvent("pipeline-build-activated"));
  });
}

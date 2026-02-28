/**
 * mod-navigation.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Page switching between Character Description and Chat Translation.
 *
 * Layout model
 * ────────────
 * The application has two top-level "pages" that share a single HTML document.
 * They live as sibling <div> elements below the shared header + nav bar:
 *
 *   <div id="page-char-description">   ← Character Description page
 *   <div id="page-chat-translation">   ← Chat Translation page
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
 * Activate one page and deactivate the other.
 *
 * Shows `activePage` and hides `inactivePage` by toggling the `"hidden"`
 * class.  Marks `activeBtn` as the current nav selection by adding the
 * `"is-active"` class and removing it from `inactiveBtn`.
 *
 * Centralising this logic avoids the two click handlers duplicating the
 * same four class-list mutations.
 *
 * @param {HTMLElement} activePage   - The page div to show.
 * @param {HTMLElement} inactivePage - The page div to hide.
 * @param {HTMLElement} activeBtn    - Nav button to mark as active.
 * @param {HTMLElement} inactiveBtn  - Nav button to mark as inactive.
 */
function switchPage(activePage, inactivePage, activeBtn, inactiveBtn) {
  activePage.classList.remove("hidden");
  inactivePage.classList.add("hidden");
  activeBtn.classList.add("is-active");
  inactiveBtn.classList.remove("is-active");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Wire the navigation bar button event listeners.
 *
 * Attaches "click" handlers to the two nav buttons:
 *   - "Character Description" (#nav-char-desc) → shows #page-char-description
 *   - "Chat Translation"      (#nav-chat-trans) → shows #page-chat-translation
 *
 * Called once during startup by the mod-events coordinator
 * ({@link module:mod-events~wireEvents}).
 *
 * @returns {void}
 */
export function wireNavigationEvents() {
  // ── Character Description nav button ────────────────────────────────── //
  dom.navCharDesc.addEventListener("click", () => {
    switchPage(
      dom.pageCharDescription, // show
      dom.pageChatTranslation,  // hide
      dom.navCharDesc,          // mark active
      dom.navChatTrans,         // mark inactive
    );
  });

  // ── Chat Translation nav button ──────────────────────────────────────── //
  dom.navChatTrans.addEventListener("click", () => {
    switchPage(
      dom.pageChatTranslation,  // show
      dom.pageCharDescription,  // hide
      dom.navChatTrans,         // mark active
      dom.navCharDesc,          // mark inactive
    );
    // Notify the chat translation module so it can re-check session state
    // (prevents stale auth after returning from the Character Description page).
    document.dispatchEvent(new CustomEvent("chat-translation-activated"));
  });
}

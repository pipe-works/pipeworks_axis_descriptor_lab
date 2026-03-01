/**
 * mod-chat-state.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Shared mutable state and DOM helpers for the Chat Translation page.
 *
 * This module is the leaf state owner for the chat page.  Extracting it keeps
 * the other chat modules acyclic: they all depend on this state module rather
 * than importing each other or reaching back into the page controller.
 *
 * Imports: mod-state
 */

import { dom } from "./mod-state.js";

/**
 * Per-character mutable state for the Chat Translation page.
 *
 * Kept separate from `mod-state.js#state` so that switching pages does not
 * accidentally corrupt the Character Description page's state.
 */
export const chatState = {
  a: { payload: null, originalAxes: null, activeAxes: null },
  b: { payload: null, originalAxes: null, activeAxes: null },
  busy: false,
  liveMode: false,
  /**
   * When true, the Ollama host URL from the settings input is included
   * in translation request bodies as ``ollama_host``.
   *
   * @type {boolean}
   */
  useAddress: false,
  logSeq: 0,
  /** @type {"standalone"|"server-prod"|"server-local"} */
  translationMode: "standalone",
  authenticated: false,
  /** @type {{id: string, name: string}[]} */
  worlds: [],
  worldId: null,
  worldConfig: null,
  worldConfigLoading: false,
  /** @type {{filename: string, content: string, is_active: boolean}[]} */
  worldPrompts: [],
  serverPromptOriginal: "",
  /**
   * @type {{
   *   ch: string,
   *   channel: string,
   *   oocMessage: string,
   *   icText: string|null,
   *   model: string,
   *   status: string,
   *   errorDetail: string|null,
   *   sentAt: string|null,
   *   durationMs: number|null,
   *   ipcId: string|null,
   *   inputHash: string|null,
   *   systemPromptHash: string|null,
   *   outputHash: string|null
   * }[]}
   */
  gameLog: [],
};

/**
 * Return the DOM ref bundle for character `ch`.
 *
 * Abstracts away the `chatA*` / `chatB*` naming difference so that all
 * character-agnostic functions can operate on a single object regardless
 * of which character they are processing.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Object} The DOM refs for the requested character panel.
 */
export function charDom(ch) {
  return ch === "a"
    ? {
        exampleSelect:  dom.chatAExampleSelect,
        btnLoadExample: dom.chatABtnLoadExample,
        jsonTextarea:   dom.chatAJson,
        jsonBadge:      dom.chatAJsonBadge,
        sliderPanel:    dom.chatASliderPanel,
        btnRandomise:   dom.chatABtnRandomise,
        autoLabel:      dom.chatAAutoLabel,
        btnRelabel:     dom.chatABtnRelabel,
        oocTextarea:    dom.chatAOoc,
        channelSelect:  dom.chatAChannel,
        btnSend:        dom.chatABtnSend,
      }
    : {
        exampleSelect:  dom.chatBExampleSelect,
        btnLoadExample: dom.chatBBtnLoadExample,
        jsonTextarea:   dom.chatBJson,
        jsonBadge:      dom.chatBJsonBadge,
        sliderPanel:    dom.chatBSliderPanel,
        btnRandomise:   dom.chatBBtnRandomise,
        autoLabel:      dom.chatBAutoLabel,
        btnRelabel:     dom.chatBBtnRelabel,
        oocTextarea:    dom.chatBOoc,
        channelSelect:  dom.chatBChannel,
        btnSend:        dom.chatBBtnSend,
      };
}

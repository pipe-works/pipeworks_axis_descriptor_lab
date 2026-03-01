/**
 * mod-chat-translation.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Chat Translation page controller.
 *
 * This module now owns only orchestration concerns:
 * - loading examples and standalone IC prompts
 * - relabel/randomise flows
 * - request construction for translate/send actions
 * - batch/live translation result handling
 * - top-level event wiring and startup
 *
 * State, slider construction, server-mode auth/prompt handling, game-log
 * persistence, and import/restore logic live in dedicated modules so this
 * controller can stay focused on page-level coordination.
 *
 * Imports: mod-state, mod-utils, mod-status, mod-chat-state,
 *          mod-chat-sliders, mod-chat-server-mode, mod-chat-game-log,
 *          mod-chat-import
 */

import { dom } from "./mod-state.js";
import { clamp, debounce, safeParse, cryptoRandomFloat } from "./mod-utils.js";
import { setStatus } from "./mod-status.js";
import { chatState, charDom } from "./mod-chat-state.js";
import { buildChatSliders, setJsonBadge, syncJsonTextarea } from "./mod-chat-sliders.js";
import {
  checkSession,
  getCurrentSystemPromptText,
  getEffectiveSystemPrompt,
  handleSessionExpired,
  isServerMode,
  updateModeBadge,
  wireServerModeEvents,
} from "./mod-chat-server-mode.js";
import {
  appendGameEntry,
  renderTranslationResult,
  wireGameLogEvents,
} from "./mod-chat-game-log.js";
import { wireChatImportEvents } from "./mod-chat-import.js";

/**
 * Timeout for `/api/translate_chat` requests.
 *
 * This is intentionally aligned with the backend's HTTP client timeout so the
 * browser does not give up materially earlier than the server.
 *
 * @type {number}
 */
const TRANSLATE_TIMEOUT_MS = 120_000;

/**
 * Fetch and load an example payload for character `ch`.
 *
 * Loading an example resets `activeAxes` to `null`, which is the module's
 * sentinel meaning "all axes currently enabled".  The slider module then
 * resolves that sentinel to the concrete set of keys on rebuild.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @param {string} name - Example name returned by `/api/examples`.
 * @returns {Promise<void>}
 */
async function loadChatExample(ch, name) {
  if (!name) return;

  try {
    const res = await fetch(`/api/examples/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    chatState[ch].payload = data;
    chatState[ch].originalAxes = JSON.parse(JSON.stringify(data.axes || {}));
    chatState[ch].activeAxes = null;

    syncJsonTextarea(ch);
    setJsonBadge(ch, true);
    buildChatSliders(ch);
    setStatus(`Character ${ch.toUpperCase()} — loaded "${name}".`);
  } catch (err) {
    setStatus(`Error loading example for ${ch.toUpperCase()}: ${err.message}`);
  }
}

/**
 * Populate both example dropdowns from `/api/examples`.
 *
 * The placeholder option is preserved for each select; only dynamically
 * loaded options are replaced.
 *
 * @returns {Promise<void>}
 */
async function loadChatExampleList() {
  try {
    const res = await fetch("/api/examples");
    if (!res.ok) return;

    const names = await res.json();
    for (const ch of ["a", "b"]) {
      const select = charDom(ch).exampleSelect;
      while (select.options.length > 1) select.remove(1);

      for (const name of names) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      }
    }
  } catch {
    // Ignore startup fetch failures so the rest of the page can still render.
  }
}

/**
 * Recompute labels for a character payload via `/api/relabel`.
 *
 * The slider module preserves the existing `activeAxes` set across the
 * rebuild, so relabeling does not silently re-enable disabled axes.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Promise<void>}
 */
async function relabelChatChar(ch) {
  if (!chatState[ch].payload) {
    setStatus(`No payload to relabel for Character ${ch.toUpperCase()}.`);
    return;
  }

  setStatus(`Recomputing labels for Character ${ch.toUpperCase()}…`, true);
  try {
    const res = await fetch("/api/relabel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatState[ch].payload),
    });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errorData.detail || `HTTP ${res.status}`);
    }

    chatState[ch].payload = await res.json();
    syncJsonTextarea(ch);
    buildChatSliders(ch);
    setStatus(`Character ${ch.toUpperCase()} — labels recomputed.`);
  } catch (err) {
    setStatus(`Relabel error (${ch.toUpperCase()}): ${err.message}`);
  }
}

/**
 * Randomise every axis score for a character.
 *
 * Scores are rounded to three decimal places so they stay aligned with the
 * slider step size and the existing JSON/UI presentation.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Promise<void>}
 */
async function randomiseChatChar(ch) {
  const payload = chatState[ch].payload;
  if (!payload || !payload.axes) {
    setStatus(`No payload to randomise for Character ${ch.toUpperCase()}.`);
    return;
  }

  for (const axisKey of Object.keys(payload.axes)) {
    const newScore = Math.round(cryptoRandomFloat() * 1000) / 1000;
    payload.axes[axisKey] = { ...payload.axes[axisKey], score: newScore };
  }

  syncJsonTextarea(ch);
  buildChatSliders(ch);

  if (charDom(ch).autoLabel.checked) {
    await relabelChatChar(ch);
  } else {
    setStatus(`Character ${ch.toUpperCase()} — scores randomised.`);
  }
}

/**
 * Populate the standalone IC prompt dropdown with prompt names prefixed
 * `ic_`, preserving the existing "default" placeholder option.
 *
 * @returns {Promise<void>}
 */
async function loadChatIcPromptList() {
  try {
    const res = await fetch("/api/prompts");
    if (!res.ok) return;

    const names = await res.json();
    const icNames = names.filter((name) => name.startsWith("ic_"));
    while (dom.chatPromptSelect.options.length > 1) dom.chatPromptSelect.remove(1);

    for (const name of icNames) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      dom.chatPromptSelect.appendChild(option);
    }
  } catch {
    // Ignore startup fetch failures so the rest of the page can still render.
  }
}

/**
 * Update the standalone IC prompt badge.
 *
 * Empty text means the backend should use the default prompt; non-empty text
 * means the user is overriding it inline.
 *
 * @returns {void}
 */
function updateChatPromptBadge() {
  const hasContent = dom.chatSystemPrompt.value.trim().length > 0;
  dom.chatPromptBadge.textContent = hasContent ? "custom" : "default";
  dom.chatPromptBadge.className = hasContent ? "badge badge--active" : "badge badge--muted";
}

/**
 * Resolve the current model name from the select/input pair.
 *
 * @returns {string}
 */
function getChatModelName() {
  const selected = dom.chatModelSelect.value.trim();
  return selected || dom.chatModelInput.value.trim();
}

/**
 * Read the raw Ollama host field value.
 *
 * @returns {string}
 */
function getChatOllamaHost() {
  return dom.chatOllamaHost.value.trim();
}

/**
 * Resolve the Ollama host value to include in API request bodies.
 *
 * The host is sent only when the "Use address" toggle is enabled; otherwise
 * the server uses its configured default host.
 *
 * @returns {string|null}
 */
function getChatOllamaHostForRequest() {
  if (!chatState.useAddress) return null;
  return getChatOllamaHost() || null;
}

/**
 * Dim or undim the Ollama host field based on `chatState.useAddress`.
 *
 * @returns {void}
 */
function syncUseAddressUI() {
  dom.chatOllamaHost.classList.toggle("input--dimmed", !chatState.useAddress);
}

/**
 * Resolve the current seed value using the page's `-1 means random` convention.
 *
 * @returns {number}
 */
function resolveChatSeed() {
  const raw = parseInt(dom.chatSeedInput.value, 10);
  if (isNaN(raw) || raw < 0) {
    return Math.floor(Math.random() * 0x100000000);
  }
  return raw;
}

/**
 * Refresh the Ollama model list from `/api/models`.
 *
 * On success the select is repopulated.  On failure the module falls back to
 * the free-text input so the user can still type a model tag manually.
 *
 * @param {string} [host] - Explicit host override.  Defaults to the current field value.
 * @returns {Promise<void>}
 */
async function refreshChatModels(host) {
  const effectiveHost = host || getChatOllamaHost();
  const url = effectiveHost
    ? `/api/models?host=${encodeURIComponent(effectiveHost)}`
    : "/api/models";

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const models = await res.json();
    const previous = getChatModelName();

    if (models.length > 0) {
      dom.chatModelSelect.innerHTML = "";
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        if (model === previous) option.selected = true;
        dom.chatModelSelect.appendChild(option);
      }
      dom.chatModelSelect.classList.remove("hidden");
      dom.chatModelInput.classList.add("hidden");
    } else {
      dom.chatModelSelect.classList.add("hidden");
      dom.chatModelInput.classList.remove("hidden");
    }
  } catch {
    dom.chatModelSelect.classList.add("hidden");
    dom.chatModelInput.classList.remove("hidden");
  }
}

/**
 * Build the filtered axis dictionary to send to the backend for a character.
 *
 * Disabled axes are intentionally omitted entirely so the backend profile
 * renderer behaves exactly as if those axes were absent from the payload.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Object|null}
 */
function buildAxesForRequest(ch) {
  const payload = chatState[ch].payload;
  if (!payload || !payload.axes) return null;

  const activeAxes = chatState[ch].activeAxes;
  const axes = {};
  for (const [axisName, axisValue] of Object.entries(payload.axes)) {
    if (activeAxes === null || activeAxes.has(axisName)) {
      axes[axisName] = axisValue;
    }
  }
  return axes;
}

/**
 * Synchronise the batch/live translation controls with `chatState.liveMode`.
 *
 * @returns {void}
 */
function updateLiveModeUI() {
  const live = chatState.liveMode;
  for (const ch of ["a", "b"]) {
    charDom(ch).btnSend.classList.toggle("hidden", !live);
  }
  dom.btnTranslate.disabled = live;
  dom.chatGameSection.classList.toggle("hidden", !live);
}

/**
 * Translate one character's OOC message in live mode.
 *
 * The game-log module records both successful translations and network-level
 * failures so the exported/saveable session reflects what the user actually
 * attempted to send.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {Promise<void>}
 */
async function sendForChar(ch) {
  if (chatState.busy) return;

  const model = isServerMode() ? (getChatModelName() || "(server)") : getChatModelName();
  if (!model) {
    setStatus("No model specified.");
    return;
  }

  if (isServerMode()) {
    const worldId = chatState.worldId || dom.chatWorldSelect?.value;
    if (!worldId) {
      setStatus("Please select a world before translating.");
      return;
    }
    if (chatState.worldConfigLoading) {
      setStatus("World config loading — please wait.");
      return;
    }
  }

  const axes = buildAxesForRequest(ch);
  if (!axes || Object.keys(axes).length === 0) {
    setStatus(`Load an example for Character ${ch.toUpperCase()} before sending.`);
    return;
  }

  const oocMessage = charDom(ch).oocTextarea.value.trim();
  if (!oocMessage) {
    setStatus(`Enter an OOC message for Character ${ch.toUpperCase()}.`);
    return;
  }

  const channel = charDom(ch).channelSelect.value;
  const requestSystemPrompt = getEffectiveSystemPrompt();
  const currentSystemPrompt = getCurrentSystemPromptText();
  const charInput = {
    axes,
    ooc_message: oocMessage,
    channel,
    active_axes: chatState[ch].activeAxes ? [...chatState[ch].activeAxes] : null,
  };

  const reqBody = {
    character_a: ch === "a" ? charInput : null,
    character_b: ch === "b" ? charInput : null,
    model,
    temperature: parseFloat(dom.chatTempInput.value),
    max_tokens: parseInt(dom.chatTokensInput.value, 10),
    seed: resolveChatSeed(),
    strict_mode: dom.chatStrictMode.checked,
    max_output_chars: parseInt(dom.chatMaxChars.value, 10),
    system_prompt: requestSystemPrompt,
    world_id: chatState.worldId || dom.chatWorldSelect?.value || null,
    ollama_host: getChatOllamaHostForRequest(),
  };

  chatState.busy = true;
  charDom(ch).btnSend.disabled = true;
  setStatus(`Sending ${ch.toUpperCase()} via ${model}…`, true);

  const outputBox = ch === "a" ? dom.chatAOutput : dom.chatBOutput;
  const metaDiv = ch === "a" ? dom.chatAMeta : dom.chatBMeta;
  const badge = ch === "a" ? dom.chatAStatusBadge : dom.chatBStatusBadge;
  outputBox.innerHTML = '<span class="placeholder-text">Translating…</span>';

  const abortController = new AbortController();
  const timer = setTimeout(() => abortController.abort(), TRANSLATE_TIMEOUT_MS);
  const sentAt = new Date().toISOString();
  const startTime = Date.now();

  try {
    const res = await fetch("/api/translate_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
      signal: abortController.signal,
    });
    if (!res.ok) {
      if (res.status === 401 && isServerMode()) {
        handleSessionExpired();
        return;
      }
      const errorData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errorData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const durationMs = Date.now() - startTime;
    const result = ch === "a" ? data.character_a : data.character_b;
    const otherCh = ch === "a" ? "b" : "a";
    const otherLastEntry = [...chatState.gameLog].reverse().find((entry) => entry.ch === otherCh);
    const otherSpHash = otherLastEntry?.systemPromptHash ?? null;

    renderTranslationResult(outputBox, metaDiv, badge, result, otherSpHash);

    const usedModel = (result && result.model) || model;
    if (result) {
      appendGameEntry(
        ch,
        channel,
        oocMessage,
        result.ic_text ?? null,
        usedModel,
        result.status ?? "success",
        result.error_detail ?? null,
        sentAt,
        durationMs,
        result.ipc_id ?? null,
        result.input_hash ?? null,
        result.system_prompt_hash ?? null,
        result.output_hash ?? null,
        currentSystemPrompt,
      );

      // Re-render the opposite character's IPC table when both hashes are now
      // known so the match/mismatch colour stays symmetric.
      if (result.system_prompt_hash && otherLastEntry?.systemPromptHash) {
        const otherOutput = otherCh === "a" ? dom.chatAOutput : dom.chatBOutput;
        const otherMeta = otherCh === "a" ? dom.chatAMeta : dom.chatBMeta;
        const otherBadge = otherCh === "a" ? dom.chatAStatusBadge : dom.chatBStatusBadge;
        renderTranslationResult(
          otherOutput,
          otherMeta,
          otherBadge,
          {
            status: "success",
            ic_text: otherLastEntry.icText,
            input_hash: otherLastEntry.inputHash,
            system_prompt_hash: otherLastEntry.systemPromptHash,
            output_hash: otherLastEntry.outputHash,
            ipc_id: otherLastEntry.ipcId,
          },
          result.system_prompt_hash,
        );
      }
    }

    setStatus(`${ch.toUpperCase()} sent (${usedModel}) — ${result ? result.status : "no result"}.`);
  } catch (err) {
    const durationMs = Date.now() - startTime;
    const msg = err.name === "AbortError"
      ? `Request timed out after ${TRANSLATE_TIMEOUT_MS / 1000}s — is the model loaded in Ollama?`
      : err.message;

    outputBox.innerHTML = `<span style="color:var(--col-err)">Error: ${msg}</span>`;
    appendGameEntry(
      ch,
      channel,
      oocMessage,
      null,
      model,
      "error",
      msg,
      sentAt,
      durationMs,
      null,
      null,
      null,
      null,
      currentSystemPrompt,
    );
    setStatus(`Send error (${ch.toUpperCase()}): ${msg}`);
  } finally {
    clearTimeout(timer);
    chatState.busy = false;
    charDom(ch).btnSend.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

/**
 * Run the main batch translation action.
 *
 * Character A is required.  Character B is optional and is included only when
 * both a payload and an OOC message are present.
 *
 * @returns {Promise<void>}
 */
export async function translate() {
  if (chatState.busy) return;

  const model = isServerMode() ? (getChatModelName() || "(server)") : getChatModelName();
  if (!model) {
    setStatus("No model specified for translation.");
    return;
  }

  if (isServerMode()) {
    const worldId = chatState.worldId || dom.chatWorldSelect?.value;
    if (!worldId) {
      setStatus("Please select a world before translating.");
      return;
    }
    if (chatState.worldConfigLoading) {
      setStatus("World config loading — please wait.");
      return;
    }
  }

  const axesA = buildAxesForRequest("a");
  if (!axesA || Object.keys(axesA).length === 0) {
    setStatus("Load an example for Character A before translating.");
    return;
  }

  const oocA = dom.chatAOoc.value.trim();
  if (!oocA) {
    setStatus("Enter an OOC message for Character A.");
    return;
  }

  const character_a = {
    axes: axesA,
    ooc_message: oocA,
    channel: dom.chatAChannel.value,
    active_axes: chatState.a.activeAxes ? [...chatState.a.activeAxes] : null,
  };

  let character_b = null;
  const axesB = buildAxesForRequest("b");
  const oocB = dom.chatBOoc.value.trim();
  if (axesB && Object.keys(axesB).length > 0 && oocB) {
    character_b = {
      axes: axesB,
      ooc_message: oocB,
      channel: dom.chatBChannel.value,
      active_axes: chatState.b.activeAxes ? [...chatState.b.activeAxes] : null,
    };
  }

  const reqBody = {
    character_a,
    character_b,
    model,
    temperature: parseFloat(dom.chatTempInput.value),
    max_tokens: parseInt(dom.chatTokensInput.value, 10),
    seed: resolveChatSeed(),
    strict_mode: dom.chatStrictMode.checked,
    max_output_chars: parseInt(dom.chatMaxChars.value, 10),
    system_prompt: getEffectiveSystemPrompt(),
    world_id: chatState.worldId || dom.chatWorldSelect?.value || null,
    ollama_host: getChatOllamaHostForRequest(),
  };

  chatState.busy = true;
  dom.btnTranslate.disabled = true;
  setStatus(`Translating via ${model}…`, true);

  dom.chatAOutput.innerHTML = '<span class="placeholder-text">Translating…</span>';
  dom.chatBOutput.innerHTML = character_b
    ? '<span class="placeholder-text">Translating…</span>'
    : '<span class="placeholder-text">Not requested.</span>';
  dom.chatAMeta.textContent = "";
  dom.chatAMeta.classList.add("hidden");
  dom.chatBMeta.textContent = "";
  dom.chatBMeta.classList.add("hidden");

  const abortController = new AbortController();
  const timer = setTimeout(() => abortController.abort(), TRANSLATE_TIMEOUT_MS);

  try {
    const res = await fetch("/api/translate_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
      signal: abortController.signal,
    });
    if (!res.ok) {
      if (res.status === 401 && isServerMode()) {
        handleSessionExpired();
        return;
      }
      const errorData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errorData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderTranslationResult(
      dom.chatAOutput,
      dom.chatAMeta,
      dom.chatAStatusBadge,
      data.character_a,
      data.character_b?.system_prompt_hash ?? null,
    );
    renderTranslationResult(
      dom.chatBOutput,
      dom.chatBMeta,
      dom.chatBStatusBadge,
      data.character_b,
      data.character_a?.system_prompt_hash ?? null,
    );

    const usedModel = (data.character_a && data.character_a.model) || model;
    const statusParts = [`A: ${data.character_a.status}`];
    if (data.character_b) statusParts.push(`B: ${data.character_b.status}`);
    setStatus(`Done (${usedModel}) — ${statusParts.join(", ")}.`);
  } catch (err) {
    const msg = err.name === "AbortError"
      ? `Request timed out after ${TRANSLATE_TIMEOUT_MS / 1000}s — is the model loaded in Ollama?`
      : err.message;
    const errorSpan = document.createElement("span");
    errorSpan.style.color = "var(--col-err)";
    errorSpan.textContent = `Error: ${msg}`;
    dom.chatAOutput.textContent = "";
    dom.chatAOutput.appendChild(errorSpan.cloneNode(true));
    setStatus(`Translation error: ${msg}`);
  } finally {
    clearTimeout(timer);
    chatState.busy = false;
    dom.btnTranslate.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

/**
 * Initialise all async data needed by the Chat Translation page.
 *
 * @returns {Promise<void>}
 */
export async function initChatTranslation() {
  const labConfig = window.__LAB_CONFIG__ || {};
  chatState.translationMode = labConfig.translationMode || "standalone";
  updateModeBadge();

  await Promise.all([loadChatExampleList(), loadChatIcPromptList()]);

  await Promise.all([
    loadChatExample("a", "example_a").then(() => relabelChatChar("a")),
    loadChatExample("b", "example_b").then(() => relabelChatChar("b")),
  ]);
  charDom("a").exampleSelect.value = "example_a";
  charDom("b").exampleSelect.value = "example_b";

  chatState.liveMode = dom.chatLiveToggle.checked;
  updateLiveModeUI();
  syncUseAddressUI();

  if (isServerMode()) {
    await checkSession();
  }
}

/**
 * Wire all page-level events for the Chat Translation page.
 *
 * Feature-specific event groups for server mode, game-log actions, and import
 * restore flow are delegated to their dedicated modules.
 *
 * @returns {void}
 */
export function wireChatTranslationEvents() {
  for (const ch of ["a", "b"]) {
    const cd = charDom(ch);

    cd.btnLoadExample.addEventListener("click", () => {
      loadChatExample(ch, cd.exampleSelect.value);
    });

    cd.jsonTextarea.addEventListener(
      "input",
      debounce(() => {
        const obj = safeParse(cd.jsonTextarea.value);
        if (!obj) {
          setJsonBadge(ch, false);
          setStatus("Invalid JSON.");
          return;
        }

        setJsonBadge(ch, true);
        chatState[ch].payload = obj;
        buildChatSliders(ch);
        setStatus("JSON updated.");
      }, 280),
    );

    cd.btnRelabel.addEventListener("click", () => relabelChatChar(ch));
    cd.btnRandomise.addEventListener("click", () => randomiseChatChar(ch));
    cd.autoLabel.addEventListener("change", () => {
      buildChatSliders(ch);
      if (cd.autoLabel.checked) relabelChatChar(ch);
    });
    cd.btnSend.addEventListener("click", () => sendForChar(ch));
  }

  dom.chatTempRange.addEventListener("input", () => {
    dom.chatTempInput.value = dom.chatTempRange.value;
  });
  dom.chatTempInput.addEventListener("input", () => {
    const value = clamp(parseFloat(dom.chatTempInput.value) || 0, 0, 2);
    dom.chatTempRange.value = value;
  });

  dom.chatOllamaHost.addEventListener(
    "input",
    debounce(() => refreshChatModels(), 600),
  );
  dom.chatUseAddress.addEventListener("change", () => {
    chatState.useAddress = dom.chatUseAddress.checked;
    syncUseAddressUI();
  });

  dom.chatBtnLoadPrompt.addEventListener("click", async () => {
    const name = dom.chatPromptSelect.value;
    if (!name) return;

    try {
      const res = await fetch(`/api/prompts/${encodeURIComponent(name)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      dom.chatSystemPrompt.value = text;
      updateChatPromptBadge();
      setStatus(`IC prompt "${name}" loaded.`);
    } catch (err) {
      setStatus(`Prompt load error: ${err.message}`);
    }
  });
  dom.chatSystemPrompt.addEventListener("input", () => updateChatPromptBadge());

  dom.btnTranslate.addEventListener("click", () => translate());
  dom.chatLiveToggle.addEventListener("change", () => {
    chatState.liveMode = dom.chatLiveToggle.checked;
    updateLiveModeUI();
  });

  wireGameLogEvents();
  wireChatImportEvents();
  wireServerModeEvents();
}

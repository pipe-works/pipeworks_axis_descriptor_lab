/**
 * mod-chat-import.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Chat save-package import and state-restore helpers for the Chat Translation
 * page.
 *
 * Imports: mod-state, mod-status, mod-chat-state, mod-chat-sliders,
 *          mod-chat-game-log
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import { chatState, charDom } from "./mod-chat-state.js";
import { buildChatSliders, setJsonBadge } from "./mod-chat-sliders.js";
import { appendGameEntry, renderTranslationResult } from "./mod-chat-game-log.js";

function updateChatPromptBadge() {
  const hasContent = dom.chatSystemPrompt.value.trim().length > 0;
  dom.chatPromptBadge.textContent = hasContent ? "custom" : "default";
  dom.chatPromptBadge.className = hasContent ? "badge badge--active" : "badge badge--muted";
}

export function restoreChatSessionState(data) {
  for (const ch of ["a", "b"]) {
    const axesDict = data[`character_${ch}`];
    if (axesDict) {
      chatState[ch].payload = { axes: axesDict, policy_hash: null, seed: null, world_id: null };
      chatState[ch].originalAxes = JSON.parse(JSON.stringify(axesDict));
      chatState[ch].activeAxes = null;
      const cd = charDom(ch);
      cd.jsonTextarea.value = JSON.stringify(chatState[ch].payload, null, 2);
      setJsonBadge(ch, true);
      buildChatSliders(ch);
    }
  }

  const modelName = data.model || "";
  if (dom.chatModelSelect && !dom.chatModelSelect.classList.contains("hidden")) {
    const opt = Array.from(dom.chatModelSelect.options).find(o => o.value === modelName);
    if (opt) {
      dom.chatModelSelect.value = modelName;
    } else {
      dom.chatModelSelect.classList.add("hidden");
      dom.chatModelInput.classList.remove("hidden");
      dom.chatModelInput.value = modelName;
    }
  } else {
    dom.chatModelInput.value = modelName;
  }
  const temp = data.temperature ?? 0;
  dom.chatTempRange.value = temp;
  dom.chatTempInput.value = temp;
  dom.chatTokensInput.value = data.max_tokens ?? 128;
  dom.chatSeedInput.value = data.seed ?? -1;

  if (data.system_prompt) {
    dom.chatSystemPrompt.value = data.system_prompt;
    updateChatPromptBadge();
  }

  const entries = data.game_log_entries || [];
  if (entries.length > 0) {
    const hashByIdx = {};
    for (const hashEntry of (data.metadata?.per_entry_hashes ?? [])) {
      hashByIdx[hashEntry.index] = hashEntry;
    }

    dom.chatGameOutput.innerHTML = "";
    chatState.gameLog = [];
    chatState.logSeq = 0;

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const hashEntry = hashByIdx[i + 1] ?? {};
      appendGameEntry(
        entry.ch, entry.channel, entry.ooc_message, entry.ic_text, data.model,
        hashEntry.status ?? entry.status ?? "success",
        hashEntry.error_detail ?? entry.error_detail ?? null,
        hashEntry.sent_at ?? entry.sent_at ?? null,
        hashEntry.duration_ms ?? entry.duration_ms ?? null,
        hashEntry.ipc_id ?? null,
        hashEntry.input_hash ?? null,
        hashEntry.system_prompt_hash ?? null,
        hashEntry.output_hash ?? null,
        hashEntry.system_prompt ?? null,
      );
    }

    const lastA = [...chatState.gameLog].reverse().find(entry => entry.ch === "a");
    const lastB = [...chatState.gameLog].reverse().find(entry => entry.ch === "b");

    for (const ch of ["a", "b"]) {
      const lastEntry = ch === "a" ? lastA : lastB;
      if (!lastEntry) continue;

      const outputBox = ch === "a" ? dom.chatAOutput : dom.chatBOutput;
      const metaDiv = ch === "a" ? dom.chatAMeta : dom.chatBMeta;
      const badge = ch === "a" ? dom.chatAStatusBadge : dom.chatBStatusBadge;
      const otherHash = (ch === "a" ? lastB : lastA)?.systemPromptHash ?? null;

      renderTranslationResult(outputBox, metaDiv, badge, {
        status: "success",
        ic_text: lastEntry.icText,
        input_hash: lastEntry.inputHash,
        system_prompt_hash: lastEntry.systemPromptHash,
        output_hash: lastEntry.outputHash,
        ipc_id: lastEntry.ipcId,
      }, otherHash);
    }

    dom.chatGameSection.classList.remove("hidden");
  }
}

export async function importChatSave() {
  const file = dom.chatImportFileInput.files[0];
  if (!file) return;

  setStatus("Importing chat session…", true);
  const fd = new FormData();
  fd.append("file", file);
  dom.chatImportFileInput.value = "";

  try {
    const res = await fetch("/api/import_chat", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    restoreChatSessionState(data);
    const warnSuffix = data.warnings.length ? ` (${data.warnings.length} warning(s))` : "";
    setStatus(`Chat session imported from ${data.folder_name}.${warnSuffix}`);
  } catch (err) {
    setStatus(`Import error: ${err.message}`);
  } finally {
    dom.spinner.classList.add("hidden");
  }
}

export function wireChatImportEvents() {
  dom.chatImportLog.addEventListener("click", () => dom.chatImportFileInput.click());
  dom.chatImportFileInput.addEventListener("change", () => {
    if (dom.chatImportFileInput.files.length) importChatSave();
  });
}

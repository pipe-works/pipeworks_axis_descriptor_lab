/**
 * mod-chat-game-log.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Output meta rendering, in-game log management, clipboard export, and chat
 * save-package submission for the Chat Translation page.
 *
 * Imports: mod-state, mod-status, mod-chat-state, mod-chat-server-mode
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import { chatState } from "./mod-chat-state.js";
import {
  getCurrentSystemPromptText,
  getEffectiveSystemPrompt,
  isServerMode,
} from "./mod-chat-server-mode.js";

/**
 * Build a compact IPC meta table element from a `ChatTranslationResult`.
 *
 * @param {Object} result - `ChatTranslationResult` object from the API.
 * @param {string|null} [otherSpHash] - The other character's system prompt hash.
 * @returns {HTMLTableElement}
 */
export function buildIpcMetaTable(result, otherSpHash = null) {
  const rows = [];
  if (result.input_hash) rows.push(["input", result.input_hash.slice(0, 16) + "\u2026", null]);
  if (result.system_prompt_hash) {
    let cssClass = null;
    if (otherSpHash != null) {
      cssClass = result.system_prompt_hash === otherSpHash ? "sp-match" : "sp-mismatch";
    }
    rows.push(["sys prompt", result.system_prompt_hash.slice(0, 16) + "\u2026", cssClass]);
  }
  if (result.output_hash) rows.push(["output", result.output_hash.slice(0, 16) + "\u2026", null]);
  if (result.ipc_id) rows.push(["ipc", result.ipc_id.slice(0, 16) + "\u2026", null]);

  const table = document.createElement("table");
  table.className = "meta-table";
  for (const [key, val, cssClass] of rows) {
    const tr = document.createElement("tr");
    if (cssClass) tr.className = cssClass;
    const tdKey = document.createElement("td");
    tdKey.className = "meta-key";
    tdKey.textContent = key;
    const tdVal = document.createElement("td");
    tdVal.className = "meta-val";
    tdVal.textContent = val;
    tr.appendChild(tdKey);
    tr.appendChild(tdVal);
    table.appendChild(tr);
  }
  return table;
}

/**
 * Render a chat translation result into the output and meta panels.
 *
 * @param {HTMLElement} outputBox
 * @param {HTMLElement} metaDiv
 * @param {HTMLElement} statusBadge
 * @param {Object|null} result
 * @param {string|null} [otherSpHash]
 * @returns {void}
 */
export function renderTranslationResult(outputBox, metaDiv, statusBadge, result, otherSpHash = null) {
  outputBox.textContent = "";
  metaDiv.textContent = "";
  metaDiv.classList.add("hidden");

  if (result === null) {
    outputBox.innerHTML = '<span class="placeholder-text">Not requested.</span>';
    statusBadge.textContent = "—";
    statusBadge.className = "badge badge--muted";
    return;
  }

  if (result.status === "success") {
    statusBadge.textContent = "ok";
    statusBadge.className = "badge";
  } else if (result.status === "fallback.api_error") {
    statusBadge.textContent = "api error";
    statusBadge.className = "badge badge--err";
  } else {
    statusBadge.textContent = "rejected";
    statusBadge.className = "badge badge--err";
  }

  if (result.ic_text) {
    outputBox.textContent = result.ic_text;
    metaDiv.appendChild(buildIpcMetaTable(result, otherSpHash));
    metaDiv.classList.remove("hidden");
  } else {
    const errSpan = document.createElement("span");
    errSpan.style.color = "var(--col-err)";
    if (result.error_detail) {
      errSpan.textContent = result.error_detail;
    } else if (result.status === "fallback.api_error") {
      errSpan.textContent = isServerMode()
        ? "Server translation failed — check server logs."
        : "Ollama unreachable or timed out — check host and model.";
    } else {
      errSpan.textContent =
        "Output rejected by validator (PASSTHROUGH or constraint violation).";
    }
    outputBox.appendChild(errSpan);
  }
}

/**
 * Append a single in-game log entry to the game output panel.
 *
 * The saved in-memory entry keeps both the prompt hash and, when available,
 * the prompt text itself so save/export can preserve prompt changes that
 * happened mid-conversation.
 *
 * @returns {void}
 */
export function appendGameEntry(
  ch, channel, oocMessage, icText, model,
  status = "success", errorDetail = null, sentAt = null, durationMs = null,
  ipcId = null, inputHash = null, systemPromptHash = null, outputHash = null, systemPrompt = null,
) {
  chatState.logSeq++;
  chatState.gameLog.push({
    ch, channel, oocMessage, icText, model,
    status, errorDetail, sentAt, durationMs,
    ipcId, inputHash, systemPromptHash, outputHash, systemPrompt,
  });
  const placeholder = dom.chatGameOutput.querySelector(".placeholder-text");
  if (placeholder) placeholder.remove();

  const isFailed = status !== "success";
  const entry = document.createElement("div");
  entry.className = `game-entry game-entry--${ch}${isFailed ? " game-entry--failed" : ""}`;

  const charEl = document.createElement("span");
  charEl.className = "game-entry__char";
  charEl.textContent = ch.toUpperCase();

  const oocEl = document.createElement("span");
  oocEl.className = "game-entry__ooc";
  oocEl.textContent = `"${oocMessage}"`;
  oocEl.title = oocMessage;

  const channelEl = document.createElement("span");
  channelEl.className = "game-entry__channel";
  channelEl.textContent = `[${channel}]`;

  const textEl = document.createElement("span");
  textEl.className = "game-entry__text";
  textEl.textContent = isFailed ? (errorDetail || status) : (icText || "");

  entry.appendChild(charEl);
  entry.appendChild(oocEl);
  entry.appendChild(channelEl);
  entry.appendChild(textEl);
  dom.chatGameOutput.appendChild(entry);
  dom.chatGameOutput.scrollTop = dom.chatGameOutput.scrollHeight;
}

function getChatModelName() {
  const sel = dom.chatModelSelect.value.trim();
  return sel || dom.chatModelInput.value.trim();
}

function resolveChatSeed() {
  const raw = parseInt(dom.chatSeedInput.value, 10);
  if (isNaN(raw) || raw < 0) {
    return Math.floor(Math.random() * 0x100000000);
  }
  return raw;
}

export function copyGameLogTxt() {
  if (chatState.gameLog.length === 0) { setStatus("No entries to copy."); return; }
  const text = chatState.gameLog
    .map(e => {
      const dur = e.durationMs != null ? `${(e.durationMs / 1000).toFixed(1)}s` : "?";
      const tag = e.status === "success" ? "ok" : "FAILED";
      const display = e.status === "success" ? (e.icText || "") : (e.errorDetail || e.status);
      return `${e.ch.toUpperCase()} | ${e.oocMessage} | [${e.channel}]: ${display} (${dur}, ${tag})`;
    })
    .join("\n");
  navigator.clipboard.writeText(text).then(() => {
    dom.chatCopyLogTxt.textContent = "Copied!";
    setTimeout(() => { dom.chatCopyLogTxt.textContent = "Copy TXT"; }, 1200);
  });
}

export function copyGameLogMd() {
  if (chatState.gameLog.length === 0) { setStatus("No entries to copy."); return; }
  const lines = [
    "| # | Char | OOC | Channel | IC Text | Status | Duration | Sent | Gap |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
  ];
  let prevSentAt = null;
  chatState.gameLog.forEach((entry, i) => {
    const oocEscaped = (entry.oocMessage || "").replace(/\|/g, "\\|");
    const icRaw = entry.status === "success" ? (entry.icText || "") : (entry.errorDetail || "");
    const icEscaped = icRaw.replace(/\|/g, "\\|");
    const statusLabel = entry.status === "success" ? "ok" : entry.status.replace("fallback.", "");
    const durStr = entry.durationMs != null ? `${(entry.durationMs / 1000).toFixed(1)}s` : "";
    let sentStr = "";
    if (entry.sentAt) {
      try { sentStr = new Date(entry.sentAt).toISOString().slice(11, 19); } catch { /* */ }
    }
    let gapStr = "";
    if (entry.sentAt && prevSentAt) {
      try {
        const gap = (new Date(entry.sentAt) - new Date(prevSentAt)) / 1000;
        gapStr = `${gap.toFixed(1)}s`;
      } catch { /* */ }
    }
    prevSentAt = entry.sentAt;
    lines.push(
      `| ${i + 1} | ${entry.ch.toUpperCase()} | ${oocEscaped} | ${entry.channel}`
      + ` | ${icEscaped} | ${statusLabel} | ${durStr} | ${sentStr} | ${gapStr} |`
    );
  });
  navigator.clipboard.writeText(lines.join("\n")).then(() => {
    dom.chatCopyLogMd.textContent = "Copied!";
    setTimeout(() => { dom.chatCopyLogMd.textContent = "Copy MD"; }, 1200);
  });
}

export async function saveChatLog() {
  if (chatState.gameLog.length === 0) { setStatus("No game log entries to save."); return; }

  const model = getChatModelName();
  const reqBody = {
    entries: chatState.gameLog.map(e => ({
      ch: e.ch,
      channel: e.channel,
      ooc_message: e.oocMessage ?? "",
      ic_text: e.icText ?? null,
      model: e.model,
      status: e.status ?? "success",
      error_detail: e.errorDetail ?? null,
      sent_at: e.sentAt ?? null,
      duration_ms: e.durationMs ?? null,
      ipc_id: e.ipcId ?? null,
      input_hash: e.inputHash ?? null,
      system_prompt_hash: e.systemPromptHash ?? null,
      output_hash: e.outputHash ?? null,
      system_prompt: e.systemPrompt ?? null,
    })),
    character_a: chatState.a.payload ? chatState.a.payload.axes : null,
    character_b: chatState.b.payload ? chatState.b.payload.axes : null,
    model,
    temperature: parseFloat(dom.chatTempInput.value),
    max_tokens: parseInt(dom.chatTokensInput.value, 10),
    seed: resolveChatSeed(),
    // Save the prompt currently visible in the UI so system_prompt.md reflects
    // the present conversation state even when the active server prompt is in use.
    system_prompt: (isServerMode() && chatState.authenticated)
      ? getCurrentSystemPromptText()
      : getEffectiveSystemPrompt(),
  };

  dom.chatSaveLog.disabled = true;
  setStatus("Saving game log…", true);

  try {
    const saveRes = await fetch("/api/save_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
    });
    if (!saveRes.ok) {
      const errData = await saveRes.json().catch(() => ({ detail: saveRes.statusText }));
      throw new Error(errData.detail || `HTTP ${saveRes.status}`);
    }
    const saveData = await saveRes.json();

    const exportRes = await fetch(
      `/api/save/${encodeURIComponent(saveData.folder_name)}/export`
    );
    if (!exportRes.ok) throw new Error(`Export HTTP ${exportRes.status}`);
    const blob = await exportRes.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${saveData.folder_name}_chat.zip`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);

    setStatus(`Chat log saved — ${saveData.files.length} files in ${saveData.folder_name}.`);
  } catch (err) {
    setStatus(`Save error: ${err.message}`);
  } finally {
    dom.chatSaveLog.disabled = false;
    dom.spinner.classList.add("hidden");
  }
}

export function wireGameLogEvents() {
  dom.chatClearLog.addEventListener("click", () => {
    dom.chatGameOutput.innerHTML = '<span class="placeholder-text">Send a message to see in-game output.</span>';
    chatState.logSeq = 0;
    chatState.gameLog = [];
  });

  dom.chatToggleOoc.addEventListener("click", () => {
    const nowVisible = dom.chatGameOutput.classList.toggle("chat-game-output--hide-ooc");
    dom.chatToggleOoc.classList.toggle("is-active", !nowVisible);
  });

  dom.chatCopyLogTxt.addEventListener("click", () => copyGameLogTxt());
  dom.chatCopyLogMd.addEventListener("click", () => copyGameLogMd());
  dom.chatSaveLog.addEventListener("click", () => saveChatLog());
}

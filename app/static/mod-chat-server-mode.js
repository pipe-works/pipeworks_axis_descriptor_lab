/**
 * mod-chat-server-mode.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Mud-server authentication, world selection, and server-prompt management
 * for the Chat Translation page.
 *
 * This module isolates the server-backed translation mode so the main chat
 * controller can stay focused on request construction and output rendering.
 *
 * Imports: mod-state, mod-status, mod-chat-state
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import { chatState, charDom } from "./mod-chat-state.js";

export function isServerMode() {
  return chatState.translationMode !== "standalone";
}

export function updateModeBadge() {
  const badge = dom.chatModeBadge;
  if (!badge) return;
  badge.classList.remove("badge--info", "badge--active", "badge--muted");
  switch (chatState.translationMode) {
    case "server-prod":
      badge.textContent = "Server (prod)";
      badge.classList.add("badge--info");
      break;
    case "server-local":
      badge.textContent = "Server (local)";
      badge.classList.add("badge--active");
      break;
    default:
      badge.textContent = "Standalone";
      badge.classList.add("badge--muted");
  }
}

export async function checkSession() {
  if (!isServerMode()) return;
  try {
    const res = await fetch("/api/mud/session");
    if (!res.ok) { showLoginPanel(); return; }
    const data = await res.json();
    if (data.authenticated) {
      await onAuthenticated();
    } else {
      showLoginPanel();
    }
  } catch {
    showLoginPanel();
  }
}

function showLoginPanel() {
  dom.chatLoginPanel.classList.remove("hidden");
  dom.chatBtnDisconnect.classList.add("hidden");
  dom.chatWorldSelector.classList.add("hidden");
  toggleServerControls(false);
}

function hideLoginPanel() {
  dom.chatLoginPanel.classList.add("hidden");
  dom.chatLoginError.classList.add("hidden");
  dom.chatLoginError.textContent = "";
}

export async function doLogin() {
  const username = dom.chatLoginUsername.value.trim();
  const password = dom.chatLoginPassword.value;
  if (!username || !password) {
    dom.chatLoginError.textContent = "Username and password required.";
    dom.chatLoginError.classList.remove("hidden");
    return;
  }
  dom.chatBtnConnect.disabled = true;
  dom.chatLoginError.classList.add("hidden");
  try {
    const res = await fetch("/api/mud/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    await onAuthenticated();
  } catch (err) {
    dom.chatLoginError.textContent = err.message;
    dom.chatLoginError.classList.remove("hidden");
  } finally {
    dom.chatBtnConnect.disabled = false;
  }
}

export async function doLogout() {
  try { await fetch("/api/mud/logout", { method: "POST" }); } catch { /* ignore */ }
  chatState.authenticated = false;
  chatState.worlds = [];
  chatState.worldId = null;
  chatState.worldConfig = null;
  chatState.worldPrompts = [];
  chatState.serverPromptOriginal = "";
  clearActiveAxesIndicators();
  dom.chatServerConfigInfo.classList.add("hidden");
  showLoginPanel();
  setStatus("Disconnected from mud server.");
}

async function onAuthenticated() {
  chatState.authenticated = true;
  hideLoginPanel();
  dom.chatBtnDisconnect.classList.remove("hidden");
  toggleServerControls(true);
  await fetchWorlds();
  setStatus("Connected to mud server.");
}

async function fetchWorlds() {
  try {
    const res = await fetch("/api/mud/worlds");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    chatState.worlds = data.worlds || [];
    populateWorldSelect();
    dom.chatWorldSelector.classList.remove("hidden");
  } catch (err) {
    setStatus(`Failed to fetch worlds: ${err.message}`);
  }
}

function populateWorldSelect() {
  const sel = dom.chatWorldSelect;
  sel.innerHTML = '<option value="">— select world —</option>';
  for (const world of chatState.worlds) {
    const opt = document.createElement("option");
    opt.value = world.world_id;
    opt.textContent = world.name || world.world_id;
    sel.appendChild(opt);
  }
  const enabled = chatState.worlds.filter(world => world.translation_enabled);
  const target = chatState.worldId
    || (enabled.length === 1 ? enabled[0].world_id : null)
    || (chatState.worlds.length === 1 ? chatState.worlds[0].world_id : null);

  if (target) {
    sel.value = target;
    if (sel.value === target) {
      chatState.worldId = target;
      selectWorld(target);
    }
  }
}

async function fetchWorldPrompts(worldId) {
  try {
    const res = await fetch(`/api/mud/world-prompts/${encodeURIComponent(worldId)}`);
    if (!res.ok) {
      if (res.status === 401) { handleSessionExpired(); return; }
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    chatState.worldPrompts = data.prompts || [];
    populateServerPromptSelect();
  } catch (err) {
    chatState.worldPrompts = [];
    setStatus(`Failed to fetch prompts: ${err.message}`);
  }
}

export async function selectWorld(worldId) {
  if (!worldId) {
    chatState.worldId = null;
    chatState.worldConfig = null;
    clearActiveAxesIndicators();
    dom.chatServerConfigInfo.classList.add("hidden");
    return;
  }
  chatState.worldConfigLoading = true;
  try {
    const selRes = await fetch("/api/mud/select-world", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ world_id: worldId }),
    });
    if (!selRes.ok) {
      if (selRes.status === 401) { handleSessionExpired(); return; }
      throw new Error(`select-world: HTTP ${selRes.status}`);
    }

    const cfgRes = await fetch(`/api/mud/world-config/${encodeURIComponent(worldId)}`);
    if (!cfgRes.ok) {
      if (cfgRes.status === 401) { handleSessionExpired(); return; }
      throw new Error(`world-config: HTTP ${cfgRes.status}`);
    }
    const config = await cfgRes.json();
    chatState.worldId = worldId;
    chatState.worldConfig = config;
    applyActiveAxesIndicators();
    updateServerConfigDisplay();
    await fetchWorldPrompts(worldId);
    setStatus(`World "${worldId}" selected.`);
  } catch (err) {
    setStatus(`World selection error: ${err.message}`);
  } finally {
    chatState.worldConfigLoading = false;
  }
}

export function handleSessionExpired() {
  chatState.authenticated = false;
  chatState.worldId = null;
  chatState.worldConfig = null;
  clearActiveAxesIndicators();
  dom.chatServerConfigInfo.classList.add("hidden");
  showLoginPanel();
  setStatus("Session expired — please log in again.");
}

export function toggleServerControls(auth) {
  const settingsDetails = document.getElementById("chat-ollama-settings-details");
  if (settingsDetails) {
    for (const el of settingsDetails.querySelectorAll("[data-server-hide]")) {
      el.classList.toggle("hidden", auth);
    }
  }
  if (dom.chatIcPromptDetails) {
    dom.chatIcPromptDetails.classList.toggle("hidden", auth);
  }
  if (dom.chatServerPromptDetails) {
    dom.chatServerPromptDetails.classList.toggle("hidden", !auth);
  }
  if (dom.chatServerConfigInfo) {
    dom.chatServerConfigInfo.classList.toggle("hidden", !auth);
  }
}

export function applyActiveAxesIndicators() {
  const config = chatState.worldConfig;
  if (!config || !config.active_axes) return;
  const activeSet = new Set(config.active_axes);
  for (const ch of ["a", "b"]) {
    const panel = charDom(ch).sliderPanel;
    for (const row of panel.querySelectorAll(".axis-row")) {
      const axis = row.dataset.axis;
      if (!axis) continue;
      const inactive = !activeSet.has(axis);
      row.classList.toggle("axis-row--inactive-server", inactive);
      if (inactive) {
        row.title = `"${axis}" is not an active axis in this world.`;
      } else {
        row.removeAttribute("title");
      }
    }
  }
}

export function clearActiveAxesIndicators() {
  for (const ch of ["a", "b"]) {
    const panel = charDom(ch).sliderPanel;
    for (const row of panel.querySelectorAll(".axis-row--inactive-server")) {
      row.classList.remove("axis-row--inactive-server");
      row.removeAttribute("title");
    }
  }
}

function updateServerConfigDisplay() {
  const config = chatState.worldConfig;
  if (!config) return;
  if (dom.chatServerModel) {
    dom.chatServerModel.textContent = config.model || "--";
  }
  if (dom.chatServerActiveAxes) {
    dom.chatServerActiveAxes.textContent = config.active_axes
      ? config.active_axes.join(", ")
      : "--";
  }
  dom.chatServerConfigInfo.classList.remove("hidden");
}

function populateServerPromptSelect() {
  const sel = dom.chatServerPromptSelect;
  if (!sel) return;

  const textarea = dom.chatServerPromptText;
  const prevSelection = sel.value;
  const hasModifications = textarea
    && chatState.serverPromptOriginal !== ""
    && textarea.value !== chatState.serverPromptOriginal;

  sel.innerHTML = "";
  for (const prompt of chatState.worldPrompts) {
    const opt = document.createElement("option");
    opt.value = prompt.filename;
    opt.textContent = prompt.is_active ? `${prompt.filename} (active)` : prompt.filename;
    sel.appendChild(opt);
  }

  if (hasModifications && prevSelection) {
    const stillExists = chatState.worldPrompts.find(prompt => prompt.filename === prevSelection);
    if (stillExists) {
      sel.value = prevSelection;
      chatState.serverPromptOriginal = stillExists.content;
      updateServerPromptBadge();
      return;
    }
  }

  const active = chatState.worldPrompts.find(prompt => prompt.is_active);
  if (active) {
    sel.value = active.filename;
    loadServerPrompt(active.filename);
  } else if (chatState.worldPrompts.length > 0) {
    sel.value = chatState.worldPrompts[0].filename;
    loadServerPrompt(chatState.worldPrompts[0].filename);
  }
}

function loadServerPrompt(filename) {
  const prompt = chatState.worldPrompts.find(entry => entry.filename === filename);
  if (!prompt) return;
  const textarea = dom.chatServerPromptText;
  if (textarea) textarea.value = prompt.content;
  chatState.serverPromptOriginal = prompt.content;
  updateServerPromptBadge();
}

function updateServerPromptBadge() {
  const badge = dom.chatServerPromptBadge;
  if (!badge) return;
  const textarea = dom.chatServerPromptText;
  if (!textarea) return;
  const modified = textarea.value !== chatState.serverPromptOriginal;
  badge.textContent = modified ? "modified" : "server";
  badge.classList.toggle("badge--warn", modified);
  badge.classList.toggle("badge--muted", !modified);
}

export function getEffectiveSystemPrompt() {
  if (isServerMode() && chatState.authenticated) {
    const textarea = dom.chatServerPromptText;
    if (!textarea) return null;
    const text = textarea.value;
    if (text === chatState.serverPromptOriginal) return null;
    return text.trim() || null;
  }
  return dom.chatSystemPrompt.value.trim() || null;
}

/**
 * Wire all mud-server authentication, world-selection, and prompt-editor
 * events for the Chat Translation page.
 *
 * Keeping these listeners inside the server-mode module avoids forcing the
 * parent chat controller to know about the server prompt/editor internals.
 *
 * @returns {void}
 */
export function wireServerModeEvents() {
  dom.chatBtnConnect.addEventListener("click", () => doLogin());

  // Allow Enter in either login field to submit credentials without
  // requiring an explicit click on the Connect button.
  dom.chatLoginPassword.addEventListener("keydown", (event) => {
    if (event.key === "Enter") doLogin();
  });
  dom.chatLoginUsername.addEventListener("keydown", (event) => {
    if (event.key === "Enter") doLogin();
  });

  dom.chatBtnDisconnect.addEventListener("click", () => doLogout());
  dom.chatWorldSelect.addEventListener("change", () => {
    const worldId = dom.chatWorldSelect.value;
    chatState.worldId = worldId || null;
    selectWorld(worldId);
  });

  if (dom.chatServerPromptSelect) {
    dom.chatServerPromptSelect.addEventListener("change", () => {
      loadServerPrompt(dom.chatServerPromptSelect.value);
    });
  }
  if (dom.chatServerPromptText) {
    dom.chatServerPromptText.addEventListener("input", () => updateServerPromptBadge());
  }
  if (dom.chatBtnResetPrompt) {
    dom.chatBtnResetPrompt.addEventListener("click", () => {
      const select = dom.chatServerPromptSelect;
      if (select && select.value) loadServerPrompt(select.value);
    });
  }

  // Re-check the session when the user navigates back to the chat page so
  // stale sessions are surfaced promptly.
  document.addEventListener("chat-translation-activated", () => {
    if (isServerMode()) checkSession();
  });
}

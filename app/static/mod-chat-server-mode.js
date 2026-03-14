/**
 * mod-chat-server-mode.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Mud-server runtime mode selection, authentication, world selection, and
 * server-prompt management shared across the full app shell.
 *
 * The auth/session controls are global and reused by Character Description
 * canonical generation, Chat Translation, and Pipeline Build server flows.
 * Chat-specific prompt behaviour still lives here to keep related mud-session
 * concerns in one module.
 *
 * Imports: mod-state, mod-status, mod-chat-state
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import { chatState, charDom } from "./mod-chat-state.js";

const CHAT_MODE_STORAGE_KEY = "padl-chat-mode";
const CHAT_MODE_DEV_URL_STORAGE_KEY = "padl-chat-dev-server-url";
const LAB_ALLOWED_ROLES = new Set(["admin", "superuser"]);

/**
 * Return true when the chat page is currently targeting a mud server.
 *
 * @returns {boolean}
 */
export function isServerMode() {
  return chatState.translationMode !== "standalone";
}

function isAuthorizedLabRole(role) {
  return LAB_ALLOWED_ROLES.has(role || "");
}

/**
 * Refresh the mode badge from the current runtime mode state.
 *
 * @returns {void}
 */
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

function updateModeUrl() {
  if (!dom.chatModeUrl) return;
  if (!isServerMode()) {
    dom.chatModeUrl.textContent = "Uses the local standalone translator.";
    return;
  }

  const label = chatState.availableModes.find((option) => option.key === chatState.modeKey)?.label;
  const prefix = label || "Mud server";
  dom.chatModeUrl.textContent = chatState.activeServerUrl
    ? `${prefix}: ${chatState.activeServerUrl}`
    : `${prefix} active.`;
}

function getDevelopmentModeOption() {
  return chatState.availableModes.find((option) => option.key === "development") || null;
}

function syncDevelopmentUrlControls() {
  const input = dom.chatModeDevUrl;
  const button = dom.chatModeDevApply;
  if (!input || !button) return;

  input.value = chatState.developmentServerUrl || "";
  button.disabled = input.value.trim().length === 0;

  const showControls = getDevelopmentModeOption() !== null && dom.chatModeSelect?.value === "development";
  input.classList.toggle("hidden", !showControls);
  button.classList.toggle("hidden", !showControls);
}

function syncModeSelect() {
  const select = dom.chatModeSelect;
  if (!select) return;

  select.innerHTML = "";
  for (const option of chatState.availableModes) {
    const el = document.createElement("option");
    el.value = option.key;
    el.textContent = option.label;
    select.appendChild(el);
  }

  if (select.querySelector(`option[value="${chatState.modeKey}"]`)) {
    select.value = chatState.modeKey;
  }
  syncDevelopmentUrlControls();
}

function applyModeConfig(data) {
  chatState.modeKey = data.mode_key;
  chatState.translationMode = data.translation_mode;
  chatState.activeServerUrl = data.active_server_url || null;
  chatState.availableModes = Array.isArray(data.available_modes) ? data.available_modes : [];
  chatState.developmentServerUrl = getDevelopmentModeOption()?.server_url || null;
  updateModeBadge();
  updateModeUrl();
  syncModeSelect();
}

function applySessionModeState(data) {
  if (data.mode_key) {
    chatState.modeKey = data.mode_key;
  }
  if (data.translation_mode) {
    chatState.translationMode = data.translation_mode;
  }
  chatState.activeServerUrl = data.active_server_url || null;
  if (chatState.modeKey === "development") {
    chatState.developmentServerUrl = data.active_server_url || chatState.developmentServerUrl;
  }
  updateModeBadge();
  updateModeUrl();
  if (dom.chatModeSelect && dom.chatModeSelect.querySelector(`option[value="${chatState.modeKey}"]`)) {
    dom.chatModeSelect.value = chatState.modeKey;
  }
  syncDevelopmentUrlControls();
}

/**
 * Broadcast mud session/mode context changes to cross-page consumers.
 *
 * Pipeline Build and other canonical flows rely on the shared mud-session
 * shell, so they need a stable event whenever runtime mode context changes.
 *
 * @param {string} reason - Short machine-readable trigger reason.
 * @returns {void}
 */
function dispatchMudSessionContextChanged(reason) {
  document.dispatchEvent(
    new CustomEvent("mud-session-context-changed", {
      detail: {
        reason,
        mode_key: chatState.modeKey,
        translation_mode: chatState.translationMode,
        active_server_url: chatState.activeServerUrl,
      },
    })
  );
}

function resetServerState({ dispatchCleared = true } = {}) {
  chatState.authenticated = false;
  chatState.worlds = [];
  chatState.worldId = null;
  chatState.worldConfig = null;
  chatState.worldConfigLoading = false;
  chatState.worldPrompts = [];
  chatState.serverPromptOriginal = "";
  clearActiveAxesIndicators();

  if (dom.chatWorldSelect) {
    dom.chatWorldSelect.innerHTML = '<option value="">— select world —</option>';
  }
  if (dom.chatServerPromptSelect) {
    dom.chatServerPromptSelect.innerHTML = '<option value="">— loading —</option>';
  }
  if (dom.chatServerPromptText) {
    dom.chatServerPromptText.value = "";
  }
  if (dom.chatServerModel) {
    dom.chatServerModel.textContent = "--";
  }
  if (dom.chatServerActiveAxes) {
    dom.chatServerActiveAxes.textContent = "--";
  }
  if (dom.chatServerConfigInfo) {
    dom.chatServerConfigInfo.classList.add("hidden");
  }

  if (dispatchCleared) {
    document.dispatchEvent(new CustomEvent("chat-world-config-cleared"));
  }
}

/**
 * Fetch the active runtime chat mode and reconcile it with any saved browser
 * preference from a previous visit.
 *
 * The backend remains authoritative.  The saved preference is applied only if
 * the server still exposes that mode in its available mode list.
 *
 * @returns {Promise<void>}
 */
export async function initMudMode() {
  const response = await fetch("/api/mud/mode");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const data = await response.json();
  applyModeConfig(data);

  const savedMode = window.localStorage.getItem(CHAT_MODE_STORAGE_KEY);
  const savedDevelopmentUrl = window.localStorage.getItem(CHAT_MODE_DEV_URL_STORAGE_KEY);
  if (savedDevelopmentUrl) {
    chatState.developmentServerUrl = savedDevelopmentUrl;
    syncDevelopmentUrlControls();
  }
  const modeExists = chatState.availableModes.some((option) => option.key === savedMode);
  if (savedMode && modeExists && savedMode !== chatState.modeKey) {
    await setRuntimeMode(savedMode, {
      serverUrl: savedMode === "development" ? savedDevelopmentUrl : null,
      persist: false,
      refreshSession: false,
      showStatus: false,
    });
  }
}

/**
 * Switch the backend chat mode at runtime and update the page state to match.
 *
 * This clears frontend-only mud session UI state before rehydrating from the
 * newly selected mode so stale world config or login state cannot bleed across
 * mode switches.
 *
 * @param {string} modeKey - Runtime mode key to activate.
 * @param {{persist?: boolean, refreshSession?: boolean, serverUrl?: string|null, showStatus?: boolean}} [options]
 *   Mode-switch behaviour flags for startup restoration vs direct user action.
 * @returns {Promise<void>}
 */
export async function setRuntimeMode(
  modeKey,
  { persist = true, refreshSession = true, serverUrl = null, showStatus = true } = {},
) {
  const trimmedServerUrl = typeof serverUrl === "string" ? serverUrl.trim() : null;
  const currentDevelopmentUrl = (chatState.developmentServerUrl || "").trim();
  const sameDevelopmentUrl = modeKey !== "development" || trimmedServerUrl === null
    || trimmedServerUrl === currentDevelopmentUrl;

  if (modeKey === chatState.modeKey && sameDevelopmentUrl) {
    if (persist) {
      window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, modeKey);
    }
    return;
  }

  const requestBody = { mode_key: modeKey };
  if (modeKey === "development" && trimmedServerUrl) {
    requestBody.server_url = trimmedServerUrl;
  }

  const response = await fetch("/api/mud/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  resetServerState();
  applyModeConfig(await response.json());

  if (persist) {
    window.localStorage.setItem(CHAT_MODE_STORAGE_KEY, chatState.modeKey);
    if (chatState.developmentServerUrl) {
      window.localStorage.setItem(CHAT_MODE_DEV_URL_STORAGE_KEY, chatState.developmentServerUrl);
    }
  }

  if (isServerMode()) {
    if (refreshSession) {
      await checkSession();
    }
  } else {
    hideLoginPanel();
    dom.chatBtnDisconnect.classList.add("hidden");
    dom.chatWorldSelector?.classList.add("hidden");
    toggleServerControls(false);
  }

  if (showStatus) {
    const label = chatState.availableModes.find((option) => option.key === chatState.modeKey)?.label;
    setStatus(`${label || "Chat mode"} selected.`);
  }

  dispatchMudSessionContextChanged("runtime_mode_changed");
}

/**
 * Query the backend for the current mud session status in the active mode.
 *
 * @returns {Promise<void>}
 */
export async function checkSession() {
  if (!isServerMode()) return;
  try {
    const res = await fetch("/api/mud/session");
    if (!res.ok) { showLoginPanel(); return; }
    const data = await res.json();
    applySessionModeState(data);
    chatState.authenticated = Boolean(data.authenticated);
    chatState.worldId = data.selected_world_id || null;
    if (data.authenticated && isAuthorizedLabRole(data.role)) {
      await onAuthenticated();
    } else if (data.authenticated) {
      resetServerState();
      showLoginPanel("This mud server account is not authorised for the Axis Lab. Admin or superuser access is required.");
    } else {
      showLoginPanel();
    }
  } catch {
    showLoginPanel();
  }
}

function showLoginPanel(message = "") {
  if (!isServerMode()) {
    hideLoginPanel();
    return;
  }
  dom.chatLoginPanel.classList.remove("hidden");
  dom.chatBtnDisconnect.classList.add("hidden");
  dom.chatWorldSelector?.classList.add("hidden");
  toggleServerControls(false);
  if (message) {
    dom.chatLoginError.textContent = message;
    dom.chatLoginError.classList.remove("hidden");
  } else {
    dom.chatLoginError.classList.add("hidden");
    dom.chatLoginError.textContent = "";
  }
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
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      throw new Error(data?.detail || res.statusText || `HTTP ${res.status}`);
    }
    if (!data?.authenticated) {
      resetServerState();
      showLoginPanel(data?.message || "Login failed.");
      return;
    }
    if (!isAuthorizedLabRole(data.role)) {
      resetServerState();
      showLoginPanel("This mud server account is not authorised for the Axis Lab. Admin or superuser access is required.");
      return;
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
  resetServerState();
  showLoginPanel();
  setStatus("Disconnected from mud server.");
}

async function onAuthenticated() {
  chatState.authenticated = true;
  hideLoginPanel();
  dom.chatBtnDisconnect.classList.remove("hidden");
  toggleServerControls(true);
  const worldsLoaded = await fetchWorlds();
  if (!worldsLoaded) return;
  setStatus("Connected to mud server.");
}

async function fetchWorlds() {
  try {
    const res = await fetch("/api/mud/worlds");
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      if (res.status === 401) {
        handleSessionExpired();
        return false;
      }
      if (res.status === 403) {
        try { await fetch("/api/mud/logout", { method: "POST" }); } catch { /* ignore */ }
        resetServerState();
        showLoginPanel(err.detail || "This mud server account is not authorised for the Axis Lab.");
        return false;
      }
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    chatState.worlds = data.worlds || [];
    populateWorldSelect();
    dom.chatWorldSelector?.classList.remove("hidden");
    return true;
  } catch (err) {
    setStatus(`Failed to fetch worlds: ${err.message}`);
    return false;
  }
}

function populateWorldSelect() {
  const sel = dom.chatWorldSelect;
  const enabled = chatState.worlds.filter(world => world.translation_enabled);
  const target = chatState.worldId
    || (enabled.length === 1 ? enabled[0].world_id : null)
    || (chatState.worlds.length === 1 ? chatState.worlds[0].world_id : null);

  if (!sel) {
    if (target) {
      chatState.worldId = target;
      selectWorld(target);
    }
    return;
  }

  sel.innerHTML = '<option value="">— select world —</option>';
  for (const world of chatState.worlds) {
    const opt = document.createElement("option");
    opt.value = world.world_id;
    opt.textContent = world.name || world.world_id;
    sel.appendChild(opt);
  }

  if (!target) return;
  sel.value = target;
  if (sel.value === target) {
    chatState.worldId = target;
    selectWorld(target);
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
    document.dispatchEvent(new CustomEvent("chat-world-config-cleared"));
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
    document.dispatchEvent(new CustomEvent("chat-world-config-applied"));
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
  resetServerState();
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

/**
 * Return the prompt text currently visible in the chat page UI.
 *
 * In server mode this is the server prompt textarea content. In standalone
 * mode this is the local IC prompt textarea content. The return value is the
 * literal prompt text currently shown to the user, regardless of whether it
 * will be sent as an explicit override or resolved server-side as the active
 * default template.
 *
 * @returns {string|null}
 */
export function getCurrentSystemPromptText() {
  if (isServerMode() && chatState.authenticated) {
    return dom.chatServerPromptText?.value.trim() || null;
  }
  return dom.chatSystemPrompt.value.trim() || null;
}

/**
 * Return the prompt override that should be sent with the next request.
 *
 * Server mode intentionally distinguishes between:
 * - the active world prompt selected on the server, which does not need an
 *   override when unmodified, and
 * - a different prompt file selected locally in developer mode, which must be
 *   sent even if the textarea is still identical to that file's content.
 *
 * @returns {string|null}
 */
export function getEffectiveSystemPrompt() {
  if (isServerMode() && chatState.authenticated) {
    const textarea = dom.chatServerPromptText;
    if (!textarea) return null;
    const text = textarea.value;
    const trimmed = text.trim() || null;
    const selectedFilename = dom.chatServerPromptSelect?.value || null;
    const activeFilename = chatState.worldPrompts.find((entry) => entry.is_active)?.filename || null;

    // Selecting a different prompt file in developer mode should immediately
    // affect subsequent translations, even when the textarea is otherwise
    // unmodified relative to that file.
    if (selectedFilename && selectedFilename !== activeFilename) {
      return trimmed;
    }

    if (text === chatState.serverPromptOriginal) return null;
    return trimmed;
  }
  return dom.chatSystemPrompt.value.trim() || null;
}

function wireWorldSelectEvent() {
  if (!dom.chatWorldSelect) return;
  dom.chatWorldSelect.addEventListener("change", () => {
    const worldId = dom.chatWorldSelect.value;
    chatState.worldId = worldId || null;
    selectWorld(worldId);
  });
}

function hideSharedWorldSelector() {
  if (dom.chatWorldSelector) {
    dom.chatWorldSelector.classList.add("hidden");
  }
}

/**
 * Wire all mud-server authentication, world-selection, and prompt-editor
 * events for the global Mud Server Session shell.
 *
 * Keeping these listeners inside the server-mode module avoids forcing the
 * parent chat controller to know about the server prompt/editor internals.
 *
 * @returns {void}
 */
export function wireServerModeEvents() {
  if (dom.chatModeSelect) {
    dom.chatModeSelect.addEventListener("change", async () => {
      try {
        const nextMode = dom.chatModeSelect.value;
        const serverUrl = nextMode === "development" ? dom.chatModeDevUrl?.value || null : null;
        await setRuntimeMode(nextMode, { serverUrl });
      } catch (err) {
        syncModeSelect();
        setStatus(`Mode switch error: ${err.message}`);
      }
    });
  }

  if (dom.chatModeDevUrl) {
    dom.chatModeDevUrl.addEventListener("input", () => {
      const value = dom.chatModeDevUrl.value.trim();
      chatState.developmentServerUrl = value || null;
      if (dom.chatModeDevApply) {
        dom.chatModeDevApply.disabled = value.length === 0;
      }
    });

    dom.chatModeDevUrl.addEventListener("keydown", async (event) => {
      if (event.key !== "Enter" || dom.chatModeSelect?.value !== "development") return;
      event.preventDefault();
      try {
        await setRuntimeMode("development", { serverUrl: dom.chatModeDevUrl.value });
      } catch (err) {
        setStatus(`Mode switch error: ${err.message}`);
      }
    });
  }

  if (dom.chatModeDevApply) {
    dom.chatModeDevApply.addEventListener("click", async () => {
      try {
        await setRuntimeMode("development", { serverUrl: dom.chatModeDevUrl?.value || null });
      } catch (err) {
        setStatus(`Mode switch error: ${err.message}`);
      }
    });
  }

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
  wireWorldSelectEvent();

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
    if (isServerMode()) {
      checkSession();
    } else {
      hideLoginPanel();
      dom.chatBtnDisconnect.classList.add("hidden");
      hideSharedWorldSelector();
      toggleServerControls(false);
    }
  });
}

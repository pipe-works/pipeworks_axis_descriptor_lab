/**
 * mod-artifact-editor.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Artifact Editor workflow for prompt templates and local Axis Payload JSON.
 *
 * Current scope
 * ─────────────
 * - browse local prompt files (including drafts)
 * - browse server-backed canonical world prompts via the lab backend
 * - browse local Axis Payload JSON files from app/examples plus drafts
 * - edit raw text in a single textarea
 * - inspect placeholder/schema reference metadata
 * - create new local draft files without overwriting shipped artifacts
 *
 * Server-backed mode remains prompt-only in this slice.  Axis Payload JSON
 * editing is local-only until the mud server exposes canonical JSON artifact
 * manifests for additional artifact types.
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";

const artifactState = {
  localListing: null,
  serverManifest: null,
  currentDocument: null,
};

const PLACEHOLDER_RE = /{{\s*([a-zA-Z0-9_]+)\s*}}/g;

function currentArtifactType() {
  return dom.artifactType.value;
}

function isPromptArtifact() {
  return currentArtifactType() === "prompt_template";
}

function isServerSource() {
  return dom.artifactSource.value === "server";
}

function currentPurpose() {
  return dom.artifactPurpose.value;
}

function currentReference() {
  if (isPromptArtifact()) {
    if (isServerSource()) {
      return artifactState.serverManifest?.reference ?? null;
    }
    return artifactState.localListing?.reference ?? artifactState.currentDocument?.reference ?? null;
  }

  return artifactState.localListing?.reference ?? artifactState.currentDocument?.reference ?? null;
}

function renderArtifactOptions(items, preferredName = "") {
  dom.artifactSelect.innerHTML = '<option value="">— choose —</option>';
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.name;

    let suffix = "";
    if (item.is_active) {
      suffix = " (active)";
    } else if (item.is_draft) {
      suffix = " (draft)";
    } else if (item.world_id) {
      suffix = ` (${item.world_id})`;
    }

    option.textContent = `${item.name}${suffix}`;
    dom.artifactSelect.appendChild(option);
  }
  if (preferredName) {
    dom.artifactSelect.value = preferredName;
  }
}

function setEditorBadge(label, active = false) {
  dom.artifactEditorBadge.textContent = label;
  dom.artifactEditorBadge.className = active ? "badge badge--active" : "badge badge--muted";
}

function renderMetaPanel(text) {
  dom.artifactMeta.textContent = text;
}

function renderPromptReference(reference) {
  const placeholderLines = reference.placeholders.length
    ? reference.placeholders.map((row) => `${row.placeholder}  ${row.description}`)
    : ["(no system-prompt placeholders for this prompt family)"];

  const noteLines = reference.notes.length ? reference.notes.map((note) => `- ${note}`) : ["- none"];
  const axesLine = reference.active_axes.length ? reference.active_axes.join(", ") : "(not axis-specific)";

  dom.artifactReference.textContent =
    `Placeholders\n${placeholderLines.join("\n")}\n\n` +
    `Active axes\n${axesLine}\n\n` +
    `Notes\n${noteLines.join("\n")}`;
}

function renderAxisPayloadReference(reference) {
  const fieldLines = reference.fields.map(
    (row) => `${row.name} (${row.type})  ${row.description}`
  );
  const noteLines = reference.notes.length ? reference.notes.map((note) => `- ${note}`) : ["- none"];

  dom.artifactReference.textContent =
    `Fields\n${fieldLines.join("\n")}\n\n` +
    `Notes\n${noteLines.join("\n")}\n\n` +
    `Sample JSON\n${reference.sample_json}`;
}

function renderReferencePanel(reference) {
  if (!reference) {
    dom.artifactReference.textContent = "Reference metadata unavailable.";
    return;
  }

  if (isPromptArtifact()) {
    renderPromptReference(reference);
    return;
  }

  renderAxisPayloadReference(reference);
}

function renderPromptPreview(reference, content, unknownPlaceholders) {
  let preview = content;
  for (const [key, value] of Object.entries(reference.sample_values || {})) {
    preview = preview.replaceAll(`{{${key}}}`, value);
  }

  const unknownLine = unknownPlaceholders.length
    ? `Unknown placeholders: ${unknownPlaceholders.join(", ")}`
    : "Unknown placeholders: none";
  const summaryLine = reference.profile_summary_example
    ? `profile_summary example\n${reference.profile_summary_example}\n\n`
    : "";

  dom.artifactPreview.textContent =
    `${unknownLine}\n\n${summaryLine}Rendered preview\n${preview.trim() || "(empty)"}`;
}

function renderAxisPayloadPreview(reference, content, parseError) {
  if (parseError) {
    dom.artifactPreview.textContent = `JSON parse error\n${parseError}`;
    return;
  }

  try {
    const parsed = JSON.parse(content);
    const normalised = JSON.stringify(parsed, null, 2);
    dom.artifactPreview.textContent =
      `JSON validation\nparse ok\n\nNormalised preview\n${normalised}\n\nSample JSON\n${reference.sample_json}`;
  } catch (err) {
    dom.artifactPreview.textContent = `JSON parse error\n${err.message}`;
  }
}

function validatePromptEditor() {
  const reference = currentReference();
  const content = dom.artifactEditor.value;
  const supported = new Set(
    (reference?.placeholders ?? []).map((row) => row.placeholder.slice(2, -2).trim())
  );
  const unknown = new Set();

  for (const match of content.matchAll(PLACEHOLDER_RE)) {
    const key = match[1].trim();
    if (supported.size > 0 && !supported.has(key)) {
      unknown.add(key);
    }
  }

  if (!content.trim()) {
    setEditorBadge("empty");
  } else if (unknown.size > 0) {
    setEditorBadge("warn");
  } else {
    setEditorBadge("ok", true);
  }

  renderPromptPreview(reference, content, [...unknown].sort());
}

function validateAxisPayloadEditor() {
  const reference = currentReference();
  const content = dom.artifactEditor.value;

  if (!content.trim()) {
    setEditorBadge("empty");
    renderAxisPayloadPreview(reference, content, null);
    return;
  }

  try {
    JSON.parse(content);
    setEditorBadge("json ok", true);
    renderAxisPayloadPreview(reference, content, null);
  } catch (err) {
    setEditorBadge("json err");
    renderAxisPayloadPreview(reference, content, err.message);
  }
}

function validateEditor() {
  if (isPromptArtifact()) {
    validatePromptEditor();
    return;
  }
  validateAxisPayloadEditor();
}

function renderLoadedDocument(doc, metaPrefix) {
  artifactState.currentDocument = doc;
  dom.artifactCurrentName.value = doc.name;
  dom.artifactEditor.value = doc.content;

  const metaLines = [
    metaPrefix,
    `name: ${doc.name}`,
    `origin: ${doc.origin_path}`,
    `draft: ${doc.is_draft ? "yes" : "no"}`,
  ];
  if (Object.hasOwn(doc, "purpose")) {
    metaLines.splice(2, 0, `purpose: ${doc.purpose}`);
  }
  if (Object.hasOwn(doc, "world_id")) {
    metaLines.splice(2, 0, `world: ${doc.world_id}`);
  }

  renderMetaPanel(metaLines.join("\n"));
  renderReferencePanel(doc.reference);
  validateEditor();
}

async function loadServerWorlds() {
  const res = await fetch("/api/mud/worlds");
  if (!res.ok) {
    throw new Error(`world list request failed (${res.status})`);
  }

  const data = await res.json();
  const worlds = data.worlds || [];
  dom.artifactWorld.innerHTML = '<option value="">— server world —</option>';
  for (const world of worlds) {
    if (!world.translation_enabled) continue;
    const option = document.createElement("option");
    option.value = world.world_id;
    option.textContent = world.name;
    dom.artifactWorld.appendChild(option);
  }
}

async function loadLocalPromptArtifacts() {
  const purpose = currentPurpose();
  const res = await fetch(
    `/api/artifacts/local/chat-prompts?purpose=${encodeURIComponent(purpose)}`
  );
  if (!res.ok) {
    throw new Error(`local prompt listing failed (${res.status})`);
  }

  artifactState.localListing = await res.json();
  artifactState.serverManifest = null;
  artifactState.currentDocument = null;
  renderArtifactOptions(artifactState.localListing.prompts);
  renderReferencePanel(artifactState.localListing.reference);
  renderMetaPanel(
    `Local prompt family\npurpose: ${artifactState.localListing.purpose}\nmode: create-only drafts`
  );
  dom.artifactCurrentName.value = "";
  dom.artifactEditor.value = "";
  validateEditor();
}

async function loadLocalAxisPayloadArtifacts() {
  const res = await fetch("/api/artifacts/local/axis-payloads");
  if (!res.ok) {
    throw new Error(`local axis payload listing failed (${res.status})`);
  }

  artifactState.localListing = await res.json();
  artifactState.serverManifest = null;
  artifactState.currentDocument = null;
  renderArtifactOptions(artifactState.localListing.payloads);
  renderReferencePanel(artifactState.localListing.reference);
  renderMetaPanel("Local AxisPayload JSON artifacts\nmode: create-only drafts");
  dom.artifactCurrentName.value = "";
  dom.artifactEditor.value = "";
  validateEditor();
}

async function loadServerArtifacts() {
  const worldId = dom.artifactWorld.value;
  if (!worldId) {
    renderArtifactOptions([]);
    renderMetaPanel("Select a mud-server world to inspect canonical prompt files.");
    renderReferencePanel(null);
    return;
  }

  const res = await fetch(`/api/artifacts/server/chat-prompts/${encodeURIComponent(worldId)}`);
  if (!res.ok) {
    throw new Error(`server prompt manifest failed (${res.status})`);
  }

  artifactState.serverManifest = await res.json();
  artifactState.localListing = null;
  artifactState.currentDocument = null;
  renderArtifactOptions(
    artifactState.serverManifest.prompts,
    artifactState.serverManifest.active_prompt_name || ""
  );
  renderReferencePanel(artifactState.serverManifest.reference);
  renderMetaPanel(
    `Canonical mud-server prompt manifest\n` +
      `world: ${artifactState.serverManifest.world_name} (${artifactState.serverManifest.world_id})\n` +
      `active prompt: ${artifactState.serverManifest.active_prompt_name || "(none)"}\n` +
      `writes: disabled`
  );

  if (artifactState.serverManifest.active_prompt_name) {
    await loadSelectedArtifact();
  } else {
    dom.artifactCurrentName.value = "";
    dom.artifactEditor.value = "";
    validateEditor();
  }
}

function syncControlState() {
  const server = isServerSource();
  const promptArtifact = isPromptArtifact();

  dom.artifactPurpose.disabled = !promptArtifact || server;
  dom.artifactWorld.disabled = !server || !promptArtifact;
  dom.artifactRefreshWorlds.disabled = !server || !promptArtifact;

  if (!promptArtifact && server) {
    dom.artifactSource.value = "local";
  }

  if (promptArtifact) {
    dom.artifactSaveHint.textContent = server
      ? "Canonical mud-server files are read-only here. Use Save local draft to clone the loaded prompt into the lab draft library."
      : "Local mode can create new draft files under app/prompts/*/drafts but never overwrites shipped prompt files.";
  } else {
    dom.artifactSaveHint.textContent =
      "Axis Payload JSON is local-only for now. Drafts are validated and saved under app/examples/drafts without overwriting shipped examples.";
  }
}

async function refreshArtifactSource() {
  syncControlState();

  if (!isPromptArtifact()) {
    try {
      await loadLocalAxisPayloadArtifacts();
      setStatus("Artifact Editor — loaded local Axis Payload JSON artifacts.");
    } catch (err) {
      setStatus(`Artifact Editor — ${err.message}.`);
    }
    return;
  }

  if (isServerSource()) {
    try {
      await loadServerWorlds();
      await loadServerArtifacts();
      setStatus("Artifact Editor — loaded mud-server world list.");
    } catch (err) {
      renderArtifactOptions([]);
      renderMetaPanel(
        "Server-backed prompt loading is unavailable until a mud-server session is active."
      );
      renderReferencePanel(null);
      setStatus(`Artifact Editor — ${err.message}.`);
    }
    return;
  }

  try {
    await loadLocalPromptArtifacts();
    setStatus("Artifact Editor — loaded local prompt artifacts.");
  } catch (err) {
    setStatus(`Artifact Editor — ${err.message}.`);
  }
}

async function loadSelectedArtifact() {
  const selectedName = dom.artifactSelect.value;
  if (!selectedName) {
    setStatus("Artifact Editor — choose an artifact to load.");
    return;
  }

  if (isPromptArtifact() && isServerSource()) {
    const prompt = artifactState.serverManifest?.prompts?.find((entry) => entry.name === selectedName);
    if (!prompt) {
      setStatus("Artifact Editor — selected server prompt is no longer available.");
      return;
    }
    renderLoadedDocument(
      {
        name: prompt.name,
        purpose: "chat_translation",
        content: prompt.content || "",
        is_draft: false,
        origin_path: prompt.origin_path,
        reference: artifactState.serverManifest.reference,
      },
      "Canonical mud-server prompt"
    );
    setStatus(`Artifact Editor — loaded server prompt '${selectedName}'.`);
    return;
  }

  const endpoint = isPromptArtifact()
    ? `/api/artifacts/local/chat-prompts/${encodeURIComponent(selectedName)}?purpose=${encodeURIComponent(
        currentPurpose()
      )}`
    : `/api/artifacts/local/axis-payloads/${encodeURIComponent(selectedName)}`;

  const res = await fetch(endpoint);
  if (!res.ok) {
    setStatus(`Artifact Editor — failed to load '${selectedName}' (${res.status}).`);
    return;
  }

  const doc = await res.json();
  renderLoadedDocument(doc, isPromptArtifact() ? "Local prompt artifact" : "Local AxisPayload JSON");
  setStatus(`Artifact Editor — loaded local artifact '${selectedName}'.`);
}

async function saveDraft() {
  const draftName = dom.artifactDraftName.value.trim();
  const content = dom.artifactEditor.value;
  if (!draftName) {
    setStatus("Artifact Editor — enter a draft name first.");
    return;
  }
  if (!content.trim()) {
    setStatus("Artifact Editor — cannot save an empty draft.");
    return;
  }

  const endpoint = isPromptArtifact()
    ? "/api/artifacts/local/chat-prompts/drafts"
    : "/api/artifacts/local/axis-payloads/drafts";
  const body = isPromptArtifact()
    ? {
        purpose: isServerSource() ? "chat_translation" : currentPurpose(),
        draft_name: draftName,
        content,
        based_on_name: dom.artifactCurrentName.value.trim() || null,
      }
    : {
        draft_name: draftName,
        content,
        based_on_name: dom.artifactCurrentName.value.trim() || null,
      };

  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const detail = await res.text();
    setStatus(`Artifact Editor — draft save failed (${res.status}): ${detail}`);
    return;
  }

  const created = await res.json();
  dom.artifactSource.value = "local";
  if (isPromptArtifact()) {
    dom.artifactPurpose.value = created.purpose;
  }
  dom.artifactDraftName.value = "";
  await refreshArtifactSource();
  dom.artifactSelect.value = created.name;
  await loadSelectedArtifact();
  setStatus(`Artifact Editor — created local draft '${created.name}'.`);
}

export async function initArtifactEditor() {
  await refreshArtifactSource();
}

export function wireArtifactEditorEvents() {
  dom.artifactType.addEventListener("change", refreshArtifactSource);
  dom.artifactSource.addEventListener("change", refreshArtifactSource);
  dom.artifactPurpose.addEventListener("change", refreshArtifactSource);
  dom.artifactWorld.addEventListener("change", loadServerArtifacts);
  dom.artifactRefreshWorlds.addEventListener("click", refreshArtifactSource);
  dom.artifactBtnLoad.addEventListener("click", loadSelectedArtifact);
  dom.artifactSelect.addEventListener("change", loadSelectedArtifact);
  dom.artifactEditor.addEventListener("input", validateEditor);
  dom.artifactSaveDraft.addEventListener("click", saveDraft);
  document.addEventListener("artifact-editor-activated", () => {
    if (isPromptArtifact() && isServerSource()) {
      refreshArtifactSource();
    }
  });
}

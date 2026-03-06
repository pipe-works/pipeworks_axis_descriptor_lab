/**
 * mod-artifact-editor.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Artifact Editor workflow for prompt templates and local deterministic JSON artifacts.
 *
 * Current scope
 * ─────────────
 * - browse local prompt files (including drafts)
 * - browse server-backed canonical world prompts and prompt drafts via the lab backend
 * - browse local Axis Payload JSON files from app/examples plus drafts
 * - browse local micro-indicator lexicon JSON files from app/data plus drafts
 * - browse local normalized world policy bundle JSON files plus drafts
 * - edit raw text in a single textarea
 * - inspect placeholder/schema reference metadata
 * - create new local draft files without overwriting shipped artifacts
 * - create new mud-server prompt and policy bundle drafts without overwriting canonical files
 * - promote mud-server prompt and policy bundle drafts explicitly
 *
 * Server-backed artifacts use the mud server as the canonical base. The editor
 * can create new draft files there, but it never overwrites canonical files.
 */

import { dom } from "./mod-state.js";
import { setStatus } from "./mod-status.js";
import { renderSourceHint } from "./mod-source-paths.js";

const artifactState = {
  localListing: null,
  serverManifest: null,
  serverPromptDraftListing: null,
  serverPolicyBundleListing: null,
  currentDocument: null,
};

const PLACEHOLDER_RE = /{{\s*([a-zA-Z0-9_]+)\s*}}/g;

function currentArtifactType() {
  return dom.artifactType.value;
}

function isPromptArtifact() {
  return currentArtifactType() === "prompt_template";
}

function isAxisPayloadArtifact() {
  return currentArtifactType() === "axis_payload";
}

function isPolicyBundleArtifact() {
  return currentArtifactType() === "policy_bundle";
}

function supportsServerSource() {
  return isPromptArtifact() || isPolicyBundleArtifact();
}

function isServerSource() {
  return dom.artifactSource.value === "server";
}

function isPromotableServerPromptDraft() {
  return (
    isPromptArtifact() &&
    isServerSource() &&
    artifactState.currentDocument?.is_draft === true &&
    artifactState.currentDocument?.purpose === "chat_translation"
  );
}

function isPromotableServerPolicyBundleDraft() {
  return (
    isPolicyBundleArtifact() &&
    isServerSource() &&
    artifactState.currentDocument?.is_draft === true
  );
}

function currentPurpose() {
  return dom.artifactPurpose.value;
}

function currentPromptPurposeForHint() {
  if (!isPromptArtifact()) {
    return "";
  }
  if (isServerSource()) {
    return "chat_translation";
  }
  return currentPurpose();
}

function getSelectedArtifactMeta() {
  const selectedName = dom.artifactSelect.value;
  if (!selectedName) {
    return null;
  }

  if (isPromptArtifact() && isServerSource()) {
    return (
      artifactState.serverManifest?.prompts?.find((entry) => entry.name === selectedName) ??
      artifactState.serverPromptDraftListing?.prompts?.find((entry) => entry.name === selectedName) ??
      null
    );
  }

  if (isPolicyBundleArtifact() && isServerSource()) {
    const listing = artifactState.serverPolicyBundleListing;
    if (!listing) {
      return null;
    }
    if (selectedName === listing.canonical.name) {
      return listing.canonical;
    }
    return listing.drafts.find((entry) => entry.name === selectedName) ?? null;
  }

  if (artifactState.localListing) {
    const entries = isPromptArtifact()
      ? artifactState.localListing.prompts || []
      : isAxisPayloadArtifact()
        ? artifactState.localListing.payloads || []
        : isPolicyBundleArtifact()
          ? artifactState.localListing.bundles || []
          : artifactState.localListing.lexicons || [];
    return entries.find((entry) => entry.name === selectedName) ?? null;
  }

  if (artifactState.currentDocument?.name === selectedName) {
    return artifactState.currentDocument;
  }

  return null;
}

function updateArtifactSelectSourceHint() {
  const emptyMessage = isPromptArtifact()
    ? "Source: select a prompt to view path."
    : isAxisPayloadArtifact()
      ? "Source: select an AxisPayload artifact to view path."
      : isPolicyBundleArtifact()
        ? "Source: select a policy bundle to view path."
        : "Source: select a lexicon artifact to view path.";

  renderSourceHint(
    dom.artifactSelectSourceHint,
    getSelectedArtifactMeta(),
    currentArtifactType(),
    currentPromptPurposeForHint(),
    emptyMessage
  );
}

function currentReference() {
  if (isPromptArtifact()) {
    if (isServerSource()) {
      return (
        artifactState.serverManifest?.reference ??
        artifactState.serverPromptDraftListing?.reference ??
        null
      );
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
    } else if (item.artifact_kind) {
      suffix = ` (${item.artifact_kind})`;
    } else if (item.world_id) {
      suffix = ` (${item.world_id})`;
    }

    option.textContent = `${item.name}${suffix}`;
    dom.artifactSelect.appendChild(option);
  }
  if (preferredName) {
    dom.artifactSelect.value = preferredName;
  }
  updateArtifactSelectSourceHint();
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

function renderStructuredJsonReference(reference, headingLabel = "Fields") {
  const fieldLines = reference.fields.map(
    (row) => `${row.name} (${row.type})  ${row.description}`
  );
  const noteLines = reference.notes.length ? reference.notes.map((note) => `- ${note}`) : ["- none"];

  dom.artifactReference.textContent =
    `${headingLabel}\n${fieldLines.join("\n")}\n\n` +
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

  if (isAxisPayloadArtifact()) {
    renderAxisPayloadReference(reference);
    return;
  }

  if (isPolicyBundleArtifact()) {
    renderStructuredJsonReference(reference, "Policy Bundle");
    return;
  }

  const heading =
    reference.artifact_kind === "catalog"
      ? "Contracts"
      : `Contract: ${reference.artifact_kind}`;
  renderStructuredJsonReference(reference, heading);
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

function renderLexiconPreview(reference, content, parseError) {
  if (parseError) {
    dom.artifactPreview.textContent = `JSON parse error\n${parseError}`;
    return;
  }

  try {
    const parsed = JSON.parse(content);
    const normalised = JSON.stringify(parsed, null, 2);
    const kindLine =
      reference?.artifact_kind && reference.artifact_kind !== "catalog"
        ? `Detected contract\n${reference.artifact_kind}\n\n`
        : "";
    dom.artifactPreview.textContent =
      `${kindLine}JSON validation\nparse ok\n\nNormalised preview\n${normalised}\n\nSample JSON\n${reference.sample_json}`;
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

  if (isAxisPayloadArtifact()) {
    validateAxisPayloadEditor();
    return;
  }

  const reference = currentReference();
  const content = dom.artifactEditor.value;

  if (!content.trim()) {
    setEditorBadge("empty");
    renderLexiconPreview(reference, content, null);
    return;
  }

  try {
    JSON.parse(content);
    setEditorBadge("json ok", true);
    renderLexiconPreview(reference, content, null);
  } catch (err) {
    setEditorBadge("json err");
    renderLexiconPreview(reference, content, err.message);
  }
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
  if (Object.hasOwn(doc, "artifact_kind")) {
    metaLines.splice(2, 0, `contract: ${doc.artifact_kind}`);
  }
  if (Object.hasOwn(doc, "version")) {
    metaLines.splice(2, 0, `version: ${doc.version}`);
  }
  if (Object.hasOwn(doc, "world_id")) {
    metaLines.splice(2, 0, `world: ${doc.world_id}`);
  }
  if (Object.hasOwn(doc, "source_kind")) {
    metaLines.splice(2, 0, `source: ${doc.source_kind}`);
  }

  renderMetaPanel(metaLines.join("\n"));
  renderReferencePanel(doc.reference);
  validateEditor();
  updateArtifactSelectSourceHint();
  syncControlState();
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
  artifactState.serverPromptDraftListing = null;
  artifactState.serverPolicyBundleListing = null;
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
  artifactState.serverPromptDraftListing = null;
  artifactState.serverPolicyBundleListing = null;
  artifactState.currentDocument = null;
  renderArtifactOptions(artifactState.localListing.payloads);
  renderReferencePanel(artifactState.localListing.reference);
  renderMetaPanel("Local AxisPayload JSON artifacts\nmode: create-only drafts");
  dom.artifactCurrentName.value = "";
  dom.artifactEditor.value = "";
  validateEditor();
}

async function loadLocalLexiconArtifacts() {
  const res = await fetch("/api/artifacts/local/lexicons");
  if (!res.ok) {
    throw new Error(`local lexicon listing failed (${res.status})`);
  }

  artifactState.localListing = await res.json();
  artifactState.serverManifest = null;
  artifactState.serverPromptDraftListing = null;
  artifactState.serverPolicyBundleListing = null;
  artifactState.currentDocument = null;
  renderArtifactOptions(artifactState.localListing.lexicons);
  renderReferencePanel(artifactState.localListing.reference);
  renderMetaPanel("Local deterministic lexicon JSON artifacts\nmode: create-only drafts");
  dom.artifactCurrentName.value = "";
  dom.artifactEditor.value = "";
  validateEditor();
}

async function loadLocalPolicyBundleArtifacts() {
  const res = await fetch("/api/artifacts/local/policy-bundles");
  if (!res.ok) {
    throw new Error(`local policy bundle listing failed (${res.status})`);
  }

  artifactState.localListing = await res.json();
  artifactState.serverManifest = null;
  artifactState.serverPromptDraftListing = null;
  artifactState.serverPolicyBundleListing = null;
  artifactState.currentDocument = null;
  renderArtifactOptions(artifactState.localListing.bundles);
  renderReferencePanel(artifactState.localListing.reference);
  renderMetaPanel("Local normalized policy bundle JSON artifacts\nmode: create-only drafts");
  dom.artifactCurrentName.value = "";
  dom.artifactEditor.value = "";
  validateEditor();
}

async function loadServerArtifacts(preferredName = "") {
  const worldId = dom.artifactWorld.value;
  if (!worldId) {
    renderArtifactOptions([]);
    renderMetaPanel("Select a mud-server world to inspect its canonical prompt files and drafts.");
    renderReferencePanel(null);
    return;
  }

  const [canonicalRes, draftsRes] = await Promise.all([
    fetch(`/api/artifacts/server/chat-prompts/${encodeURIComponent(worldId)}`),
    fetch(`/api/artifacts/server/chat-prompts/${encodeURIComponent(worldId)}/drafts`),
  ]);
  if (!canonicalRes.ok) {
    throw new Error(`server prompt manifest failed (${canonicalRes.status})`);
  }
  if (!draftsRes.ok) {
    throw new Error(`server prompt draft listing failed (${draftsRes.status})`);
  }

  artifactState.serverManifest = await canonicalRes.json();
  artifactState.serverPromptDraftListing = await draftsRes.json();
  artifactState.localListing = null;
  artifactState.serverPolicyBundleListing = null;
  artifactState.currentDocument = null;
  renderArtifactOptions(
    [...artifactState.serverManifest.prompts, ...(artifactState.serverPromptDraftListing.prompts || [])],
    preferredName || artifactState.serverManifest.active_prompt_name || ""
  );
  renderReferencePanel(artifactState.serverManifest.reference);
  renderMetaPanel(
    `Canonical mud-server prompt manifest\n` +
      `world: ${artifactState.serverManifest.world_name} (${artifactState.serverManifest.world_id})\n` +
      `active prompt: ${artifactState.serverManifest.active_prompt_name || "(none)"}\n` +
      `writes: create-only drafts`
  );

  if (
    preferredName ||
    artifactState.serverManifest.active_prompt_name
  ) {
    await loadSelectedArtifact();
  } else {
    dom.artifactCurrentName.value = "";
    dom.artifactEditor.value = "";
    validateEditor();
  }
}

async function loadServerPolicyBundleArtifact(preferredName = "") {
  const worldId = dom.artifactWorld.value;
  if (!worldId) {
    renderArtifactOptions([]);
    renderMetaPanel("Select a mud-server world to inspect its canonical policy bundle.");
    renderReferencePanel(null);
    return;
  }

  const [canonicalRes, draftsRes] = await Promise.all([
    fetch(`/api/artifacts/server/policy-bundles/${encodeURIComponent(worldId)}`),
    fetch(`/api/artifacts/server/policy-bundles/${encodeURIComponent(worldId)}/drafts`),
  ]);
  if (!canonicalRes.ok) {
    throw new Error(`server policy bundle request failed (${canonicalRes.status})`);
  }
  if (!draftsRes.ok) {
    throw new Error(`server policy bundle draft listing failed (${draftsRes.status})`);
  }

  const doc = await canonicalRes.json();
  const draftListing = await draftsRes.json();
  artifactState.serverManifest = null;
  artifactState.localListing = null;
  artifactState.serverPolicyBundleListing = {
    worldId,
    canonical: doc,
    drafts: draftListing.bundles || [],
  };
  artifactState.currentDocument = doc;
  renderArtifactOptions(
    [{ name: doc.name, world_id: doc.world_id, is_draft: false }, ...artifactState.serverPolicyBundleListing.drafts],
    preferredName || doc.name
  );
  if (preferredName && preferredName !== doc.name) {
    await loadSelectedArtifact();
    return;
  }
  renderLoadedDocument(doc, "Canonical mud-server policy bundle");
  setStatus(`Artifact Editor — loaded server policy bundle for '${worldId}'.`);
}

function syncControlState() {
  const server = isServerSource();
  const serverBackedArtifact = supportsServerSource();
  const promptArtifact = isPromptArtifact();
  const serverPolicyBundle = server && isPolicyBundleArtifact();
  const promotablePromptDraft = isPromotableServerPromptDraft();
  const promotablePolicyBundleDraft = isPromotableServerPolicyBundleDraft();

  dom.artifactPurpose.disabled = !promptArtifact || server;
  dom.artifactWorld.disabled = !server || !serverBackedArtifact;
  dom.artifactRefreshWorlds.disabled = !server || !serverBackedArtifact;
  dom.artifactPromoteDraft.classList.toggle(
    "hidden",
    !(promotablePromptDraft || promotablePolicyBundleDraft)
  );

  if (!serverBackedArtifact && server) {
    dom.artifactSource.value = "local";
  }

  if (promptArtifact) {
    dom.artifactSaveDraft.textContent = server ? "Save server draft" : "Save local draft";
    dom.artifactDraftNameLabel.textContent = promotablePromptDraft
      ? "Promotion target"
      : "New draft name";
    dom.artifactDraftName.placeholder = promotablePromptDraft
      ? "example_canonical_prompt"
      : "example_new_artifact";
    dom.artifactSaveHint.textContent = server
      ? promotablePromptDraft
        ? "Mud-server prompt drafts can be promoted explicitly. Promote draft creates a new canonical policies/<name>.txt file, updates the world's active prompt_template_path to it, and never overwrites an existing canonical file."
        : "Mud-server prompt files are canonical read-only bases here. Save server draft creates a new prompt under the world's policies/drafts directory and never overwrites canonical or existing draft files."
      : "Local mode can create new draft files under app/prompts/*/drafts but never overwrites shipped prompt files.";
  } else if (isAxisPayloadArtifact()) {
    dom.artifactSaveDraft.textContent = "Save local draft";
    dom.artifactDraftNameLabel.textContent = "New draft name";
    dom.artifactDraftName.placeholder = "example_new_artifact";
    dom.artifactSaveHint.textContent =
      "Axis Payload JSON is local-only for now. Drafts are validated and saved under app/examples/drafts without overwriting shipped examples.";
  } else if (isPolicyBundleArtifact()) {
    dom.artifactSaveDraft.textContent = serverPolicyBundle ? "Save server draft" : "Save local draft";
    dom.artifactDraftNameLabel.textContent = "New draft name";
    dom.artifactDraftName.placeholder = "example_new_artifact";
    dom.artifactSaveHint.textContent = serverPolicyBundle
      ? promotablePolicyBundleDraft
        ? "Mud-server policy bundle drafts can be promoted explicitly. Promote draft rewrites canonical policies/axes.yaml, policies/thresholds.yaml, and policies/resolution.yaml from the normalized draft, then reloads the world's axis engine. The draft file remains in place."
        : "Mud-server policy bundles are canonical read-only bases here. Save server draft creates a new JSON bundle under the world's policies/drafts directory and never overwrites existing files."
      : "Local mode validates Policy Bundle JSON drafts and saves them under app/artifacts/policy_bundles/drafts without overwriting shipped starter bundles.";
  } else {
    dom.artifactSaveDraft.textContent = "Save local draft";
    dom.artifactDraftNameLabel.textContent = "New draft name";
    dom.artifactDraftName.placeholder = "example_new_artifact";
    dom.artifactSaveHint.textContent =
      "Lexicon JSON is local-only for now. Drafts are validated and saved under app/data/drafts without overwriting shipped canonical files.";
  }
  updateArtifactSelectSourceHint();
}

async function refreshArtifactSource() {
  syncControlState();

  if (!isPromptArtifact()) {
    if (isServerSource() && isPolicyBundleArtifact()) {
      try {
        await loadServerWorlds();
        await loadServerPolicyBundleArtifact();
      } catch (err) {
        renderArtifactOptions([]);
        renderMetaPanel(
          "Server-backed policy bundle loading is unavailable until a mud-server session is active."
        );
        renderReferencePanel(null);
        setStatus(`Artifact Editor — ${err.message}.`);
      }
      return;
    }

    try {
      if (isAxisPayloadArtifact()) {
        await loadLocalAxisPayloadArtifacts();
        setStatus("Artifact Editor — loaded local Axis Payload JSON artifacts.");
      } else if (isPolicyBundleArtifact()) {
        await loadLocalPolicyBundleArtifacts();
        setStatus("Artifact Editor — loaded local policy bundle JSON artifacts.");
      } else {
        await loadLocalLexiconArtifacts();
        setStatus("Artifact Editor — loaded local lexicon JSON artifacts.");
      }
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
    if (prompt) {
      renderLoadedDocument(
        {
          name: prompt.name,
          purpose: "chat_translation",
          content: prompt.content || "",
          is_draft: false,
          origin_path: prompt.origin_path,
          source_kind: prompt.source_kind || "server",
          world_id: prompt.world_id || dom.artifactWorld.value,
          reference: artifactState.serverManifest.reference,
        },
        "Canonical mud-server prompt"
      );
      setStatus(`Artifact Editor — loaded server prompt '${selectedName}'.`);
      return;
    }

    const draft = artifactState.serverPromptDraftListing?.prompts?.find(
      (entry) => entry.name === selectedName
    );
    if (!draft) {
      setStatus("Artifact Editor — selected server prompt is no longer available.");
      return;
    }

    const res = await fetch(
      `/api/artifacts/server/chat-prompts/${encodeURIComponent(
        dom.artifactWorld.value
      )}/drafts/${encodeURIComponent(selectedName)}`
    );
    if (!res.ok) {
      setStatus(`Artifact Editor — failed to load server draft '${selectedName}' (${res.status}).`);
      return;
    }

    const doc = await res.json();
    artifactState.currentDocument = doc;
    renderLoadedDocument(doc, "Mud-server prompt draft");
    setStatus(`Artifact Editor — loaded server draft '${selectedName}'.`);
    return;
  }

  if (isPolicyBundleArtifact() && isServerSource()) {
    const listing = artifactState.serverPolicyBundleListing;
    if (!listing) {
      await loadServerPolicyBundleArtifact();
      return;
    }
    if (selectedName === listing.canonical.name) {
      renderLoadedDocument(listing.canonical, "Canonical mud-server policy bundle");
      setStatus(`Artifact Editor — loaded server policy bundle '${selectedName}'.`);
      return;
    }

    const res = await fetch(
      `/api/artifacts/server/policy-bundles/${encodeURIComponent(
        dom.artifactWorld.value
      )}/drafts/${encodeURIComponent(selectedName)}`
    );
    if (!res.ok) {
      setStatus(`Artifact Editor — failed to load server draft '${selectedName}' (${res.status}).`);
      return;
    }

    const doc = await res.json();
    artifactState.currentDocument = doc;
    renderLoadedDocument(doc, "Mud-server policy bundle draft");
    setStatus(`Artifact Editor — loaded server draft '${selectedName}'.`);
    return;
  }

  const endpoint = isPromptArtifact()
    ? `/api/artifacts/local/chat-prompts/${encodeURIComponent(selectedName)}?purpose=${encodeURIComponent(
        currentPurpose()
      )}`
    : isAxisPayloadArtifact()
      ? `/api/artifacts/local/axis-payloads/${encodeURIComponent(selectedName)}`
      : isPolicyBundleArtifact()
        ? `/api/artifacts/local/policy-bundles/${encodeURIComponent(selectedName)}`
        : `/api/artifacts/local/lexicons/${encodeURIComponent(selectedName)}`;

  const res = await fetch(endpoint);
  if (!res.ok) {
    setStatus(`Artifact Editor — failed to load '${selectedName}' (${res.status}).`);
    return;
  }

  const doc = await res.json();
  const metaPrefix = isPromptArtifact()
    ? "Local prompt artifact"
    : isAxisPayloadArtifact()
      ? "Local AxisPayload JSON"
      : isPolicyBundleArtifact()
        ? "Local policy bundle JSON"
        : "Local lexicon JSON";
  renderLoadedDocument(doc, metaPrefix);
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
  if (isPolicyBundleArtifact() && isServerSource() && !dom.artifactWorld.value) {
    setStatus("Artifact Editor — choose a mud-server world before saving a server draft.");
    return;
  }

  const endpoint = isPromptArtifact()
    ? isServerSource()
      ? `/api/artifacts/server/chat-prompts/${encodeURIComponent(dom.artifactWorld.value)}/drafts`
      : "/api/artifacts/local/chat-prompts/drafts"
    : isAxisPayloadArtifact()
      ? "/api/artifacts/local/axis-payloads/drafts"
      : isPolicyBundleArtifact()
        ? isServerSource()
          ? `/api/artifacts/server/policy-bundles/${encodeURIComponent(
              dom.artifactWorld.value
            )}/drafts`
          : "/api/artifacts/local/policy-bundles/drafts"
        : "/api/artifacts/local/lexicons/drafts";
  const body = isPromptArtifact()
    ? {
        draft_name: draftName,
        content,
        based_on_name: dom.artifactCurrentName.value.trim() || null,
        ...(isServerSource() ? {} : { purpose: currentPurpose() }),
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
  const savedToServer = isServerSource() && (isPromptArtifact() || isPolicyBundleArtifact());
  if (!savedToServer) {
    dom.artifactSource.value = "local";
    if (isPromptArtifact()) {
      dom.artifactPurpose.value = created.purpose;
    }
    await refreshArtifactSource();
    dom.artifactSelect.value = created.name;
    await loadSelectedArtifact();
  }
  dom.artifactDraftName.value = "";
  if (savedToServer) {
    if (isPromptArtifact()) {
      await loadServerArtifacts(created.name);
      setStatus(`Artifact Editor — created mud-server draft '${created.name}'.`);
      return;
    }
    await loadServerPolicyBundleArtifact(created.name);
    setStatus(`Artifact Editor — created mud-server draft '${created.name}'.`);
    return;
  }
  setStatus(`Artifact Editor — created local draft '${created.name}'.`);
}

async function promoteDraft() {
  const currentName = dom.artifactCurrentName.value.trim();
  const worldId = dom.artifactWorld.value;

  if (isPromotableServerPromptDraft()) {
    const targetName = dom.artifactDraftName.value.trim();
    if (!targetName) {
      setStatus("Artifact Editor — enter a promotion target name first.");
      return;
    }

    const res = await fetch(
      `/api/artifacts/server/chat-prompts/${encodeURIComponent(
        worldId
      )}/drafts/${encodeURIComponent(currentName)}/promote`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_name: targetName }),
      }
    );

    if (!res.ok) {
      const detail = await res.text();
      setStatus(`Artifact Editor — draft promotion failed (${res.status}): ${detail}`);
      return;
    }

    const promoted = await res.json();
    dom.artifactDraftName.value = "";
    await loadServerArtifacts(promoted.canonical_name);
    setStatus(
      `Artifact Editor — promoted '${promoted.name}' to canonical prompt '${promoted.canonical_name}'.`
    );
    return;
  }

  if (!isPromotableServerPolicyBundleDraft()) {
    setStatus("Artifact Editor — load a mud-server draft before promoting.");
    return;
  }

  const res = await fetch(
    `/api/artifacts/server/policy-bundles/${encodeURIComponent(
      worldId
    )}/drafts/${encodeURIComponent(currentName)}/promote`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    }
  );

  if (!res.ok) {
    const detail = await res.text();
    setStatus(`Artifact Editor — draft promotion failed (${res.status}): ${detail}`);
    return;
  }

  const promoted = await res.json();
  dom.artifactDraftName.value = "";
  await loadServerPolicyBundleArtifact();
  setStatus(`Artifact Editor — promoted '${promoted.name}' to canonical policy bundle.`);
}

export async function initArtifactEditor() {
  await refreshArtifactSource();
}

export function wireArtifactEditorEvents() {
  dom.artifactType.addEventListener("change", refreshArtifactSource);
  dom.artifactSource.addEventListener("change", refreshArtifactSource);
  dom.artifactPurpose.addEventListener("change", refreshArtifactSource);
  dom.artifactWorld.addEventListener("change", () => {
    if (!isServerSource()) return;
    if (isPromptArtifact()) {
      loadServerArtifacts();
      return;
    }
    if (isPolicyBundleArtifact()) {
      loadServerPolicyBundleArtifact();
    }
  });
  dom.artifactRefreshWorlds.addEventListener("click", refreshArtifactSource);
  dom.artifactBtnLoad.addEventListener("click", loadSelectedArtifact);
  dom.artifactSelect.addEventListener("change", () => {
    updateArtifactSelectSourceHint();
    loadSelectedArtifact();
  });
  dom.artifactEditor.addEventListener("input", validateEditor);
  dom.artifactSaveDraft.addEventListener("click", saveDraft);
  dom.artifactPromoteDraft.addEventListener("click", promoteDraft);
  document.addEventListener("artifact-editor-activated", () => {
    if (supportsServerSource() && isServerSource()) {
      refreshArtifactSource();
    }
  });
}

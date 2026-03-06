/**
 * mod-source-paths.js
 * -----------------------------------------------------------------------------
 * Shared helpers for displaying local artifact source paths in dropdown hints.
 *
 * The world-layout migration introduces multiple artifact roots (world-scoped,
 * lab-only, and legacy fallback). These helpers convert backend metadata into
 * stable, human-readable source labels so users can see which environment a
 * selected option comes from.
 */

const SOURCE_KIND_LABEL = {
  world_canonical: "world canonical",
  world_draft: "world draft",
  lab_only: "lab-only",
  legacy: "legacy fallback",
  server: "mud server",
};

/**
 * Infer a source kind when older payloads omit explicit source metadata.
 *
 * @param {Object} meta - Artifact metadata row.
 * @param {string} [meta.source_kind] - Explicit source kind when provided.
 * @param {string} [meta.origin_path] - Resolver origin path.
 * @returns {string} Normalized source kind token.
 */
function inferSourceKind(meta) {
  if (!meta) return "";
  if (meta.source_kind) return meta.source_kind;
  const originPath = String(meta.origin_path || "");
  if (originPath.startsWith("policies/drafts/")) return "world_draft";
  if (originPath.startsWith("policies/")) return "world_canonical";
  if (originPath.startsWith("drafts/")) return "lab_only";
  return "legacy";
}

/**
 * Build a local/project path for one artifact metadata row.
 *
 * @param {Object} meta - Artifact metadata row.
 * @param {string} meta.origin_path - Resolver origin path.
 * @param {string} [meta.world_id] - World id when world-scoped.
 * @param {string} artifactType - prompt_template|axis_payload|lexicon_json|policy_bundle
 * @param {string} [purpose] - prompt family for prompt artifacts.
 * @returns {{displayPath: string, sourceKind: string}}
 */
export function resolveArtifactSourcePath(meta, artifactType, purpose = "") {
  if (!meta) {
    return { displayPath: "(no source metadata)", sourceKind: "" };
  }

  const originPath = String(meta.origin_path || "");
  const worldId = String(meta.world_id || "pipeworks_web");
  const sourceKind = inferSourceKind(meta);

  if (sourceKind === "world_canonical" || sourceKind === "world_draft") {
    return {
      displayPath: `app/worlds/${worldId}/${originPath}`,
      sourceKind,
    };
  }

  if (sourceKind === "lab_only") {
    if (artifactType === "prompt_template") {
      return {
        displayPath: `app/lab_only/prompts/${purpose}/${originPath}`,
        sourceKind,
      };
    }
    if (artifactType === "axis_payload") {
      return {
        displayPath: `app/lab_only/axis/examples/${originPath}`,
        sourceKind,
      };
    }
    if (artifactType === "lexicon_json") {
      return {
        displayPath: `app/lab_only/axis/lexicons/${originPath}`,
        sourceKind,
      };
    }
    if (artifactType === "policy_bundle") {
      return {
        displayPath: `app/lab_only/policy_bundles/${originPath}`,
        sourceKind,
      };
    }
  }

  if (sourceKind === "legacy") {
    if (artifactType === "prompt_template") {
      return {
        displayPath: `app/prompts/${purpose}/${originPath}`,
        sourceKind,
      };
    }
    if (artifactType === "axis_payload") {
      return {
        displayPath: `app/examples/${originPath}`,
        sourceKind,
      };
    }
    if (artifactType === "lexicon_json") {
      return {
        displayPath: `app/data/${originPath}`,
        sourceKind,
      };
    }
    if (artifactType === "policy_bundle") {
      return {
        displayPath: `app/artifacts/policy_bundles/${originPath}`,
        sourceKind,
      };
    }
  }

  if (sourceKind === "server") {
    return {
      displayPath: originPath || "mud server artifact",
      sourceKind,
    };
  }

  return {
    displayPath: originPath || "(unknown)",
    sourceKind,
  };
}

/**
 * Render a concise source hint line under a dropdown.
 *
 * @param {HTMLElement|null} element - Target hint element.
 * @param {Object|null} meta - Artifact metadata row.
 * @param {string} artifactType - prompt_template|axis_payload|lexicon_json|policy_bundle
 * @param {string} [purpose] - Prompt family when artifactType=prompt_template.
 * @param {string} [emptyMessage] - Message shown when no selection exists.
 * @returns {void}
 */
export function renderSourceHint(
  element,
  meta,
  artifactType,
  purpose = "",
  emptyMessage = "Source: select an item to view path."
) {
  if (!element) return;
  if (!meta) {
    element.textContent = emptyMessage;
    return;
  }

  const resolved = resolveArtifactSourcePath(meta, artifactType, purpose);
  const label = SOURCE_KIND_LABEL[resolved.sourceKind] || "unknown";
  element.textContent = `Source: ${resolved.displayPath} (${label})`;
}

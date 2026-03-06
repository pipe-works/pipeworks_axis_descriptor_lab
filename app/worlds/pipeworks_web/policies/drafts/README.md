# Draft Policy Workspace (`pipeworks_web`)

Purpose:

- store safe, editable draft variants of world policy assets
- support experimentation without changing canonical policy files under `../policies/`
- preserve provenance by keeping draft files close to canonical structure

Rules:

- never edit canonical files directly from draft workflows
- prefer creating new, explicitly named draft files
- include a small header note in each draft file (why/when/by whom)
- keep test fixtures and runtime payload assumptions in sync with any promoted draft

Directory map:

- `translation/prompts/ic/` -> chat translation prompt drafts
- `axis/examples/` -> sample axis payload drafts for lab workflows
- `image/descriptor_layers/` -> descriptor layer text drafts
- `image/blocks/species/` -> species block YAML drafts
- `image/blocks/clothing/{environment,activity,wealth}/` -> clothing block text drafts
- `image/registries/` -> registry YAML drafts
- `policy_bundles/` -> normalized policy-bundle JSON drafts

# Artifact Editor

The Artifact Editor is the lab's text-box driven workspace for prompt and
deterministic JSON artifacts. Its purpose is to let you experiment locally,
inspect mud-server canonical state, and create draft or promotion candidates
without turning the lab into a second source of truth.

## Design rule

The mud server remains authoritative for world-backed artifacts.

- The lab may inspect canonical server artifacts.
- The lab may create new draft artifacts locally or on the mud server.
- The lab must not overwrite canonical server files during draft save.
- Promotion is an explicit workflow, separate from editing.

This is the same architecture rule used elsewhere in the project: the
deterministic system is authoritative; the lab is an experimentation surface.

## Source modes

The page has two source modes.

### Local drafts

Local mode is for lab-owned artifacts and safe iteration inside this repo.

Supported local artifact families:

- `Prompt Template`
  - `app/prompts/character_description/`
  - `app/prompts/chat_translation/`
- `Axis Payload JSON`
  - `app/examples/`
- `Policy Bundle JSON`
  - `app/artifacts/policy_bundles/`
- `Lexicon JSON`
  - `app/data/`

Draft saves are create-only and go to dedicated `drafts/` directories. Shipped
files are never overwritten by the editor.

### Mud server canonical

Server-backed mode uses `pipeworks_mud_server` as the canonical source for
world-backed artifacts.

Currently supported server-backed artifact families:

- `Prompt Template`
- `Policy Bundle JSON`

The editor loads canonical world artifacts read-only, plus any saved drafts
under the world's `policies/drafts/` directory.

## Supported workflows

### Prompt templates

The editor can:

- load local chat or character-description prompts
- load canonical mud-server world prompt files
- show placeholder/reference guidance for chat prompts
- create local prompt drafts
- create mud-server prompt drafts
- promote mud-server prompt drafts to canonical active prompt files

### Axis Payload JSON

The editor can:

- browse example payloads from `app/examples/`
- validate edits against the `AxisPayload` schema
- create local draft payloads under `app/examples/drafts/`

Axis Payload artifacts are local-only today.

### Policy Bundle JSON

The editor uses a normalized JSON form of the mud-server world policy package.
That normalized bundle combines the canonical:

- `policies/axes.yaml`
- `policies/thresholds.yaml`
- `policies/resolution.yaml`

into one text-editable JSON document.

The editor can:

- load local normalized policy bundle JSON
- load canonical mud-server normalized policy bundles
- create local policy bundle drafts
- create mud-server policy bundle drafts
- promote mud-server policy bundle drafts back into canonical YAML

### Lexicon JSON

The editor can browse and draft the deterministic micro-indicator lexicon
files in `app/data/`. These are local-only artifacts.

## Prompt behavior

### Chat Translation page versus Artifact Editor

There are two prompt-selection contexts in the lab.

- In standalone Chat Translation mode, the prompt dropdown is local and reads
  from the lab's `chat_translation` prompt family.
- In server Chat Translation mode, the page uses the mud server's canonical
  world prompt list.
- The Artifact Editor is the only place that shows mud-server drafts.

That means:

- creating a mud-server prompt draft does **not** make it appear in the Chat
  Translation page
- promoting a mud-server prompt draft does make it canonical and active

### What "active" means

For server-backed prompt artifacts, `active` means the prompt file currently
selected by the mud server world configuration:

- it matches `translation_layer.prompt_template_path` in the world's
  `world.json`
- it is the default prompt used by server-backed chat translation requests

### Runtime behavior after prompt promotion

Prompt promotion is live on the running mud server.

Promotion:

- writes a new canonical `policies/<name>.txt` file
- updates `world.json`
- reinitializes the world's translation service immediately

No process restart is required. The server starts using the new active prompt
for subsequent requests right away.

The Chat Translation page may still need a world reselect or page refresh to
show the updated prompt list in its UI.

## Policy bundle behavior

### What promotion does

Promoting a mud-server policy bundle draft:

- validates the normalized draft payload
- checks that the world's configured `translation_layer.active_axes` are still
  compatible with the promoted bundle
- rewrites canonical `axes.yaml`, `thresholds.yaml`, and `resolution.yaml`
- reloads the world's axis engine immediately

Like prompt promotion, this does not require a full server restart.

### Important limitation

Policy bundle promotion rewrites canonical YAML in deterministic machine
format. Existing comments and hand-formatted layout in the canonical YAML
files are not preserved.

## Safety rules

The editor follows four non-negotiable rules.

### Draft save is create-only

- local draft save never overwrites shipped files
- mud-server draft save never overwrites canonical files
- mud-server draft save never overwrites an existing draft filename

### Promotion is explicit

Saving a draft and promoting a draft are separate actions.

### Canonical server files stay authoritative

Server-backed editing is based on the mud server's current world state, not on
lab-local guesses.

### The lab is not a silent deploy surface

Canonical adoption requires an explicit promotion step. The editor never
silently changes a world's live configuration just because a draft was saved.

## Current limitations

The current implementation is deliberately narrower than the long-term plan.

- There is no generic `world-artifacts` manifest endpoint yet.
- There is no generic server-side validation endpoint.
- There is no PR/export-based promotion workflow yet.
- There is no stale-canonical hash guard on promotion yet.
- The Chat Translation page does not live-refresh when Artifact Editor
  promotion changes the server's canonical prompt.

These are hardening and scaling tasks rather than blockers for the current
draft-and-promote workflow.

# World-Scoped Lab Assets

This directory stores world-scoped assets that Axis Lab may read locally when
they are intentionally checked in for inspection or deterministic lab flows.

Goals:

- keep world-scoped local assets separate from `app/lab_only` experimental artifacts
- allow deterministic local reads via `app/path_resolver.py`
- make it obvious which local assets are world-scoped rather than lab-only

Each world follows:

- `worlds/<world_id>/policies/` for world-scoped assets aligned to current mud-server policy shape
- `worlds/<world_id>/policies/drafts/` for draft overlays

Important boundary:

- `pipeworks_mud_server` remains the runtime authority
- checked-in files here do not create a second source of truth
- this directory exists for deliberate local inspection and development support,
  not to preserve the older mirror-heavy workstation model

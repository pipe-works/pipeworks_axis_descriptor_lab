# World-Scoped Lab Assets

This directory stores world-scoped policy assets for Axis Lab migration work.

Goals:

- mirror MUD world policy structure so canonical and lab representations are easy to compare
- allow local reads using deterministic precedence via `app/path_resolver.py`
- separate canonical world assets from `app/lab_only` exploratory artifacts

Each world follows:

- `worlds/<world_id>/policies/` for canonical-style assets
- `worlds/<world_id>/policies/drafts/` for draft overlays

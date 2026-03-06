# Pipeworks Web Policy Layout (Axis Lab)

This directory mirrors the canonical policy shape from the MUD server for `pipeworks_web`.

It is scaffolded for migration and does not automatically imply canonical authority.
Canonical authority remains in `pipeworks_mud_server` APIs.

During migration, local reads may resolve from this folder, `app/lab_only`, or legacy
locations using deterministic precedence in `app/path_resolver.py`.

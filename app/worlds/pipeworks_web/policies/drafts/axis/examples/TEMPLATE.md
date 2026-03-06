# Template: Axis Payload Draft Example

Purpose:

- create deterministic sample payloads for lab testing and UI previews

Suggested filename:

- `<scenario_name>_draft_v1.json`

Suggested structure:

```json
{
  "axes": {
    "demeanor": {"label": "guarded", "score": 0.62},
    "health": {"label": "weary", "score": 0.41},
    "physique": {"label": "lean", "score": 0.58},
    "wealth": {"label": "modest", "score": 0.37},
    "facial_signal": {"label": "stern", "score": 0.66}
  },
  "policy_hash": "<optional_hash_or_placeholder>",
  "seed": 12345,
  "world_id": "pipeworks_web"
}
```

Notes:

- JSON does not support comments, so keep rationale in commit messages or companion docs
- keep axis keys aligned with active world policy axis names

# Template: Policy Bundle Draft (JSON)

Purpose:

- draft normalized policy bundles used for lab review and promotion workflows

Suggested filename:

- `pipeworks_web_policy_bundle_<variant>_draft_v1.json`

Suggested structure:

```json
{
  "world_id": "pipeworks_web",
  "version": "0.1.0",
  "source": "draft bundle",
  "policy_hash": null,
  "axes_order": ["physique", "wealth", "health"],
  "axes": {
    "physique": {
      "group": "character",
      "ordering": ["frail", "hunched", "lean"],
      "thresholds": [
        {"label": "frail", "min": 0.0, "max": 0.16}
      ]
    }
  },
  "chat_rules": {
    "channel_multipliers": {"say": 1.0, "yell": 1.5, "whisper": 0.5},
    "min_gap_threshold": 0.05,
    "axes": {
      "physique": {"resolver": "no_effect"}
    }
  }
}
```

Notes:

- JSON does not support comments; keep rationale in adjacent markdown or commit notes
- maintain strict axis/threshold/chat-rule consistency expected by validators

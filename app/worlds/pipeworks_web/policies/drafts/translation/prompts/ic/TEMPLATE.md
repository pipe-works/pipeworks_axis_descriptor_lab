# Template: Translation Prompt Draft

Purpose:

- draft world IC translation prompts used by canonical compile pathways

Suggested filename:

- `ic_prompt_<variant>_draft_v1.txt`

Suggested structure:

```text
[Draft metadata]
- Purpose: <what this prompt variant changes>
- Author: <name>
- Date: <YYYY-MM-DD>
- Based on: <canonical filename>

[System instructions]
<translation rules>

[Template placeholders]
- {{profile_summary}}
- {{channel}}
- {{player_input}}

[Output contract]
- one in-character line
- no out-of-character commentary
- obey channel intensity guidance
```

Notes:

- keep placeholders unchanged unless compiler contract changes
- avoid embedding policy logic that belongs in YAML/registries

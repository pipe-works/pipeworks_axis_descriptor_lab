# Template: Species Block Draft (YAML)

Purpose:

- draft species canon blocks used during image prompt compilation

Suggested filename:

- `<species>_draft_v1.yaml`

Suggested structure:

```yaml
# Draft metadata: purpose/author/date
id: goblin_draft_v1
version: 1
block_type: species
compatible_genders: [male, female]
render_priority: 100
prompt_block: |
  <species canon text>
```

Notes:

- YAML comments are supported; keep high-level rationale near changed fields
- keep `compatible_genders` aligned with identity contract values

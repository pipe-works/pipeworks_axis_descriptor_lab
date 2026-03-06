# Template: Image Registry Draft (YAML)

Purpose:

- draft registry metadata for species/clothing block selection

Suggested filenames:

- `species_registry_draft_v1.yaml`
- `clothing_registry_draft_v1.yaml`

Suggested structure:

```yaml
# Draft metadata: purpose/author/date
registry_id: species_registry_draft_v1
version: 1
policy_schema: pipeworks_policy_v1
entries:
  - id: goblin_draft_v1
    version: 1
    block_type: species
    compatible_species: [goblin]
    compatible_genders: [male, female]
    tags: [coastal]
    selection_rules:
      priority: 100
```

Notes:

- prefer explicit metadata in registries rather than inferring from filenames
- keep ids stable once referenced by manifests or bundles

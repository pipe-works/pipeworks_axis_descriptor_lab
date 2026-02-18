(policy-rules)=

# Policy Rules Reference

## Introduction

The Axis Descriptor Lab includes a **server-side relabelling endpoint**
(`POST /api/relabel`) that applies a hardcoded policy table to map normalised
axis scores (0.0--1.0) to human-readable labels.  In the UI this is the
**Auto (policy)** toggle.

The policy is intentionally simple and Pipe-Works-flavoured -- it is **not** a
substitute for a real policy engine.  It exists to:

1. **Make sliders meaningful** -- dragging a score slider produces a
   semantically appropriate label.
2. **Demonstrate policy drift** -- changing thresholds or labels shows how
   upstream policy decisions propagate through to LLM output.
3. **Support A/B comparison** -- relabelling after score changes enables
   controlled experiments (same score, different label vs. same label,
   different score).

:::{admonition} Pipe-Works Design Philosophy
:class: note

In the broader Pipe-Works ecosystem, policy would come from the MUD server's
authoritative game state, not from hardcoded thresholds.  The lab's policy
table is a stand-in for experimentation.
:::

## Mechanism

Matching uses **piecewise threshold lookup**.  For each known axis:

1. Thresholds are checked in ascending order.
2. The first threshold where `score < upper_bound` wins.
3. A final entry with an upper bound of 1.01 acts as the catch-all, ensuring
   scores of exactly 1.0 are captured.

**Unknown axes** (any axis name not listed in the policy table) are left
unchanged -- their existing labels are preserved as-is.

## Policy Table

### Age

| Score Range       | Label       |
|-------------------|-------------|
| 0.000 -- 0.249    | young       |
| 0.250 -- 0.499    | middle-aged |
| 0.500 -- 0.749    | old         |
| 0.750 -- 1.000    | ancient     |

4 labels.  Even spacing at 0.25 intervals.

---

### Demeanor

| Score Range       | Label     |
|-------------------|-----------|
| 0.000 -- 0.199    | cordial   |
| 0.200 -- 0.399    | guarded   |
| 0.400 -- 0.599    | resentful |
| 0.600 -- 0.799    | hostile   |
| 0.800 -- 1.000    | menacing  |

5 labels.  Even spacing at 0.20 intervals.  Progression from warm to
threatening.

---

### Dependency

| Score Range       | Label         |
|-------------------|---------------|
| 0.000 -- 0.329    | dispensable   |
| 0.330 -- 0.659    | necessary     |
| 0.660 -- 1.000    | indispensable |

3 labels.  Even thirds (~0.33 intervals).

---

### Facial Signal

| Score Range       | Label        |
|-------------------|--------------|
| 0.000 -- 0.299    | open         |
| 0.300 -- 0.599    | asymmetrical |
| 0.600 -- 1.000    | closed       |

3 labels.  Even thirds (~0.30 intervals).  Maps from readable expression to
guarded.

---

### Health

| Score Range       | Label    |
|-------------------|----------|
| 0.000 -- 0.249    | vigorous |
| 0.250 -- 0.499    | weary    |
| 0.500 -- 0.749    | ailing   |
| 0.750 -- 1.000    | failing  |

4 labels.  Even spacing at 0.25 intervals.  Progression from strong to dying.

---

### Legitimacy

| Score Range       | Label        |
|-------------------|--------------|
| 0.000 -- 0.249    | unchallenged |
| 0.250 -- 0.499    | tolerated    |
| 0.500 -- 0.649    | questioned   |
| 0.650 -- 0.799    | contested    |
| 0.800 -- 1.000    | illegitimate |

5 labels.  **Uneven spacing** -- the upper end is compressed (0.15 bands for
"questioned" and "contested") reflecting how authority erodes faster once
doubt sets in.

---

### Moral Load

| Score Range       | Label      |
|-------------------|------------|
| 0.000 -- 0.299    | clear      |
| 0.300 -- 0.599    | conflicted |
| 0.600 -- 1.000    | burdened   |

3 labels.  Even thirds (~0.30 intervals).

---

### Physique

| Score Range       | Label   |
|-------------------|---------|
| 0.000 -- 0.299    | gaunt   |
| 0.300 -- 0.449    | lean    |
| 0.450 -- 0.549    | stocky  |
| 0.550 -- 0.699    | hunched |
| 0.700 -- 1.000    | imposing|

5 labels.  **Uneven spacing** -- the middle bands ("lean", "stocky",
"hunched") are narrower (0.10--0.15), creating a tighter cluster around the
midpoint.  This makes the neutral range more granular while the extremes
("gaunt", "imposing") occupy wider bands.

---

### Risk Exposure

| Score Range       | Label     |
|-------------------|-----------|
| 0.000 -- 0.329    | sheltered |
| 0.330 -- 0.659    | hazardous |
| 0.660 -- 1.000    | perilous  |

3 labels.  Even thirds (~0.33 intervals).

---

### Visibility

| Score Range       | Label     |
|-------------------|-----------|
| 0.000 -- 0.329    | obscure   |
| 0.330 -- 0.659    | routine   |
| 0.660 -- 1.000    | prominent |

3 labels.  Even thirds (~0.33 intervals).

---

### Wealth

| Score Range       | Label      |
|-------------------|------------|
| 0.000 -- 0.249    | destitute  |
| 0.250 -- 0.449    | threadbare |
| 0.450 -- 0.549    | well-kept  |
| 0.550 -- 0.749    | comfortable|
| 0.750 -- 1.000    | affluent   |

5 labels.  **Uneven spacing** -- the middle band ("well-kept") is deliberately
narrow (0.10), creating a slim "just adequate" zone.  The extremes
("destitute", "affluent") occupy wider bands.

---

## Summary

| Axis           | Labels | Spacing       | Notes                    |
|----------------|--------|---------------|--------------------------|
| age            | 4      | Even (0.25)   |                          |
| demeanor       | 5      | Even (0.20)   |                          |
| dependency     | 3      | Even (0.33)   |                          |
| facial_signal  | 3      | Even (0.30)   |                          |
| health         | 4      | Even (0.25)   |                          |
| legitimacy     | 5      | Uneven        | Upper end compressed     |
| moral_load     | 3      | Even (0.30)   |                          |
| physique       | 5      | Uneven        | Tight midpoint cluster   |
| risk_exposure  | 3      | Even (0.33)   |                          |
| visibility     | 3      | Even (0.33)   |                          |
| wealth         | 5      | Uneven        | Narrow middle band       |

**Total:** 11 axes, 43 labels across all axes.

## Implementation Details

Source location
: `app/main.py`, inside the `relabel()` function (lines 509--575)

Endpoint
: `POST /api/relabel`

Data structure
: `dict[str, list[tuple[float, str]]]` -- axis name maps to a list of
  `(upper_bound_exclusive, label)` tuples

Matching algorithm
: Linear scan; first threshold where `score < upper_bound` wins

Catch-all
: Final tuple uses 1.01 as upper bound to capture scores of exactly 1.0

Statefulness
: The policy is hardcoded and stateless -- no database, no config file, no
  runtime modification

Unknown axes
: Passed through unchanged (label preserved as-is)

Return value
: New `AxisPayload` with updated labels; all non-axis fields (`seed`,
  `policy_hash`, `world_id`) unchanged

## Potential Improvements

:::{admonition} Not Currently Planned
:class: tip

These are noted for completeness.  The current hardcoded policy is sufficient
for the lab's experimental purpose.
:::

- Externalise the policy table to a YAML/JSON config file for easier editing
- Add API endpoint to inspect/modify the policy at runtime
- Support per-world policy overrides (different worlds could have different
  label vocabularies)
- Add policy versioning to track how threshold changes affect LLM output
  over time
- Include the policy table hash in IPC calculations (currently only
  `policy_hash` from the payload is used)

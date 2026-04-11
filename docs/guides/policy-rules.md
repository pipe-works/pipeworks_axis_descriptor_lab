(policy-rules)=

# Policy Rules Reference

## Introduction

The Axis Descriptor Lab includes a server-side relabelling endpoint,
`POST /api/relabel`, used by the **Auto (policy)** toggle in the UI.

That endpoint tracks the current axis policy shipped in
`pipeworks_mud_server` for the bundled world:

- `data/worlds/pipeworks_web/policies/axes.yaml`
- `data/worlds/pipeworks_web/policies/thresholds.yaml`

The mud server remains the runtime authority. Axis Lab keeps a checked-in
lab-side copy of the current ordering and thresholds in
`app/relabel_policy.py` so the UI can inspect and relabel deterministically
without expanding the older local-mirror model into a second source of truth.

## Mechanism

Matching uses **inclusive score ranges** derived from the mud-server
`thresholds.yaml` files.

For each known axis:

1. Ranges are checked in declared order.
2. A label matches when `min_score <= score <= max_score`.
3. Unknown axes are left unchanged.
4. If a score falls outside every configured range, the existing label is
   preserved because the lab schema requires a non-empty string label.

## Canonical Axis Order

The standalone slider UI uses the same full-axis order as the mud-server
`axes.yaml` file:

1. `physique`
2. `wealth`
3. `health`
4. `demeanor`
5. `age`
6. `facial_signal`
7. `legitimacy`
8. `visibility`
9. `moral_load`
10. `dependency`
11. `risk_exposure`

When the chat page is connected to a mud server, the selected world's
`active_axes` order is shown first, and any remaining axes fall back to the
canonical order above.

## Policy Table

### Physique

| Score Range    | Label  |
|----------------|--------|
| 0.00 -- 0.16   | frail  |
| 0.17 -- 0.32   | hunched |
| 0.33 -- 0.48   | skinny |
| 0.49 -- 0.64   | wiry   |
| 0.65 -- 0.80   | broad  |
| 0.81 -- 1.00   | stocky |

### Wealth

| Score Range    | Label     |
|----------------|-----------|
| 0.00 -- 0.19   | poor      |
| 0.20 -- 0.39   | modest    |
| 0.40 -- 0.59   | well-kept |
| 0.60 -- 0.79   | wealthy   |
| 0.80 -- 1.00   | decadent  |

### Health

| Score Range    | Label   |
|----------------|---------|
| 0.00 -- 0.19   | sickly  |
| 0.20 -- 0.39   | limping |
| 0.40 -- 0.59   | weary   |
| 0.60 -- 0.79   | scarred |
| 0.80 -- 1.00   | hale    |

### Demeanor

| Score Range    | Label      |
|----------------|------------|
| 0.00 -- 0.19   | timid      |
| 0.20 -- 0.39   | suspicious |
| 0.40 -- 0.59   | resentful  |
| 0.60 -- 0.79   | alert      |
| 0.80 -- 1.00   | proud      |

### Age

| Score Range    | Label       |
|----------------|-------------|
| 0.00 -- 0.24   | young       |
| 0.25 -- 0.49   | middle-aged |
| 0.50 -- 0.74   | old         |
| 0.75 -- 1.00   | ancient     |

### Facial Signal

| Score Range    | Label          |
|----------------|----------------|
| 0.00 -- 0.14   | understated    |
| 0.15 -- 0.29   | pronounced     |
| 0.30 -- 0.44   | exaggerated    |
| 0.45 -- 0.59   | asymmetrical   |
| 0.60 -- 0.74   | weathered      |
| 0.75 -- 0.89   | soft-featured  |
| 0.90 -- 1.00   | sharp-featured |

### Legitimacy

| Score Range    | Label      |
|----------------|------------|
| 0.00 -- 0.24   | sanctioned |
| 0.25 -- 0.49   | tolerated  |
| 0.50 -- 0.74   | questioned |
| 0.75 -- 1.00   | illicit    |

### Visibility

| Score Range    | Label       |
|----------------|-------------|
| 0.00 -- 0.24   | hidden      |
| 0.25 -- 0.49   | discrete    |
| 0.50 -- 0.74   | routine     |
| 0.75 -- 1.00   | conspicuous |

### Moral Load

| Score Range    | Label      |
|----------------|------------|
| 0.00 -- 0.24   | neutral    |
| 0.25 -- 0.49   | burdened   |
| 0.50 -- 0.74   | conflicted |
| 0.75 -- 1.00   | corrosive  |

### Dependency

| Score Range    | Label       |
|----------------|-------------|
| 0.00 -- 0.24   | optional    |
| 0.25 -- 0.49   | useful      |
| 0.50 -- 0.74   | necessary   |
| 0.75 -- 1.00   | unavoidable |

### Risk Exposure

| Score Range    | Label      |
|----------------|------------|
| 0.00 -- 0.24   | benign     |
| 0.25 -- 0.49   | straining  |
| 0.50 -- 0.74   | hazardous  |
| 0.75 -- 1.00   | eroding    |

## Summary

| Axis           | Labels | Order Source |
|----------------|--------|--------------|
| physique       | 6      | `axes.yaml`  |
| wealth         | 5      | `axes.yaml`  |
| health         | 5      | `axes.yaml`  |
| demeanor       | 5      | `axes.yaml`  |
| age            | 4      | `axes.yaml`  |
| facial_signal  | 7      | `axes.yaml`  |
| legitimacy     | 4      | `axes.yaml`  |
| visibility     | 4      | `axes.yaml`  |
| moral_load     | 4      | `axes.yaml`  |
| dependency     | 4      | `axes.yaml`  |
| risk_exposure  | 4      | `axes.yaml`  |

**Total:** 11 axes, 52 configured labels.

## Implementation Details

Source location
: `app/relabel_policy.py`

Endpoint
: `POST /api/relabel`

Data structures
: `AXIS_ORDER`, `AXIS_LABEL_ORDER`, and
  `RELABEL_POLICY: dict[str, list[tuple[float, float, str]]]`

Matching algorithm
: Inclusive range scan; first tuple satisfying `min_score <= score <= max_score`
  wins

Unknown axes
: Passed through unchanged

Unmatched scores
: Existing label preserved, because the lab payload schema requires a
  non-empty string label

Return value
: New `AxisPayload` with relabelled axes and unchanged `policy_hash`, `seed`,
  and `world_id`

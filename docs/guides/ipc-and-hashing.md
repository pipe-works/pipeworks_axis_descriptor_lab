(ipc-and-hashing)=

# Interpretive Provenance Chain (IPC) and Hashing System

## Introduction

The Axis Descriptor Lab generates descriptive text from deterministic axis
payloads using a constrained LLM layer (Ollama).  The system is built on a
simple premise: a set of numerical scores and labels (the *axes*) are
authoritative facts about an entity, and the LLM produces prose that
*interprets* those facts without overriding them.

This raises a question that is deceptively difficult to answer:

> When the output changes, what caused the change?

Was it a different axis score?  A modified system prompt?  A model upgrade?
A different sampling temperature?  Or just the inherent randomness of
language model inference?

Without a systematic way to fingerprint every variable that influences a
generation, any experiment with the lab is **observational** -- you can see
what happened, but you cannot determine *why*.  Behavioural shifts may be
misattributed to axis changes when the real cause was a prompt edit.  Model
upgrades may alter output in ways that are invisible without a baseline.
Seed variance may introduce noise that masks real signal.

The **Interpretive Provenance Chain (IPC)** is the project's answer to this
problem.  It is a composite SHA-256 fingerprint of *every* variable that
influences a generation, implemented as a chain of individual hashes
combined into a single identifier.  The guarantee is simple:

> Two generations with the same IPC ID used identical inputs in every
> respect.  If their outputs differ, the difference is attributable solely
> to LLM stochasticity.

The IPC upgrades the lab from an observational tool to a reproducible
scientific instrument.

:::{admonition} Pipe-Works Design Philosophy
:class: note

The IPC is grounded in four principles from the broader Pipe-Works project:

- **Determinism over optimisation** -- the system controls what it can; the
  LLM is allowed to vary only within fingerprinted boundaries.
- **Inspectability over mystique** -- every variable is visible, hashed,
  and stored.
- **Programmatic truth over narrative authority** -- the LLM interprets
  but never decides.
- **Failure as data** -- even when the LLM produces unexpected output,
  the IPC ensures the conditions are recorded for analysis.

*Source: `_working/axis_lab/pipeworks_interpretive_provenance_chains.md`,
section 8.*
:::

**What this document covers:**

1. The authoritative/ornamental boundary that makes IPC meaningful
2. The four hash functions and their normalisation rules
3. The normalisation philosophy in depth
4. A complete generation walkthrough showing the chain in action
5. Every integration point in the codebase
6. Practical use cases for drift detection and reproducibility
7. Design decisions and trade-offs
8. How the hashing system is tested
9. References to the original design documents
10. A glossary of key terms

---

## The Authoritative/Ornamental Boundary

The Axis Descriptor Lab operates on a strict two-layer architecture.
Understanding this boundary is essential to understanding why the IPC
exists and what it fingerprints.

### The Authoritative Layer (Deterministic)

The authoritative layer consists of data that the system controls
completely:

- **Axis scores and labels** -- numerical values (0.0--1.0) and their
  human-readable descriptors (e.g., "weary", "threadbare").  These are
  the entity's ground truth.
- **Policy rules** -- the mapping table that assigns labels to score
  ranges (e.g., age 0.75 maps to "old").  Identified by a `policy_hash`.
- **Seed** -- the deterministic RNG seed that produced the axis scores.
  Same seed, same scores.
- **World ID** -- the Pipe-Works world context identifier.

### The Ornamental Layer (Stochastic)

The ornamental layer is the LLM's output:

- A paragraph of descriptive prose that *interprets* the authoritative
  data.
- It modulates tone, chooses words, and compresses structured state into
  natural language.
- It **never** makes decisions, overrides scores, or introduces facts
  not present in the payload.

This principle is enforced at every level of the system.  The system
prompt itself contains the line:

> "The system is authoritative.  You are ornamental."
>
> -- `app/prompts/system_prompt_v01.txt`, line 29

### Why the Boundary Matters for Hashing

The boundary between authoritative and ornamental is *exactly* the
boundary the IPC fingerprints.  The input side (payload, prompt, model,
parameters) is the **experimental condition**.  The output side (LLM text)
is the **experimental observation**.

```text
AUTHORITATIVE (deterministic)           ORNAMENTAL (stochastic)
=================================       ========================
 AxisPayload (axes, scores, seed)        LLM-generated paragraph
 Policy rules (policy_hash)              - interprets, never decides
 System prompt (constraint text)         - modulates tone
 Model + sampling parameters             - compresses state to prose
          |                                        |
          +---- IPC fingerprints this boundary ----+
                      completely
```

The IPC captures the complete experimental condition so that
observations can be meaningfully compared.  If two runs share the
same IPC ID, any difference in output is pure stochastic variation --
not a change in the experiment.

---

## The Four Hashes

The IPC system uses four SHA-256 hash functions, each targeting a
different component of the generation pipeline.  All are implemented in
the dedicated `app/hashing.py` module and return 64-character lowercase
hexadecimal digest strings.

### Payload Hash (`compute_payload_hash`)

**What it fingerprints:**  The complete deterministic entity state --
all axis labels, all axis scores, the world ID, the policy hash, and
the seed.  This is the authoritative data that drives the generation.

**Normalisation:**  The payload dictionary is serialised to JSON with
`sort_keys=True` to eliminate Python dict insertion-order variation.
`ensure_ascii=False` preserves Unicode characters faithfully.

```python
canonical = json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)
return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

**Key design choice:**  The function accepts a plain `dict`, not a
Pydantic model, to keep the hashing module dependency-free.  Callers
convert their models first:

```python
input_hash = compute_payload_hash(payload.model_dump())
```

**Example:**

Given this payload:

```json
{
  "axes": {
    "health": {"label": "weary", "score": 0.5},
    "age": {"label": "old", "score": 0.7}
  },
  "policy_hash": "abc123",
  "seed": 42,
  "world_id": "test_world"
}
```

The canonical JSON (sorted keys) is:

```json
{"axes": {"age": {"label": "old", "score": 0.7}, "health": {"label": "weary", "score": 0.5}}, "policy_hash": "abc123", "seed": 42, "world_id": "test_world"}
```

Note how `age` now appears before `health` due to key sorting.  This
ensures that the same payload always produces the same hash, regardless
of the order keys were inserted in Python.

*See also: {func}`app.hashing.compute_payload_hash`*

### System Prompt Hash (`compute_system_prompt_hash`)

**What it fingerprints:**  The constraint layer -- the system prompt
that tells the LLM what it is and is not allowed to do.

**Why it matters:**  System prompts evolve.  Small changes such as
adding "avoid metaphor", changing sentence limits, or adjusting
escalation rules can materially change LLM behaviour.  Without hashing
the prompt, behavioural shifts may be misattributed to axis changes or
model differences.  The prompt text is part of the experimental
condition and must be fingerprinted.

*Source: `_working/axis_lab/pipeworks_lab_prompt_hash.md`, section 3.*

**Normalisation rules** (applied in order):

1. Split the text into individual lines.
2. Strip leading and trailing whitespace from **each** line.  This
   removes editor-introduced indentation and trailing spaces without
   altering the words on the line.
3. Rejoin the stripped lines with `\n`.
4. Strip leading and trailing blank lines from the **entire** result.
   Internal blank lines (paragraph breaks) are preserved because they
   may carry structural meaning in multi-section prompts.
5. Do **not** lowercase.  Case is semantic -- "NEVER" and "never" may
   carry different emphasis for the LLM.

**Example:**

These two prompt strings produce the **same** hash:

```text
Raw A:  "  line one  \n  line two  "
Raw B:  "line one\nline two"
```

Both normalise to `"line one\nline two"` before hashing.

*See also: {func}`app.hashing.compute_system_prompt_hash` and
{func}`app.hashing._normalise_system_prompt`*

### Output Hash (`compute_output_hash`)

**What it fingerprints:**  The observable result -- the exact text the
LLM produced.

**Why it exists:**  The output hash enables detecting whether two runs
produced identical output, even when the input conditions were the
same.  If two generations share the same IPC ID but have different
output hashes, the LLM exhibited stochastic drift -- it produced
different prose from identical inputs.

**Normalisation rules** (applied in order):

1. Strip leading and trailing whitespace from the entire string.
2. Collapse runs of two or more consecutive ASCII space characters
   (`U+0020`) into a single space.  Only ASCII space is targeted --
   newlines, tabs, and other whitespace are left intact so that
   paragraph structure is preserved.
3. Preserve punctuation exactly as-is.
4. Preserve letter casing exactly as-is.
5. Preserve sentence and line order exactly as-is.

The core normalisation is a single regex:

```python
collapsed = re.sub(r" {2,}", " ", stripped)
```

This targets only ASCII space runs (`" {2,}"`), not the broader `\s`
pattern, which would also collapse newlines and tabs.

:::{important}
The output hash normalisation is intentionally **more conservative** than
the system prompt normalisation.  LLM output may use tabs or specific
spacing patterns as part of its structure.  Only the most clearly
non-semantic variation (runs of multiple spaces) is normalised.
:::

*See also: {func}`app.hashing.compute_output_hash` and
{func}`app.hashing._normalise_output`*

### IPC Identifier (`compute_ipc_id`)

**What it fingerprints:**  The complete provenance chain -- every
variable that could influence the generation, combined into a single
identifier.

**Formula:**

```text
IPC_ID = SHA-256(
    input_hash
    + ":" + system_prompt_hash
    + ":" + model
    + ":" + str(temperature)
    + ":" + str(max_tokens)
    + ":" + str(seed)
)
```

**Why a composite hash:**  Combining component hashes into a single
identifier means you can answer "did these two generations use
identical conditions?" with a single string comparison rather than
comparing six fields individually.  This is especially valuable when
grouping runs in log analysis or drift detection.

**The colon delimiter:**  The colon (`:`) is a non-hex character that
prevents collisions from field concatenation.  Without it, the
concatenation `input_hash="ab"` + `system_prompt_hash="cd"` would
produce the same string as `input_hash="abc"` + `system_prompt_hash="d"`.
The delimiter makes these unambiguous: `"ab:cd"` vs `"abc:d"`.

*Source: Implementation recommendation from
`_working/axis_lab/pipeworks_axis_descriptor_lab_proposed_enhancements.md`.*

**Example:**

```python
ipc_id = compute_ipc_id(
    input_hash="d8bd1395713454e4...",       # 64-char hex
    system_prompt_hash="7f3a9c6e4b8f1d2c...", # 64-char hex
    model="gemma2:2b",
    temperature=0.2,
    max_tokens=120,
    seed=2954173979,
)
# Returns: "4a2e7f91..." (64-char hex)
```

If any single field changes -- even the seed by one digit -- the IPC
ID changes.  This is verified by the test suite, which modifies each
field individually and asserts a different hash.

*See also: {func}`app.hashing.compute_ipc_id`*

---

## Normalisation Philosophy

Normalisation is the most subtle and error-prone part of the hashing
system.  Getting it wrong silently undermines the entire provenance
chain.  This section explains the conservative approach in depth.

### The Normalisation Dilemma

There is a tension between two failure modes:

- **Too aggressive** (e.g., lowercasing, removing all whitespace):
  Semantically different texts hash identically, creating
  false-positive matches.  You think two runs used the same prompt,
  but they did not.
- **Too conservative** (e.g., hashing raw bytes):  Identical texts
  with trivial formatting differences hash differently, creating
  false-positive mismatches.  You think two runs used different
  prompts, but the only difference was a trailing space added by an
  editor.

The system must find the precise boundary between semantic and
non-semantic variation.

### Normalisation Principles

| Rule | Rationale | Applied To |
|------|-----------|------------|
| Never lowercase | Case carries semantic meaning ("NEVER" vs "never") | Prompts, Output |
| Preserve internal structure | Line order, sentence order, paragraph breaks may be meaningful | Prompts, Output |
| Strip edge whitespace | Editors often add trailing spaces/newlines; these are noise | Prompts, Output |
| Collapse 2+ spaces to 1 | LLMs sometimes produce inconsistent spacing; this is noise | Output only |
| Strip per-line whitespace | Editor indentation is noise in prompt files | Prompts only |
| Preserve tabs in output | Tab characters in LLM output may represent structure | Output only |

### Why Prompts and Outputs Have Different Rules

**System prompts** are human-authored text files that accumulate editor
artifacts over time: indentation from copy-paste, trailing spaces from
line editing, blank lines added by auto-formatters.  The per-line strip
is appropriate because no human writes a prompt where leading
indentation is semantically meaningful to the LLM.

**LLM output** is machine-generated and may use tabs or specific spacing
patterns as part of its structure.  Only the most clearly non-semantic
variation -- runs of multiple consecutive spaces -- is normalised.  The
regex `re.sub(r" {2,}", " ", text)` targets only ASCII space (`U+0020`),
never `\s`, so newlines, tabs, and other whitespace are preserved.

### What Normalisation Does NOT Do

The following operations are intentionally avoided:

- **Does not remove duplicate sentences** -- repetition may be a
  legitimate LLM output pattern.
- **Does not reorder anything** -- line order, sentence order, and
  paragraph order are always preserved.
- **Does not normalise Unicode** (e.g., NFC/NFKC) -- this is a
  conscious choice.  If Unicode normalisation becomes needed, it should
  be added explicitly as a documented change.
- **Does not strip internal blank lines from prompts** -- these are
  paragraph breaks that may carry structural meaning.
- **Does not touch punctuation** -- commas, periods, dashes, and all
  other punctuation are preserved exactly.
- **Does not trim internal whitespace from prompts** -- only
  leading/trailing whitespace on each line is stripped.

### The Invariant

The normalisation rules are designed to uphold a single invariant:

> Any edit that changes meaning **will** change the hash.
> Any edit that does not change meaning **will not** change the hash.

This ensures that the hashing system is both sensitive to real changes
and robust against formatting noise.

---

## The Provenance Chain in Action

This section walks through a complete generation cycle, showing exactly
when and where each hash is computed, what data flows through the chain,
and what the final response looks like.

### End-to-End Data Flow

```text
 +-------------------------------------------------------------------+
 |  Frontend: User edits axes, selects model, clicks Generate        |
 |                                                                   |
 |  Sends POST /api/generate:                                       |
 |    { payload: {...}, model: "gemma2:2b",                          |
 |      temperature: 0.2, max_tokens: 120 }                         |
 +-------------------------------+-----------------------------------+
                                 |
                                 v
 +-------------------------------------------------------------------+
 |  Backend: generate() route handler                                |
 |                                                                   |
 |  1. Resolve system prompt                                         |
 |     - Use custom override from request, OR                        |
 |     - Load default from app/prompts/system_prompt_v01.txt         |
 |                                                                   |
 |  2. Serialise payload as pretty-printed JSON                      |
 |     - This becomes the "user turn" sent to the LLM               |
 |                                                                   |
 |  3. Call Ollama                                                   |
 |     - Send system prompt + serialised payload                     |
 |     - Receive generated text + usage metrics                      |
 |                                                                   |
 |  4. Compute IPC hashes:                                           |
 |     input_hash        = compute_payload_hash(payload.model_dump())|
 |     system_prompt_hash = compute_system_prompt_hash(prompt)       |
 |     output_hash       = compute_output_hash(text)                 |
 |     ipc_id            = compute_ipc_id(                           |
 |         input_hash, system_prompt_hash,                           |
 |         model, temperature, max_tokens, seed)                     |
 +-------------------------------+-----------------------------------+
                                 |
                                 v
 +-------------------------------------------------------------------+
 |  GenerateResponse (JSON):                                         |
 |  {                                                                |
 |    "text": "A weathered figure stands near the threshold...",     |
 |    "model": "gemma2:2b",                                          |
 |    "temperature": 0.2,                                            |
 |    "input_hash": "d8bd139571345...",                              |
 |    "system_prompt_hash": "7f3a9c6e4b8f1...",                     |
 |    "output_hash": "9cbe31f3d1e7a...",                             |
 |    "ipc_id": "4a2e7f9183bc2..."                                   |
 |  }                                                                |
 +-------------------------------+-----------------------------------+
                                 |
                                 v
 +-------------------------------------------------------------------+
 |  Frontend: Display in UI                                          |
 |                                                                   |
 |  Output text appears in the output box.                           |
 |  Meta area shows three lines:                                     |
 |                                                                   |
 |  model: gemma2:2b  .  temp: 0.2  .  seed: 2954173979             |
 |  input: d8bd139571345...  .  prompt: 7f3a9c6e4b8f1...            |
 |    .  output: 9cbe31f3d1e7a...                                   |
 |  ipc: 4a2e7f9183bc2...                                            |
 +-------------------------------------------------------------------+
```

### Step-by-Step Walkthrough

**Step 1 -- Prompt resolution.**  The backend checks whether the request
includes a custom `system_prompt` override.  If not, it loads the default
prompt from `app/prompts/system_prompt_v01.txt`.  The resolved prompt is
the text that will be hashed as `system_prompt_hash`.

**Step 2 -- Payload serialisation.**  The `AxisPayload` is serialised
to pretty-printed JSON (with 2-space indentation) and sent as the user
turn to Ollama.  This is the text the LLM "sees" as the user message.

**Step 3 -- Ollama generation.**  The backend calls Ollama's
`/api/generate` endpoint with the system prompt and serialised payload.
Ollama returns the generated text and optional usage metrics (prompt
tokens, generation tokens).

**Step 4 -- Hash computation.**  After the Ollama call succeeds, the
backend computes all four hashes.  The order matters: `input_hash` and
`system_prompt_hash` are computed from the request data, `output_hash`
from the response text, and `ipc_id` from the combination of all
provenance fields.  All four are included in the `GenerateResponse`.

**Step 5 -- Frontend display.**  The frontend displays the hashes in a
three-line meta area below the output text.  All hashes are truncated
to 16 characters (out of 64) for UI readability.  The full 64-character
hashes are available in the JSON response and in saved files.

---

## Integration Points

This section catalogues every place in the codebase where IPC hashes are
computed, transmitted, stored, or displayed.

### `/api/generate` -- Full IPC Chain

Every successful generation returns all four hashes.  The hashes are
computed after the Ollama call completes (the `output_hash` needs the
generated text).

The `GenerateResponse` Pydantic model defines the four IPC fields:

- `input_hash` -- SHA-256 of the canonical AxisPayload
- `system_prompt_hash` -- SHA-256 of the normalised system prompt
- `output_hash` -- SHA-256 of the normalised output text
- `ipc_id` -- the composite Interpretive Provenance Chain identifier

All four fields are `str | None` with `default=None` on the schema,
but the generate endpoint always populates them.

*Implementation: `app/main.py`, generate route handler.
Schema: `app/schema.py`, `GenerateResponse` class.*

### `/api/log` -- Backward-Compatible IPC

The log endpoint appends structured entries to `logs/run_log.jsonl`.
It supports IPC hashes with backward compatibility:

- `input_hash` and `output_hash` are **always** computed (these only
  need the payload and output text, which are required parameters).
- `system_prompt_hash` and `ipc_id` are **only** computed when the
  optional `system_prompt` parameter is provided.
- When `system_prompt` is omitted, both fields are `null` in the
  log entry.

This design ensures that older frontend versions (or external callers)
that do not send the prompt can still log successfully.  The IPC fields
on `LogEntry` are `Optional[str]` with `default=None`, so existing
JSONL records written before IPC was implemented can still be
deserialised without error.

*Implementation: `app/main.py`, log_run route handler.
Schema: `app/schema.py`, `LogEntry` class.*

### `/api/save` -- Persistent IPC Record

The save endpoint writes session state to a timestamped folder under
`data/`.  IPC hashes are persisted in two locations:

**1. `metadata.json`** -- All four hashes appear as top-level fields:

```json
{
  "folder_name": "20260218_143022_d8bd1395",
  "timestamp": "2026-02-18T14:30:22.504478+00:00",
  "input_hash": "d8bd1395713454e4...",
  "system_prompt_hash": "7f3a9c6e4b8f1d2c...",
  "output_hash": "9cbe31f3d1e7a2c6...",
  "ipc_id": "4a2e7f9183bc2d4e...",
  "model": "gemma2:2b",
  "temperature": 0.2,
  "max_tokens": 120,
  "seed": 2954173979,
  "world_id": "pipeworks_web",
  "policy_hash": "d845cdcf...",
  "axis_count": 11
}
```

**2. `output.md`** -- The `system_prompt_hash` and `ipc_id` appear as
HTML comments in the provenance header, making the saved file
self-documenting:

```html
<!-- system_prompt_hash: 7f3a9c6e4b8f1d2c... -->
<!-- ipc_id: 4a2e7f9183bc2d4e... -->
```

**Conditional computation:**  The `system_prompt_hash` is always
computed (the system prompt is a required field in `SaveRequest`).
However, `output_hash` and `ipc_id` are only computed when the user
has generated output before saving.  Without output, the provenance
chain is incomplete, and both fields are `null`.

*Implementation: `app/main.py`, save route handler.
Schema: `app/schema.py`, `SaveRequest` and `SaveResponse` classes.*

### Frontend Display

The frontend (`app/static/app.js`) displays IPC hashes in a
three-line meta area below the generated output:

```text
model: gemma2:2b  .  temp: 0.2  .  seed: 2954173979
input: d8bd139571345...  .  prompt: 7f3a9c6e4b8f1...  .  output: 9cbe31f3d1e7a...
ipc: 4a2e7f9183bc2...
```

All hashes are truncated to 16 characters via `.slice(0, 16)` followed
by an ellipsis character.  The `ipc_id` gets its own line to visually
distinguish the composite identifier from the component hashes.

The CSS class `.output-meta` uses `white-space: pre-wrap` so that the
newline characters in the meta string render as line breaks.

### Hash Availability Summary

| Endpoint | `input_hash` | `system_prompt_hash` | `output_hash` | `ipc_id` |
|----------|:---:|:---:|:---:|:---:|
| `/api/generate` | Always | Always | Always | Always |
| `/api/log` | Always | When prompt provided | Always | When prompt provided |
| `/api/save` response | Always | Always | When output exists | When output exists |
| `metadata.json` (saved) | Always | Always | When output exists | When output exists |
| `output.md` (saved) | In header | In header | N/A (is the content) | In header |
| Frontend display | Always | Always | Always | Always |

---

## Use Cases and Experimental Scenarios

The IPC system enables four categories of analysis that were not
possible before its implementation.

### Detecting Prompt Drift

**Scenario:**  You modify the system prompt from v01 to v02 (e.g.,
adding "avoid metaphor").  You generate text from the same payload
with the same model and parameters.

**What the hashes reveal:**

- `input_hash` -- **unchanged** (same payload)
- `system_prompt_hash` -- **different** (prompt text changed)
- `ipc_id` -- **different** (one component changed)
- `output_hash` -- **different** (the LLM responded differently)

**Conclusion:**  The behavioural change is attributable to the prompt
change, not to the model or the input.  The IPC separates the
variables.

Without the IPC, you would see different output and have no way to
determine whether the cause was the prompt, the model, or random
variation.

### Detecting Model Drift

**Scenario:**  You upgrade Ollama's model (e.g., a new release of
`gemma2:2b`).  Same payload, same prompt, same parameters.

**What the hashes reveal:**

- `input_hash` -- **unchanged**
- `system_prompt_hash` -- **unchanged**
- `ipc_id` -- **unchanged** (if the model string is identical)
- `output_hash` -- **may differ** (model internals changed)

**Conclusion:**  If the IPC ID matches but the output hash differs,
the model's internal behaviour changed even though all user-controlled
inputs were identical.  This is Ollama-level or model-weight-level
drift -- invisible without the IPC.

### Reproducibility Audit

**Scenario:**  You need to verify that a saved session can be
reproduced.

**Procedure:**

1. Load `metadata.json` from a saved session.
2. Extract the `ipc_id`.
3. Re-run the generation with the same payload, prompt, model, and
   parameters.
4. Compare the new `ipc_id` to the saved one -- they should match.
5. Compare the `output_hash` -- if they differ, the model exhibited
   stochastic variation even under identical conditions.

This provides a rigorous, quantifiable measure of reproducibility.

### Grouping Runs for Analysis

**Scenario:**  You have 50 log entries in `run_log.jsonl` and want
to analyse output stability.

**Procedure:**

1. Group entries by `ipc_id`.
2. Within each group, examine `output_hash` variation to measure
   output stability under identical conditions.
3. Across groups, compare how different conditions produce different
   outputs.
4. Use `system_prompt_hash` to isolate the effect of prompt changes.
5. Use `input_hash` to isolate the effect of payload changes.

This enables systematic, data-driven analysis of LLM behaviour --
the kind of analysis that is impossible without provenance tracking.

---

## Design Decisions and Trade-offs

This section documents the *why* behind specific technical choices,
for future maintainers and contributors.

### Why SHA-256?

SHA-256 is fast, widely available in Python's standard library
(`hashlib`), and produces a 64-character hex digest that is long enough
to be collision-resistant for this use case.  It is the same algorithm
used elsewhere in the project (e.g., `policy_hash`).  There is no need
for a cryptographic commitment scheme -- the hashes are fingerprints,
not signatures.

### Why Not Hash Raw Bytes?

Raw file contents include editor artifacts: trailing whitespace,
different line endings (LF vs CRLF), BOM characters.  Two developers
editing the same prompt in different editors would produce different
hashes for semantically identical prompts.  Normalisation eliminates
this source of false mismatch.

*Source: `_working/axis_lab/pipeworks_lab_prompt_hash.md`, section 4.1:
"We do not hash raw file contents.  We hash a normalised version to
avoid meaningless diffs."*

### Why Is the Hashing Module Dependency-Free?

`app/hashing.py` imports only `hashlib`, `json`, and `re` from the
standard library.  It does not import Pydantic models from
`app/schema.py`.  This is intentional: it keeps the module usable from
any context (tests, scripts, future tools) without pulling in the web
framework's dependencies.  Callers convert Pydantic models to plain
dicts before calling `compute_payload_hash`.

### Why `sort_keys=True` for Payload Canonicalisation?

Python dicts are insertion-ordered since 3.7, but different code paths
may construct the same payload with keys in different order.  Sorted-key
serialisation guarantees the same JSON string regardless of construction
order.  This is verified by the test
`TestComputePayloadHash.test_order_independent` in
`tests/test_hashing.py`.

### Why Colon Delimiters in the IPC ID?

Without a delimiter, concatenating field values could produce ambiguous
strings.  The colon is a non-hex character that creates unambiguous
field boundaries.  This prevents collisions like
`"ab" + "cd"` vs `"abc" + "d"`, which would produce the same
concatenated string `"abcd"` but different colon-delimited strings
`"ab:cd"` vs `"abc:d"`.

This is verified by the test
`TestComputeIpcId.test_colon_delimiter_prevents_collision` in
`tests/test_hashing.py`.

*Source: Implementation recommendation from
`_working/axis_lab/pipeworks_axis_descriptor_lab_proposed_enhancements.md`:
"concatenate the hex digests with a non-hex delimiter (like a colon)
before hashing the final string."*

### Why Are IPC Fields Optional on LogEntry?

Backward compatibility.  The logging endpoint existed before IPC hashing
was implemented.  Existing JSONL records have no IPC fields; making them
`Optional` with `default=None` means those records can still be
deserialised by the updated `LogEntry` schema without breaking.

### Future Considerations

The original design document proposes an "Optional advanced version" of
the IPC ID that includes the Ollama build hash, model digest, and host
platform for even higher reproducibility guarantees:

```text
IPC_ID_advanced = SHA-256(
    input_hash
    + ":" + system_prompt_hash
    + ":" + model
    + ":" + temperature
    + ":" + max_tokens
    + ":" + seed
    + ":" + ollama_build_hash
    + ":" + model_digest
    + ":" + host_platform
)
```

This depends on Ollama API support for exposing build and model digest
information.  It remains a valuable long-term goal.

*Source: `_working/axis_lab/pipeworks_interpretive_provenance_chains.md`,
section 5.*

---

## Testing the Hashing System

The hashing system has comprehensive test coverage in
`tests/test_hashing.py`.  This section describes the test organisation
and the properties verified.

### Test Organisation

The test module contains six test classes, each targeting a single
function:

| Test Class | Function Under Test | Tests |
|------------|-------------------|:-----:|
| `TestNormaliseSystemPrompt` | `_normalise_system_prompt` | 8 |
| `TestNormaliseOutput` | `_normalise_output` | 7 |
| `TestComputeSystemPromptHash` | `compute_system_prompt_hash` | 5 |
| `TestComputeOutputHash` | `compute_output_hash` | 5 |
| `TestComputeIpcId` | `compute_ipc_id` | 5 |
| `TestComputePayloadHash` | `compute_payload_hash` | 4 |
| **Total** | | **34** |

### Properties Verified

Every test class verifies four properties (as stated in the test module
docstring):

1. **Correctness of normalisation rules** -- edge cases, boundary
   conditions, empty inputs, whitespace-only inputs.
2. **Determinism** -- same input always produces same output.
3. **Sensitivity** -- different inputs produce different outputs.
4. **Format** -- 64-character lowercase hex digest (where applicable).

### Running the Tests

```bash
# Run hashing tests only
pytest tests/test_hashing.py -v

# Run with coverage
pytest tests/test_hashing.py -v --cov=app.hashing --cov-report=term
```

### Extending the Normalisation Rules

If a normalisation rule needs to change (e.g., adding Unicode NFC
normalisation), follow this procedure:

1. **Add tests for the new behaviour first** (test-driven development).
2. **Update the private normalisation function** in `app/hashing.py`.
3. **Verify that all existing tests still pass** -- unless the change
   is intentionally breaking.
4. **Update this guide** to document the new rule.

:::{warning}
Changing normalisation rules **invalidates all previously computed
hashes** for affected hash types.  This is a breaking change for any
stored data (saved sessions, JSONL logs) that references those hashes.
Treat normalisation changes with the same care as a database schema
migration.
:::

---

## Design Document References

The IPC system was designed based on three documents in the project's
`_working/axis_lab/` directory.  These documents are preserved for
historical reference and contain the original reasoning and
specifications.

### Primary Design Document

**`_working/axis_lab/pipeworks_interpretive_provenance_chains.md`**
-- *"Axis Descriptor Lab -- Reproducibility & Trace Methodology"*

This document formalises the IPC concept, defines the required hashes,
specifies the IPC ID formula, and articulates the experimental
guarantees.  It establishes the theoretical framework that the
implementation realises.

### System Prompt Hashing Specification

**`_working/axis_lab/pipeworks_lab_prompt_hash.md`**
-- *"System Prompt Hashing -- Reproducibility & Drift Control"*

This document identifies the gap that system prompt changes were
untracked, defines the normalisation strategy for prompt text, and
proposes a three-phase rollout from observational to scientific
capability.  Its concluding statement captures the motivation:

> "Without system_prompt_hash, the lab is observational.
> With system_prompt_hash, the lab becomes scientific."

### Enhancement Assessment

**`_working/axis_lab/pipeworks_axis_descriptor_lab_proposed_enhancements.md`**
-- *"Assessment of Proposed Enhancements for Axis Descriptor Lab"*

A technical assessment of both design documents, evaluating their
feasibility and alignment with the project's philosophy.  This document
recommended the colon-delimiter approach for the IPC ID and noted the
value of consolidating hashing logic into a single utility module --
both of which were adopted in the implementation.

:::{note}
These documents are in the `_working/` directory and are not part of
the published package.  They represent the design rationale and are
preserved as historical reference within the repository.
:::

---

## Glossary

Authoritative layer
:   The deterministic data that drives generation: axis scores, labels,
    seeds, policy rules.  Never overridden by the LLM.  The system is
    authoritative.

Ornamental layer
:   The LLM-generated descriptive text.  Interprets authoritative data
    but has no authority of its own.  The LLM is ornamental.

Interpretive Provenance Chain (IPC)
:   A composite SHA-256 fingerprint of all variables that influence a
    generation.  The chain links payload, prompt, model, sampling
    parameters, and seed into a single reproducibility signature.

IPC ID
:   The single 64-character hexadecimal digest computed from the
    provenance chain.  Two generations with identical IPC IDs used
    identical inputs in every respect.

Normalisation
:   The process of reducing text to a canonical form before hashing.
    Removes noise (trivial whitespace, editor artifacts) while
    preserving signal (case, punctuation, structure).  Ensures that
    semantically identical texts produce identical hashes.

Payload hash (`input_hash`)
:   SHA-256 of the canonically serialised AxisPayload (sorted-key JSON).
    Fingerprints the complete deterministic entity state.

System prompt hash (`system_prompt_hash`)
:   SHA-256 of the normalised system prompt text.  Fingerprints the
    constraint layer that governs LLM behaviour.

Output hash (`output_hash`)
:   SHA-256 of the normalised LLM output text.  Fingerprints the
    observable interpretive artifact produced by the generation.

Drift
:   A change in LLM output that is not attributable to a change in
    user-controlled inputs.  May be caused by model updates, prompt
    changes, or stochastic sampling.  The IPC enables detecting and
    attributing drift to specific variables.

Provenance
:   The complete record of origin and processing history for a generated
    output.  In this system, provenance includes the payload, prompt,
    model, sampling parameters, and seed -- everything needed to
    reproduce the generation.

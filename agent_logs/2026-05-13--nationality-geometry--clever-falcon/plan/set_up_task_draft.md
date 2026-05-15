---
name: nationality_capitals
models:
  - meta-llama/Llama-3.1-8B
  - meta-llama/Llama-3.2-1B-Instruct
---

# Nationality Capital Retrieval

## Section 1: Task Name

Nationality Capital Retrieval (`nationality_capitals`)

## Section 2: Task Description

The model is given a country and must produce its capital city. Input format:
`"The capital of {country} is"` → expected continuation `" {capital}"`.

This is a factual retrieval task: each country deterministically maps to a single capital.
The behavioral purpose is incidental — the real goal is to probe **how the model represents
nationality / country identity** at intermediate layers. We average per-country activations
across paraphrased templates to obtain a robust country centroid, then study the geometry of
those centroids (clustering by continent, Western vs non-Western axes, colonial-pair
proximity, linguistic groupings, geographic isomorphism).

## Section 3: Behavioral Causal Model

**Input variables (2):**

- `country` — categorical, one of the curated 40-country list (see Section 4).
- `template` — categorical, one of 4 paraphrase templates (see "Templates" below).
  Included so that the per-country centroid is averaged over template variation (analogous
  to how weekdays' `result` centroid is averaged over `(entity, number)` combinations).

**Output variable:**

- `capital` — the capital city string corresponding to `country`. The dependence on
  `template` is trivial (templates leave the answer unchanged).

**raw_input format (one per template):**

1. `"The capital of {country} is"`
2. `"{country}'s capital city is"`
3. `"When you visit {country}, the capital city you'd arrive in is"`
4. `"The capital city of {country} is"`

**Mechanism:** `capital = CAPITAL_OF[country]` (table lookup; `template` is a no-op for output).

## Section 4: Input Samplers

`enumerate_all: true` — enumerate every (country, template) pair. With 40 countries × 4
templates = 160 examples. Stratified by continent so each region is represented:

| Continent | Countries (n=40) |
|---|---|
| Europe (12) | France, Germany, Italy, Spain, Britain, Russia, Greece, Sweden, Poland, Portugal, Austria, Norway |
| Africa (8) | Egypt, Nigeria, Kenya, Morocco, Ethiopia, Senegal, Ghana, Tunisia |
| Asia (10) | China, Japan, India, Vietnam, Thailand, Indonesia, Iran, Iraq, Turkey, Pakistan |
| North America (3) | Canada, Mexico, Cuba |
| South America (5) | Brazil, Argentina, Chile, Peru, Colombia |
| Oceania (2) | Australia, New Zealand |

Capitals (deterministic lookup):

```
France→Paris, Germany→Berlin, Italy→Rome, Spain→Madrid, Britain→London, Russia→Moscow,
Greece→Athens, Sweden→Stockholm, Poland→Warsaw, Portugal→Lisbon, Austria→Vienna, Norway→Oslo,
Egypt→Cairo, Nigeria→Abuja, Kenya→Nairobi, Morocco→Rabat, Ethiopia→"Addis Ababa",
Senegal→Dakar, Ghana→Accra, Tunisia→Tunis,
China→Beijing, Japan→Tokyo, India→"New Delhi", Vietnam→Hanoi, Thailand→Bangkok,
Indonesia→Jakarta, Iran→Tehran, Iraq→Baghdad, Turkey→Ankara, Pakistan→Islamabad,
Canada→Ottawa, Mexico→"Mexico City", Cuba→Havana,
Brazil→Brasília, Argentina→"Buenos Aires", Chile→Santiago, Peru→Lima, Colombia→Bogotá,
Australia→Canberra, "New Zealand"→Wellington
```

The baseline step will report which (country, template) cells the model gets right; cells
where the model fails (e.g. Nigeria→Lagos instead of Abuja) are filtered out downstream by
the standard correct-only filter.

## Section 5: Causal Model Hypotheses with Intermediate Structure

**Base hypothesis:** no intermediate variable. `country → capital` direct.

**Optional extension (not in v1):** add an intermediate `continent` variable to test
whether the model routes via a coarse geography stage before retrieving the capital.
If pursued, the chain becomes `country → continent → capital`. Hold this for a follow-up
session — for v1 the geometry analysis directly probes whether continent structure shows up
in the country representation, without needing it as an explicit variable.

## Section 6: Counterfactuals

**`change_country`** — change only `country`, keep `template` fixed; require `capital` to
change. This is the workhorse counterfactual for `locate` (pairwise mode) and gives the
canonical patching test for "where is country identity encoded?"

**`change_template`** — change only `template`, keep `country` fixed; `capital` stays the
same. Useful as a *negative control*: an intervention site that flips outputs on
`change_country` but not on `change_template` is selective for country, not for template
or generic semantics.

**`random`** — change both `country` and `template` independently. Baseline counterfactual.

Centroid mode (`resample_variable: all`) averages per-`capital` over the (country, template)
slice — since each capital corresponds to one country, this collapses to averaging the 4
templates per country, which is exactly the per-country centroid we want for the geometry
analysis.

## Section 7: Language Model

```yaml
models:
  - meta-llama/Llama-3.1-8B
  - meta-llama/Llama-3.2-1B-Instruct
```

Primary: Llama-3.1-8B. Cheap fallback for sanity-checking on CPU: Llama-3.2-1B-Instruct
(expected to drop several less-famous capitals; baseline will tell us how many survive).

## Section 8: Token Positions

- `country_start` — first token of the country name in the prompt.
- `country_end` — last token of the country name (multi-token countries like "New Zealand"
  span more than one token).
- `is_token` — the "is" token at the end of the prompt (immediately before the answer).
- `last_token` — the final input token before generation. Equal to `is_token` for all four
  templates (each ends with `" is"`).

Each template positions the country span differently; the token-position factories must
locate `country_start` / `country_end` by string match rather than a fixed index, since
country names vary in token length (e.g. "France" is 1 token, "New Zealand" is 2).

## Output Token Mode

```
output_token_mode: first_token_only
```

Many capitals are multi-token in Llama BPE (e.g. "Buenos Aires", "Mexico City", "Addis
Ababa", "New Delhi", "Brasília"). `first_token_only` keeps the full country list usable
without filtering half of it out. The checker compares the first generated token to the
first token of the expected capital. We accept that this introduces some ambiguity (e.g.
"Mexico" starts both the country and the capital "Mexico City"); the baseline step will
expose any first-token collisions and we can drop those countries from the analysis pool.

## Notes for downstream analyses (informational; not consumed by `/setup-task`)

The geometry probe is what this task is built for. After running this task through
`baseline → locate → subspace`, downstream interpretation will look at per-country
centroids for:

- Continent clustering (silhouette score with continent labels)
- A Western/non-Western dominant axis (PC0 projection of G7 vs Global South)
- Colonial pair proximity (UK↔India, France↔Senegal, Spain↔Mexico vs random pairs)
- Linguistic clusters (Francophone, Anglophone, Arabic) cutting across geography
- Approximate latitude/longitude isomorphism of 2D PCA

These belong in `plan/research_objective.md` (written by `/plan-experiment`), not here.

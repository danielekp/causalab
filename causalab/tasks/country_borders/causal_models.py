"""Causal model for the country_borders task.

DAG: (country, direction, template) → raw_input → raw_output
     (country, direction) → neighbor  (table lookup over NEIGHBOR_OF)

`template` is included as an input variable so per-(country, direction)
centroids can average over template paraphrases. The interesting averaging,
however, happens at the *answer-country* level: many distinct (country,
direction) cells share the same primary neighbor, so the centroid for a
given answer-country (e.g. France) averages across many entity tokens.
"""
from __future__ import annotations

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism, input_var

from .config import (
    TASK_NAME,
    COUNTRIES,
    DIRECTIONS,
    DIRECTION_PHRASE,
    NEIGHBOR_OF,
    COUNTRY_FIRST_TOKEN_OF,
    VALID_CELLS,
    primary_neighbor,
)
from .templates import TEMPLATES, fill_template


def _compute_raw_input(t: CausalTrace) -> str:
    country = t["country"]
    direction = t["direction"]
    dir_phrase = DIRECTION_PHRASE[direction]
    return fill_template(t["template"], country, dir_phrase)


def _compute_raw_output(t: CausalTrace) -> str:
    """First BPE token of the primary neighbor for (country, direction).

    Cells outside VALID_CELLS (e.g. a (country, direction) pair with no land
    neighbor) return an empty string — the framework's correct-only filter
    will drop these naturally.
    """
    key = (t["country"], t["direction"])
    if key not in NEIGHBOR_OF:
        return ""
    return COUNTRY_FIRST_TOKEN_OF[primary_neighbor(t["country"], t["direction"])]


values: dict[str, list | None] = {
    "country":   COUNTRIES,
    "direction": DIRECTIONS,
    "template":  TEMPLATES,
    "raw_input": None,
    "raw_output": None,
}

mechanisms = {
    "country":   input_var(COUNTRIES),
    "direction": input_var(DIRECTIONS),
    "template":  input_var(TEMPLATES),
    "raw_input": Mechanism(
        parents=["country", "direction", "template"],
        compute=_compute_raw_input,
    ),
    "raw_output": Mechanism(
        parents=["country", "direction"],
        compute=_compute_raw_output,
    ),
}

# Country-index embedding for downstream PCA/manifold tools that want a
# scalar per-country handle.
_country_to_index = {c: i for i, c in enumerate(COUNTRIES)}
_direction_to_index = {d: i for i, d in enumerate(DIRECTIONS)}
embeddings = {
    "country":   lambda v, _m=_country_to_index:   [float(_m[v])],
    "direction": lambda v, _m=_direction_to_index: [float(_m[v])],
}

causal_model = CausalModel(
    mechanisms,
    values,
    id=TASK_NAME,
    embeddings=embeddings,
)

# Exports consumed by causalab.tasks.loader.load_task:
CAUSAL_MODEL = causal_model
TEMPLATE = TEMPLATES  # list → loader will dispatch to create_token_positions(templates=...)
TARGET_VARIABLE = "country"  # primary intervention variable; overridable via runner cfg
EMBEDDINGS = embeddings

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
    LAT_LON_OF,
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

# Geographic parameterization of the answer-country variable: country ->
# [lat, lon] of its capital. This is the metric causal parameter that makes
# the framework's manifold/path_steering machinery applicable — exactly
# graph_walk's `node_coordinates` pattern (identity embedding of a numeric
# coordinate vector). A length-2 embedding yields params `country_0`,
# `country_1` (spline.builders.extract_parameters_from_dataset), so
# activation_manifold can fit a 2-D manifold (intrinsic_dim=2,
# intrinsic_mode=parameter) and geodesics are literally geographic — the
# substrate for the "does B appear at the midpoint between far A and C?"
# test. The §6.2-§6.6 geometry results do not consume this embedding (they
# use external capital coords + activation PCA), so they are unaffected.
def _embed_country_latlon(c: str) -> list[float]:
    lat, lon = LAT_LON_OF[c]
    return [float(lat), float(lon)]


_direction_to_index = {d: i for i, d in enumerate(DIRECTIONS)}
embeddings = {
    "country":   _embed_country_latlon,
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

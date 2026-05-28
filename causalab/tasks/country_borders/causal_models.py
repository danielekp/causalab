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
    EXTRA_ACCEPT,
    SYNONYMS,
    all_neighbors,
)
from .templates import TEMPLATES, fill_template


def _compute_raw_input(t: CausalTrace) -> str:
    country = t["country"]
    direction = t["direction"]
    dir_phrase = DIRECTION_PHRASE[direction]
    return fill_template(t["template"], country, dir_phrase)


def _compute_raw_output(t: CausalTrace) -> list[str]:
    """First BPE tokens of ALL valid neighbors for (country, direction).

    Any element counts as a correct answer — ``compute_base_accuracy`` and the
    task checker both accept a match against any item in the list. The leading
    entry is the canonical answer country (used for centroid grouping). Cells
    with no in-set neighbor are excluded from the dataset by the model's
    ``input_filter`` (set below), so an empty list is only returned if this is
    called directly on such a cell.
    """
    key = (t["country"], t["direction"])
    if key not in NEIGHBOR_OF:
        return []
    neighbors = all_neighbors(*key)
    accepted = [COUNTRY_FIRST_TOKEN_OF[n] for n in neighbors]
    for n in neighbors:  # surface-form synonyms (e.g. Holland for Netherlands)
        accepted += SYNONYMS.get(n, [])
    accepted += EXTRA_ACCEPT.get(key, [])  # real out-of-set neighbors (scoring only)
    seen: set[str] = set()
    return [x for x in accepted if not (x in seen or seen.add(x))]


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

# Only enumerate / sample (country, direction) pairs that have at least one
# in-set neighbor. Without this, the full 30x8 product would include ~74
# unanswerable cells (coastal/edge directions) whose empty raw_output would be
# scored as wrong, deflating accuracy and polluting PCA/centroid activations.
causal_model.input_filter = lambda t: (t["country"], t["direction"]) in NEIGHBOR_OF

# Exports consumed by causalab.tasks.loader.load_task:
CAUSAL_MODEL = causal_model
TEMPLATE = TEMPLATES  # list → loader will dispatch to create_token_positions(templates=...)
TARGET_VARIABLE = "country"  # primary intervention variable; overridable via runner cfg
EMBEDDINGS = embeddings

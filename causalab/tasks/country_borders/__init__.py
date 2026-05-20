"""country_borders task — predict the country bordering a given country in a
given cardinal/intercardinal direction.

30 European countries × 8 directions = up to 240 (country, direction) cells,
of which only the geographically valid ones (entries in NEIGHBOR_OF) become
prompts. Probes how Llama represents European country geometry along the
border-relation it would have to compute over to answer the prompts.
"""
from .causal_models import causal_model, CAUSAL_MODEL, TEMPLATE, TARGET_VARIABLE
from .counterfactuals import COUNTERFACTUAL_GENERATORS, generate_dataset
from .token_positions import create_token_positions
from .config import (
    TASK_NAME,
    COUNTRIES,
    DIRECTIONS,
    DIRECTION_PHRASE,
    NEIGHBOR_OF,
    VALID_CELLS,
    COUNTRY_FIRST_TOKEN_OF,
    LINGUISTIC_FAMILY_OF,
    EAST_WEST_OF,
    EU_MEMBER_OF,
    LAT_LON_OF,
    MAX_TASK_TOKENS,
    MAX_NEW_TOKENS,
    OUTPUT_TOKEN_MODE,
    OUTPUT_PREFIX,
    primary_neighbor,
    all_neighbors,
)
from .templates import TEMPLATES, fill_template

__all__ = [
    "causal_model",
    "CAUSAL_MODEL",
    "TEMPLATE",
    "TEMPLATES",
    "TARGET_VARIABLE",
    "COUNTERFACTUAL_GENERATORS",
    "generate_dataset",
    "create_token_positions",
    "fill_template",
    "TASK_NAME",
    "COUNTRIES",
    "DIRECTIONS",
    "DIRECTION_PHRASE",
    "NEIGHBOR_OF",
    "VALID_CELLS",
    "COUNTRY_FIRST_TOKEN_OF",
    "LINGUISTIC_FAMILY_OF",
    "EAST_WEST_OF",
    "EU_MEMBER_OF",
    "LAT_LON_OF",
    "MAX_TASK_TOKENS",
    "MAX_NEW_TOKENS",
    "OUTPUT_TOKEN_MODE",
    "OUTPUT_PREFIX",
    "primary_neighbor",
    "all_neighbors",
]

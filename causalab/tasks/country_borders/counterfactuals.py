"""Counterfactual generators for the country_borders task.

Four counterfactual types:

- `change_country`: swap only country (direction, template held fixed). The
  answer changes — workhorse counterfactual for `locate` pairwise mode when
  isolating country identity.
- `change_direction`: swap only direction (country, template held fixed). The
  answer typically changes — isolates direction identity.
- `change_template`: swap only template. The answer stays the same — negative
  control for selectivity tests.
- `random`: swap all three independently. Baseline.

Counterfactuals that produce an invalid (country, direction) cell (one with
no land neighbor in the country set) are retried.
"""
from __future__ import annotations

import random

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace

from .causal_models import causal_model
from .config import COUNTRIES, DIRECTIONS, NEIGHBOR_OF
from .templates import TEMPLATES


def _is_valid_cell(country: str, direction: str) -> bool:
    return (country, direction) in NEIGHBOR_OF


def _sample_valid_input() -> CausalTrace:
    """Sample an input whose (country, direction) cell is in NEIGHBOR_OF."""
    while True:
        base = causal_model.sample_input()
        if _is_valid_cell(base["country"], base["direction"]):
            return base


def _resample_field(
    base: CausalTrace,
    field: str,
    choices: list[str],
    require_valid: bool = True,
    max_tries: int = 200,
) -> CausalTrace:
    """Return a copy of base with `field` replaced by a different value.

    If require_valid, also require the resulting (country, direction) cell to
    be in NEIGHBOR_OF (so the cf has a defined answer).
    """
    if len(choices) <= 1:
        return base.copy()
    for _ in range(max_tries):
        new_val = random.choice(choices)
        if new_val == base[field]:
            continue
        cf = base.copy().intervene(field, new_val)
        if not require_valid or _is_valid_cell(cf["country"], cf["direction"]):
            return cf
    return base.copy()


def change_country() -> CounterfactualExample:
    """Change only country; direction and template fixed; result is a valid cell."""
    base = _sample_valid_input()
    cf = _resample_field(base, "country", COUNTRIES, require_valid=True)
    return CounterfactualExample(input=base, counterfactual_inputs=[cf])


def change_direction() -> CounterfactualExample:
    """Change only direction; country and template fixed; result is a valid cell."""
    base = _sample_valid_input()
    cf = _resample_field(base, "direction", DIRECTIONS, require_valid=True)
    return CounterfactualExample(input=base, counterfactual_inputs=[cf])


def change_template() -> CounterfactualExample:
    """Change only template; country and direction fixed (validity preserved)."""
    base = _sample_valid_input()
    cf = _resample_field(base, "template", TEMPLATES, require_valid=False)
    return CounterfactualExample(input=base, counterfactual_inputs=[cf])


def random_counterfactual() -> CounterfactualExample:
    """Change all three fields independently; both base and cf are valid cells."""
    base = _sample_valid_input()
    cf = _sample_valid_input()
    return CounterfactualExample(input=base, counterfactual_inputs=[cf])


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate n random counterfactual examples — entry point used by some runners."""
    state = random.getstate()
    random.seed(seed)
    examples = []
    for _ in range(n):
        examples.append(random_counterfactual())
    random.setstate(state)
    return examples


COUNTERFACTUAL_GENERATORS = {
    "change_country":   change_country,
    "change_direction": change_direction,
    "change_template":  change_template,
    "random":           random_counterfactual,
}

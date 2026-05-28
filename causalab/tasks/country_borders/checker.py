"""Output checker for the country_borders task.

For cells with multiple valid neighbors (e.g. Germany W → Netherlands/Belgium/
France), the checker accepts ANY of the neighbors as a correct answer by
first-token match. Since ``raw_output`` is now the full neighbor list, the
``compute_base_accuracy`` path already credits any listed neighbor too; this
pipeline-bound checker additionally re-tokenizes per the model's tokenizer and
can recover the cell from the input sample, so it stays robust when the same
first sub-token is shared (e.g. multi-token country names).
"""
from __future__ import annotations

from typing import Any

from causalab.neural.pipeline import LMPipeline

from .config import NEIGHBOR_OF, COUNTRY_FIRST_TOKEN_OF


def make_checker(pipeline: LMPipeline):
    """Build a multi-neighbor first-token checker bound to the pipeline tokenizer.

    The checker accepts the prediction if its first token matches the first
    token of any valid neighbor for the (country, direction) cell. Falls back
    to plain first-token equality if the input sample doesn't carry (country,
    direction) information.
    """
    tokenizer = pipeline.tokenizer

    def _first_token_id(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    def checker(neural_output: dict[str, Any], causal_output: str,
                input_sample: dict | None = None) -> bool:
        actual = neural_output["string"]
        actual_first = _first_token_id(actual)
        if actual_first is None:
            return False

        # If we know the input cell, accept any valid neighbor.
        if input_sample is not None:
            key = (input_sample.get("country"), input_sample.get("direction"))
            if key in NEIGHBOR_OF:
                accepted = {
                    _first_token_id(COUNTRY_FIRST_TOKEN_OF[n])
                    for n in NEIGHBOR_OF[key]
                }
                return actual_first in accepted

        # Fallback: first-token match against any entry of causal_output
        # (raw_output is a list of neighbor first-tokens; a single string is
        # also tolerated).
        expected = causal_output if isinstance(causal_output, list) else [causal_output]
        accepted = {_first_token_id(e) for e in expected}
        accepted.discard(None)
        return actual_first in accepted

    return checker


def checker(neural_output: dict[str, Any], causal_output: str | list[str]) -> bool:
    """Fallback string-level checker for when no tokenizer has been bound.

    Accepts the prediction if its first whitespace-stripped word matches that
    of any acceptable neighbor in ``causal_output`` (the raw_output list).
    """
    actual = neural_output["string"].strip().split()
    if not actual:
        return False
    expected = causal_output if isinstance(causal_output, list) else [causal_output]
    firsts = {e.strip().split()[0] for e in expected if e.strip().split()}
    return actual[0] in firsts

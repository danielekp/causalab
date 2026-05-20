"""Output checker for the country_borders task.

For cells with multiple valid neighbors (e.g. Germany W → France/Belgium/
Netherlands), the checker accepts ANY of the neighbors as a correct answer
by first-token match. The strict-equality `compute_base_accuracy` path will
still only credit predictions matching the canonical primary; this richer
checker is used by analyses that pass a pipeline-bound `make_checker` —
mainly the post-hoc accuracy summary that we add downstream.
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

        # Fallback: strict first-token equality against causal_output.
        expected_first = _first_token_id(causal_output)
        return expected_first is not None and actual_first == expected_first

    return checker


def checker(neural_output: dict[str, Any], causal_output: str) -> bool:
    """Fallback string-level checker for when no tokenizer has been bound.

    Compares the first whitespace-stripped word.
    """
    actual = neural_output["string"].strip().split()
    expected = causal_output.strip().split()
    if not actual or not expected:
        return False
    return actual[0] == expected[0]

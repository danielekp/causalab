"""Metrics for the country_borders task.

The metric signature is fixed: `metric(neural_output, causal_output) -> bool`.
No access to logits / tokenizer / pipeline / input sample. We compare the
first whitespace-delimited word of each output, which approximates first-
token agreement against the *primary* canonical neighbor. For multi-neighbor
acceptance, use `checker.make_checker(pipeline)` which has access to the
input sample.
"""
from __future__ import annotations

from typing import Any


def metric(neural_output: dict[str, Any], causal_output: str) -> bool:
    """First-word match between model output and expected (primary) neighbor."""
    actual = neural_output["string"].strip().split()
    expected = causal_output.strip().split()
    if not actual or not expected:
        return False
    return actual[0] == expected[0]

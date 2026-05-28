"""Metrics for the country_borders task.

The metric signature is fixed: `metric(neural_output, causal_output) -> bool`.
No access to logits / tokenizer / pipeline / input sample. ``causal_output`` is
``raw_output`` — the list of acceptable neighbor first-tokens — so we count the
prediction correct when its first word matches the first word of ANY listed
neighbor. (A legacy single-string ``causal_output`` is also accepted.)
"""
from __future__ import annotations

from typing import Any


def metric(neural_output: dict[str, Any], causal_output: str | list[str]) -> bool:
    """First-word match between model output and any acceptable neighbor."""
    actual = neural_output["string"].strip().split()
    if not actual:
        return False
    expected = causal_output if isinstance(causal_output, list) else [causal_output]
    expected_firsts = {e.strip().split()[0] for e in expected if e.strip().split()}
    return actual[0] in expected_firsts

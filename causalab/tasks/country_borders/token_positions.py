"""Token position factories for the country_borders task.

Positions exposed:

- `last_token` — final input token before generation (the `:` of `"\\nA:"` for
  all four templates). Primary probe site for the geometry analysis.
- `country` — last sub-token of the country span (single-index for pyvene
  compatibility — multi-token countries like "Czech Republic" must have
  uniform-length unit_locations).
- `direction` — last sub-token of the direction span.

Multi-template handling follows the pattern in `natural_domains_arithmetic`:
build per-template TokenPosition objects, then dispatch on `trace["template"]`
at index time.
"""
from __future__ import annotations

from typing import Any

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import (
    TokenPosition,
    build_token_position_factories,
)


TokenPositionSpec = dict[str, Any]


def _build_specs(template: str) -> dict[str, TokenPositionSpec]:
    return {
        "last_token": {"type": "index", "position": -1},
        "country": {
            "type": "index",
            "position": -1,
            "scope": {"variable": "country"},
        },
        "direction": {
            "type": "index",
            "position": -1,
            "scope": {"variable": "direction"},
        },
    }


def create_token_positions(
    pipeline: LMPipeline,
    template: str | None = None,
    templates: list[str] | None = None,
) -> dict[str, TokenPosition]:
    """Build token positions for the task."""
    if templates is not None:
        return _create_multi_template_positions(pipeline, templates)

    if template is None:
        raise ValueError(
            "template is required for country_borders — "
            "use task.create_token_positions(pipeline) instead of calling directly"
        )

    specs = _build_specs(template)
    factories = build_token_position_factories(specs, template)
    return {name: factory(pipeline) for name, factory in factories.items()}


def _create_multi_template_positions(
    pipeline: LMPipeline, templates: list[str]
) -> dict[str, TokenPosition]:
    per_template: dict[str, dict[str, TokenPosition]] = {}
    for tmpl in templates:
        specs = _build_specs(tmpl)
        factories = build_token_position_factories(specs, tmpl)
        per_template[tmpl] = {
            name: factory(pipeline) for name, factory in factories.items()
        }

    all_names: set[str] = set()
    for positions in per_template.values():
        all_names.update(positions.keys())

    result: dict[str, TokenPosition] = {}
    for name in all_names:
        template_positions = {
            tmpl: positions[name]
            for tmpl, positions in per_template.items()
            if name in positions
        }

        def make_indexer(tp_map):
            def indexer(input_sample):
                tmpl = input_sample["template"]
                return tp_map[tmpl].index(input_sample)

            return indexer

        result[name] = TokenPosition(
            make_indexer(template_positions), pipeline, id=name
        )

    return result

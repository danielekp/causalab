"""Template definitions and fill function for the country_borders task.

Four Q&A paraphrased templates, all ending with "\\nA:" so `last_token` aligns
across templates and the model is firmly in "answer mode". The {direction}
slot takes the human-readable form (e.g. "northeast") via DIRECTION_PHRASE
in config.py.

Instruct models need different prompt framing. Two override paths, checked
in order:

1. ``COUNTRY_BORDERS_TEMPLATES_FILE=<path.json>`` — JSON list of pre-rendered
   templates, each containing ``{country}`` and ``{direction}`` slots. Used
   when the canonical format ships in the tokenizer's chat_template (e.g.
   ``processor.apply_chat_template`` for Gemma-4 reasoning models). Render
   in the notebook and write to ``/tmp/...``.
2. ``COUNTRY_BORDERS_TEMPLATES=gemma_chat`` — hand-rolled Gemma-3-style
   ``<start_of_turn>user\\n...\\n<start_of_turn>model\\n`` fallback.

If neither env var is set, the original Q&A templates are used.
"""

import json as _json
import os

_QA_TEMPLATES: list[str] = [
    "Q: Which country lies to the {direction} of {country}?\nA:",
    "Q: What country borders {country} on the {direction}?\nA:",
    "Q: To the {direction} of {country} is the country of\nA:",
    "Q: The country directly {direction} of {country} is\nA:",
]

_GEMMA_CHAT_TEMPLATES: list[str] = [
    "<start_of_turn>user\n"
    "Which country lies to the {direction} of {country}? "
    "Reply with only the country name.<end_of_turn>\n"
    "<start_of_turn>model\n",
    "<start_of_turn>user\n"
    "What country borders {country} on the {direction}? "
    "Reply with only the country name.<end_of_turn>\n"
    "<start_of_turn>model\n",
    "<start_of_turn>user\n"
    "To the {direction} of {country}, what country lies there? "
    "Reply with only the country name.<end_of_turn>\n"
    "<start_of_turn>model\n",
    "<start_of_turn>user\n"
    "The country directly {direction} of {country} is which one? "
    "Reply with only the country name.<end_of_turn>\n"
    "<start_of_turn>model\n",
]

_templates_file = os.environ.get("COUNTRY_BORDERS_TEMPLATES_FILE")
_template_set = os.environ.get("COUNTRY_BORDERS_TEMPLATES", "qa")

if _templates_file and os.path.exists(_templates_file):
    with open(_templates_file) as _f:
        TEMPLATES: list[str] = _json.load(_f)
elif _template_set == "gemma_chat":
    TEMPLATES = _GEMMA_CHAT_TEMPLATES
else:
    TEMPLATES = _QA_TEMPLATES


def fill_template(template: str, country: str, direction: str) -> str:
    """Substitute country and direction into the template.

    Uses str.replace rather than str.format so that chat-template strings
    rendered by ``processor.apply_chat_template`` (which may contain stray
    ``{`` from special tokens like ``<|channel>``) don't trigger KeyError.
    """
    return template.replace("{country}", country).replace("{direction}", direction)

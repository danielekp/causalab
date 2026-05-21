"""Template definitions and fill function for the country_borders task.

Four Q&A paraphrased templates, all ending with "\\nA:" so `last_token` aligns
across templates and the model is firmly in "answer mode". The {direction}
slot takes the human-readable form (e.g. "northeast") via DIRECTION_PHRASE
in config.py.
"""

TEMPLATES: list[str] = [
    "Q: Which country lies to the {direction} of {country}?\nA:",
    "Q: What country borders {country} on the {direction}?\nA:",
    "Q: To the {direction} of {country} is the country of\nA:",
    "Q: The country directly {direction} of {country} is\nA:",
]


def fill_template(template: str, country: str, direction: str) -> str:
    """Substitute country and direction into the template."""
    return template.format(country=country, direction=direction)

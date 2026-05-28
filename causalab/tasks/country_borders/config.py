"""Configuration and constants for the country_borders task.

The model is given a (country, direction) pair and must produce the country
that borders the given country in the specified direction. With 8 directions
(N, NE, E, SE, S, SW, W, NW), many cells have more than one valid answer, so
every cell lists ALL acceptable neighbors (most-central first). A prediction
is scored correct when it matches ANY neighbor in the cell's list: raw_output
is the full list of neighbor first-tokens (compute_base_accuracy credits any
match), and the custom checker honours the same rule. The leading entry is the
canonical "answer country" used for centroid grouping in downstream geometry
analyses. (country, direction) pairs with no in-set neighbor are absent from
the table and are excluded from the dataset via the causal model's
input_filter, so they never count against accuracy.

Design rationale: Border queries map multiple distinct (country, direction)
prompts onto the same answer country, e.g. (Spain, NE), (Italy, NW),
(Germany, W), (Switzerland, W), (Belgium, S) → France. Averaging activations
at the answer position across these prompts yields a centroid for "France-as-
the-target" that is disentangled from the entity tokens — analogous to how
the weekdays-arithmetic task in natural_domains_arithmetic averages over
(entity, increment) pairs whose result is the same day. This is the key
methodological property that capital-retrieval (one entity → one answer) does
not have.
"""
from __future__ import annotations

TASK_NAME = "country_borders"

# 30 European countries. Iceland is dropped (no land neighbors); Britain and
# Ireland are dropped (they only border each other, so each centroid would
# have a single source entity and be entity-contaminated).
COUNTRIES: list[str] = [
    "France", "Italy", "Spain", "Portugal", "Romania",
    "Germany", "Netherlands", "Belgium", "Austria", "Switzerland",
    "Norway", "Sweden", "Denmark",
    "Poland", "Czech Republic", "Slovakia", "Russia", "Ukraine",
    "Belarus", "Bulgaria", "Serbia", "Croatia", "Slovenia",
    "Finland", "Hungary", "Estonia",
    "Latvia", "Lithuania",
    "Greece", "Albania",
]

# 8 compass directions. Intercardinals included to make most border queries
# unambiguous (e.g. Germany E → Poland, Germany SE → Czech Republic).
DIRECTIONS: list[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Human-readable direction phrase for prompt templating.
DIRECTION_PHRASE: dict[str, str] = {
    "N":  "north",
    "NE": "northeast",
    "E":  "east",
    "SE": "southeast",
    "S":  "south",
    "SW": "southwest",
    "W":  "west",
    "NW": "northwest",
}

# Adjacency table: (country, direction) -> list of ALL valid neighboring
# countries for that compass octant (most-central first). EVERY country in a
# list counts as a correct answer for that (country, direction) query — the
# checker and base-accuracy both credit any listed neighbor (see metrics.py /
# checker.py and _compute_raw_output in causal_models.py).
#
# Construction (see git history / PR for full derivation):
# - Edges are the real shared land borders among the 30 in-set countries,
#   PLUS three short sea borders (Denmark-Sweden via Oresund, Denmark-Norway
#   via Skagerrak, Finland-Estonia via the Gulf of Finland), PLUS one
#   micro-state "tunnel": France-Germany across the Luxembourg gap. We do NOT
#   tunnel through full-size countries, so e.g. Serbia does not border Albania
#   (Kosovo/Montenegro lie between), and Kaliningrad is ignored (Poland and
#   Lithuania are not treated as bordering Russia).
# - "All plausible octants": a neighbor is listed under every octant its
#   relation reasonably spans -- the union of (a) the octant(s) of the
#   capital-to-capital bearing (+/-33.75 deg) and (b) the hand-checked
#   border-facing direction. This is why most cells list more than one answer.
NEIGHBOR_OF: dict[tuple[str, str], list[str]] = {
    # France
    ("France", "N"): ["Belgium"],
    ("France", "NE"): ["Belgium", "Germany"],
    ("France", "E"): ["Switzerland", "Germany"],
    ("France", "SE"): ["Italy", "Switzerland"],
    ("France", "S"): ["Spain"],
    ("France", "SW"): ["Spain"],

    # Italy
    ("Italy", "N"): ["Slovenia", "Austria", "Switzerland"],
    ("Italy", "NE"): ["Austria", "Slovenia"],
    ("Italy", "NW"): ["France", "Switzerland"],

    # Spain
    ("Spain", "N"): ["France"],
    ("Spain", "NE"): ["France"],
    ("Spain", "SW"): ["Portugal"],
    ("Spain", "W"): ["Portugal"],

    # Portugal
    ("Portugal", "N"): ["Spain"],
    ("Portugal", "NE"): ["Spain"],
    ("Portugal", "E"): ["Spain"],

    # Romania
    ("Romania", "N"): ["Ukraine"],
    ("Romania", "NE"): ["Ukraine"],
    ("Romania", "S"): ["Bulgaria"],
    ("Romania", "SW"): ["Bulgaria", "Serbia"],
    ("Romania", "W"): ["Serbia", "Hungary"],
    ("Romania", "NW"): ["Hungary"],

    # Germany
    ("Germany", "N"): ["Denmark"],
    ("Germany", "E"): ["Poland"],
    ("Germany", "SE"): ["Austria", "Czech Republic"],
    ("Germany", "S"): ["Czech Republic", "Austria"],
    ("Germany", "SW"): ["Switzerland", "France", "Belgium"],
    ("Germany", "W"): ["Netherlands", "Belgium", "France"],
    ("Germany", "NW"): ["Netherlands"],

    # Netherlands
    ("Netherlands", "E"): ["Germany"],
    ("Netherlands", "S"): ["Belgium"],
    ("Netherlands", "SW"): ["Belgium"],

    # Belgium
    ("Belgium", "N"): ["Netherlands"],
    ("Belgium", "NE"): ["Germany", "Netherlands"],
    ("Belgium", "E"): ["Germany"],
    ("Belgium", "S"): ["France"],
    ("Belgium", "SW"): ["France"],

    # Austria
    ("Austria", "N"): ["Germany", "Czech Republic"],
    ("Austria", "NE"): ["Slovakia"],
    ("Austria", "E"): ["Slovakia", "Hungary"],
    ("Austria", "SE"): ["Hungary"],
    ("Austria", "S"): ["Italy", "Slovenia"],
    ("Austria", "SW"): ["Slovenia", "Italy", "Switzerland"],
    ("Austria", "W"): ["Switzerland"],
    ("Austria", "NW"): ["Czech Republic", "Germany"],

    # Switzerland
    ("Switzerland", "N"): ["Germany"],
    ("Switzerland", "NE"): ["Germany", "Austria"],
    ("Switzerland", "E"): ["Austria"],
    ("Switzerland", "SE"): ["Italy"],
    ("Switzerland", "S"): ["Italy"],
    ("Switzerland", "W"): ["France"],
    ("Switzerland", "NW"): ["France"],

    # Norway
    ("Norway", "NE"): ["Finland", "Sweden"],
    ("Norway", "E"): ["Russia", "Sweden", "Finland"],
    ("Norway", "SE"): ["Denmark"],
    ("Norway", "S"): ["Denmark"],

    # Sweden
    ("Sweden", "NE"): ["Finland"],
    ("Sweden", "E"): ["Finland"],
    ("Sweden", "S"): ["Denmark"],
    ("Sweden", "SW"): ["Denmark"],
    ("Sweden", "W"): ["Norway"],
    ("Sweden", "NW"): ["Norway"],

    # Denmark
    ("Denmark", "N"): ["Norway"],
    ("Denmark", "NE"): ["Sweden"],
    ("Denmark", "E"): ["Sweden"],
    ("Denmark", "S"): ["Germany"],
    ("Denmark", "NW"): ["Norway"],

    # Poland
    ("Poland", "NE"): ["Lithuania", "Belarus"],
    ("Poland", "E"): ["Ukraine", "Belarus"],
    ("Poland", "SE"): ["Ukraine"],
    ("Poland", "S"): ["Slovakia", "Czech Republic"],
    ("Poland", "SW"): ["Slovakia", "Czech Republic"],
    ("Poland", "W"): ["Germany", "Czech Republic"],

    # Czech Republic
    ("Czech Republic", "N"): ["Germany", "Poland"],
    ("Czech Republic", "NE"): ["Poland"],
    ("Czech Republic", "E"): ["Poland", "Slovakia"],
    ("Czech Republic", "SE"): ["Slovakia", "Austria"],
    ("Czech Republic", "S"): ["Austria"],
    ("Czech Republic", "W"): ["Germany"],
    ("Czech Republic", "NW"): ["Germany"],

    # Slovakia
    ("Slovakia", "N"): ["Poland"],
    ("Slovakia", "NE"): ["Poland", "Ukraine"],
    ("Slovakia", "E"): ["Ukraine", "Hungary"],
    ("Slovakia", "SE"): ["Hungary"],
    ("Slovakia", "S"): ["Hungary"],
    ("Slovakia", "SW"): ["Austria"],
    ("Slovakia", "W"): ["Austria", "Czech Republic"],
    ("Slovakia", "NW"): ["Czech Republic"],

    # Russia
    ("Russia", "SW"): ["Ukraine", "Belarus"],
    ("Russia", "W"): ["Belarus", "Latvia", "Norway", "Estonia"],
    ("Russia", "NW"): ["Finland", "Estonia", "Norway", "Latvia"],

    # Ukraine
    ("Ukraine", "N"): ["Belarus"],
    ("Ukraine", "NE"): ["Russia"],
    ("Ukraine", "E"): ["Russia"],
    ("Ukraine", "S"): ["Romania"],
    ("Ukraine", "SW"): ["Romania", "Hungary"],
    ("Ukraine", "W"): ["Slovakia", "Hungary", "Poland"],
    ("Ukraine", "NW"): ["Belarus", "Poland"],

    # Belarus
    ("Belarus", "N"): ["Latvia", "Lithuania"],
    ("Belarus", "NE"): ["Russia"],
    ("Belarus", "E"): ["Russia"],
    ("Belarus", "SE"): ["Ukraine"],
    ("Belarus", "S"): ["Ukraine"],
    ("Belarus", "SW"): ["Poland"],
    ("Belarus", "W"): ["Poland", "Lithuania"],
    ("Belarus", "NW"): ["Lithuania", "Latvia"],

    # Bulgaria
    ("Bulgaria", "N"): ["Romania"],
    ("Bulgaria", "NE"): ["Romania"],
    ("Bulgaria", "S"): ["Greece"],
    ("Bulgaria", "SW"): ["Greece"],
    ("Bulgaria", "W"): ["Serbia"],
    ("Bulgaria", "NW"): ["Serbia"],

    # Serbia
    ("Serbia", "N"): ["Hungary"],
    ("Serbia", "NE"): ["Romania"],
    ("Serbia", "E"): ["Romania", "Bulgaria"],
    ("Serbia", "SE"): ["Bulgaria"],
    ("Serbia", "W"): ["Croatia"],
    ("Serbia", "NW"): ["Croatia", "Hungary"],

    # Croatia
    ("Croatia", "N"): ["Hungary", "Slovenia"],
    ("Croatia", "NE"): ["Hungary"],
    ("Croatia", "E"): ["Serbia"],
    ("Croatia", "SE"): ["Serbia"],
    ("Croatia", "W"): ["Slovenia"],
    ("Croatia", "NW"): ["Slovenia"],

    # Slovenia
    ("Slovenia", "N"): ["Austria"],
    ("Slovenia", "NE"): ["Austria", "Hungary"],
    ("Slovenia", "E"): ["Croatia", "Hungary"],
    ("Slovenia", "SE"): ["Croatia"],
    ("Slovenia", "S"): ["Italy", "Croatia"],
    ("Slovenia", "SW"): ["Italy"],
    ("Slovenia", "W"): ["Italy"],

    # Finland
    ("Finland", "N"): ["Norway"],
    ("Finland", "E"): ["Russia"],
    ("Finland", "SE"): ["Russia"],
    ("Finland", "S"): ["Estonia"],
    ("Finland", "W"): ["Norway", "Sweden"],
    ("Finland", "NW"): ["Norway"],

    # Hungary
    ("Hungary", "N"): ["Slovakia"],
    ("Hungary", "NE"): ["Ukraine"],
    ("Hungary", "E"): ["Ukraine", "Romania"],
    ("Hungary", "SE"): ["Romania", "Serbia"],
    ("Hungary", "S"): ["Serbia", "Croatia"],
    ("Hungary", "SW"): ["Croatia", "Slovenia"],
    ("Hungary", "W"): ["Austria", "Slovenia", "Slovakia"],
    ("Hungary", "NW"): ["Slovakia", "Austria"],

    # Estonia
    ("Estonia", "N"): ["Finland"],
    ("Estonia", "E"): ["Russia"],
    ("Estonia", "SE"): ["Russia"],
    ("Estonia", "S"): ["Latvia"],
    ("Estonia", "NW"): ["Finland"],

    # Latvia
    ("Latvia", "N"): ["Estonia"],
    ("Latvia", "E"): ["Russia"],
    ("Latvia", "SE"): ["Belarus", "Lithuania"],
    ("Latvia", "S"): ["Lithuania"],

    # Lithuania
    ("Lithuania", "N"): ["Latvia"],
    ("Lithuania", "E"): ["Belarus"],
    ("Lithuania", "SE"): ["Belarus"],
    ("Lithuania", "S"): ["Poland"],
    ("Lithuania", "SW"): ["Poland"],
    ("Lithuania", "NW"): ["Latvia"],

    # Greece
    ("Greece", "N"): ["Bulgaria"],
    ("Greece", "NE"): ["Bulgaria"],
    ("Greece", "W"): ["Albania"],
    ("Greece", "NW"): ["Albania"],

    # Albania
    ("Albania", "SE"): ["Greece"],
    ("Albania", "S"): ["Greece"],
}


def primary_neighbor(country: str, direction: str) -> str:
    """Return the canonical primary neighbor for a (country, direction) cell."""
    return NEIGHBOR_OF[(country, direction)][0]


def all_neighbors(country: str, direction: str) -> list[str]:
    """Return the full list of valid neighbors for a (country, direction) cell."""
    return NEIGHBOR_OF[(country, direction)]


# Valid (country, direction) cells — these are the input combinations we
# enumerate. Sorted for determinism.
VALID_CELLS: list[tuple[str, str]] = sorted(NEIGHBOR_OF.keys())

# First BPE token of " {country}" under the Llama-3.1-8B tokenizer, verified
# offline (no collisions across the 30 first-tokens; 29 single-token, only
# Czech Republic is multi-token with first sub-token " Czech"). Used as
# raw_output so the framework's strict-equality check honors first_token_only
# semantics (MAX_NEW_TOKENS=1).
COUNTRY_FIRST_TOKEN_OF: dict[str, str] = {
    "France":         " France",
    "Italy":          " Italy",
    "Spain":          " Spain",
    "Portugal":       " Portugal",
    "Romania":        " Romania",
    "Germany":        " Germany",
    "Netherlands":    " Netherlands",
    "Belgium":        " Belgium",
    "Austria":        " Austria",
    "Switzerland":    " Switzerland",
    "Norway":         " Norway",
    "Sweden":         " Sweden",
    "Denmark":        " Denmark",
    "Poland":         " Poland",
    "Czech Republic": " Czech",
    "Slovakia":       " Slovakia",
    "Russia":         " Russia",
    "Ukraine":        " Ukraine",
    "Belarus":        " Belarus",
    "Bulgaria":       " Bulgaria",
    "Serbia":         " Serbia",
    "Croatia":        " Croatia",
    "Slovenia":       " Slovenia",
    "Finland":        " Finland",
    "Hungary":        " Hungary",
    "Estonia":        " Estonia",
    "Latvia":         " Latvia",
    "Lithuania":      " Lithuania",
    "Greece":         " Greece",
    "Albania":        " Albania",
}

# --- Labeling axes (carry over from v2 of nationality_capitals, restricted
# to the 30 countries here) for downstream centroid-coloring analysis ---

LINGUISTIC_FAMILY_OF: dict[str, str] = {
    "France": "Romance", "Italy": "Romance", "Spain": "Romance",
    "Portugal": "Romance", "Romania": "Romance",
    "Germany": "Germanic", "Netherlands": "Germanic", "Belgium": "Germanic",
    "Austria": "Germanic", "Switzerland": "Germanic",
    "Norway": "Germanic", "Sweden": "Germanic", "Denmark": "Germanic",
    "Poland": "Slavic", "Czech Republic": "Slavic", "Slovakia": "Slavic",
    "Russia": "Slavic", "Ukraine": "Slavic", "Belarus": "Slavic",
    "Bulgaria": "Slavic", "Serbia": "Slavic", "Croatia": "Slavic",
    "Slovenia": "Slavic",
    "Finland": "Uralic", "Hungary": "Uralic", "Estonia": "Uralic",
    "Latvia": "Baltic", "Lithuania": "Baltic",
    "Greece": "Hellenic",
    "Albania": "Albanian",
}

EAST_WEST_OF: dict[str, str] = {
    "France": "West", "Italy": "West", "Spain": "West", "Portugal": "West",
    "Germany": "West", "Netherlands": "West", "Belgium": "West",
    "Austria": "West", "Switzerland": "West",
    "Norway": "West", "Sweden": "West", "Denmark": "West",
    "Finland": "West", "Greece": "West",
    "Romania": "East", "Poland": "East", "Czech Republic": "East",
    "Slovakia": "East", "Russia": "East", "Ukraine": "East", "Belarus": "East",
    "Bulgaria": "East", "Serbia": "East", "Croatia": "East",
    "Slovenia": "East", "Hungary": "East", "Estonia": "East", "Latvia": "East",
    "Lithuania": "East", "Albania": "East",
}

EU_MEMBER_OF: dict[str, bool] = {
    "France": True, "Italy": True, "Spain": True, "Portugal": True,
    "Romania": True, "Germany": True, "Netherlands": True, "Belgium": True,
    "Austria": True, "Sweden": True, "Denmark": True, "Poland": True,
    "Czech Republic": True, "Slovakia": True, "Bulgaria": True, "Croatia": True,
    "Slovenia": True, "Finland": True, "Hungary": True, "Estonia": True,
    "Latvia": True, "Lithuania": True, "Greece": True,
    "Switzerland": False, "Norway": False,
    "Russia": False, "Ukraine": False, "Belarus": False, "Serbia": False,
    "Albania": False,
}

# Capital city latitude/longitude (used for geographic isomorphism analysis).
LAT_LON_OF: dict[str, tuple[float, float]] = {
    "France": (48.86, 2.35),
    "Italy": (41.90, 12.50),
    "Spain": (40.42, -3.70),
    "Portugal": (38.72, -9.14),
    "Romania": (44.43, 26.10),
    "Germany": (52.52, 13.41),
    "Netherlands": (52.37, 4.90),
    "Belgium": (50.85, 4.35),
    "Austria": (48.21, 16.37),
    "Switzerland": (46.95, 7.45),
    "Norway": (59.91, 10.75),
    "Sweden": (59.33, 18.07),
    "Denmark": (55.68, 12.57),
    "Poland": (52.23, 21.01),
    "Czech Republic": (50.08, 14.44),
    "Slovakia": (48.15, 17.11),
    "Russia": (55.75, 37.62),
    "Ukraine": (50.45, 30.52),
    "Belarus": (53.90, 27.57),
    "Bulgaria": (42.70, 23.32),
    "Serbia": (44.79, 20.46),
    "Croatia": (45.81, 15.98),
    "Slovenia": (46.06, 14.51),
    "Finland": (60.17, 24.94),
    "Hungary": (47.50, 19.05),
    "Estonia": (59.44, 24.75),
    "Latvia": (56.95, 24.11),
    "Lithuania": (54.69, 25.28),
    "Greece": (37.98, 23.73),
    "Albania": (41.33, 19.82),
}

# Token budget: longest template ("Q: What country borders {country} on the
# {direction}?\nA:") with the longest country ("Czech Republic") and longest
# direction phrase ("northeast" / "southeast") fits well under 32 tokens.
# MAX_NEW_TOKENS=1 because output_token_mode=first_token_only.
MAX_TASK_TOKENS = 32
MAX_NEW_TOKENS = 1

OUTPUT_TOKEN_MODE = "first_token_only"

OUTPUT_PREFIX = " "

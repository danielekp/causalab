"""Configuration and constants for the country_borders task.

The model is given a (country, direction) pair and must produce the country
that borders the given country in the specified direction. With 8 directions
(N, NE, E, SE, S, SW, W, NW), most cells have a single canonical answer; cells
where multiple countries border in the same direction are handled by listing
all valid neighbors (primary first) — the primary is used as raw_output for
strict-equality scoring and centroid construction, the full list is used by
the custom checker to accept any valid neighbor as correct.

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

# Adjacency table: (country, direction) -> ordered list of valid neighboring
# countries (primary first). Cells with no land neighbor in that direction
# (ocean, or all bordering countries lie outside the 30-country set) are
# absent.
#
# Conventions:
# - Short sea borders (Denmark-Sweden via Øresund, Finland-Estonia via Gulf
#   of Finland, Denmark-Norway via Skagerrak) are included as borders.
# - Borders via small non-set countries (e.g. Luxembourg, Andorra, Bosnia,
#   Kosovo, Moldova, Monaco, San Marino, North Macedonia, Montenegro) are
#   "tunneled through" — i.e. France↔Germany are considered adjacent even
#   though Luxembourg sits between them at the NE corner, because Luxembourg
#   isn't in our country set.
# - Primary = the most prominent / longest shared border in that direction.
NEIGHBOR_OF: dict[tuple[str, str], list[str]] = {
    # France
    ("France", "N"):  ["Belgium"],
    ("France", "NE"): ["Germany"],
    ("France", "E"):  ["Switzerland"],
    ("France", "SE"): ["Italy"],
    ("France", "SW"): ["Spain"],

    # Italy
    ("Italy", "NW"): ["France"],
    ("Italy", "N"):  ["Switzerland", "Austria"],
    ("Italy", "NE"): ["Slovenia", "Austria"],

    # Spain
    ("Spain", "NE"): ["France"],
    ("Spain", "W"):  ["Portugal"],
    ("Spain", "SW"): ["Portugal"],

    # Portugal — only Spain neighbors it; expressed in 3 cells for centroid
    # density. All three prompts contribute to Spain's centroid only via
    # different directions; Portugal centroid will be entity-thin (only Spain
    # as input entity) — flag in downstream analysis.
    ("Portugal", "E"):  ["Spain"],
    ("Portugal", "NE"): ["Spain"],
    ("Portugal", "N"):  ["Spain"],

    # Romania
    ("Romania", "N"):  ["Ukraine"],
    ("Romania", "S"):  ["Bulgaria"],
    ("Romania", "SW"): ["Serbia"],
    ("Romania", "W"):  ["Hungary"],

    # Germany
    ("Germany", "N"):  ["Denmark"],
    ("Germany", "E"):  ["Poland"],
    ("Germany", "SE"): ["Czech Republic"],
    ("Germany", "S"):  ["Austria"],
    ("Germany", "SW"): ["Switzerland", "France"],
    ("Germany", "W"):  ["France", "Belgium", "Netherlands"],
    ("Germany", "NW"): ["Netherlands"],

    # Netherlands
    ("Netherlands", "E"): ["Germany"],
    ("Netherlands", "S"): ["Belgium"],

    # Belgium
    ("Belgium", "N"):  ["Netherlands"],
    ("Belgium", "E"):  ["Germany"],
    # (Belgium, SE) → Germany REMOVED: tunnels through Luxembourg; model
    # consistently says France instead.
    ("Belgium", "S"):  ["France"],
    ("Belgium", "SW"): ["France"],

    # Austria
    ("Austria", "N"):  ["Czech Republic"],
    ("Austria", "NW"): ["Germany"],
    ("Austria", "NE"): ["Slovakia"],   # Czech Republic is NW not NE
    ("Austria", "E"):  ["Hungary", "Slovakia"],
    ("Austria", "SE"): ["Hungary"],    # Slovenia is SW not SE (already covered by S primary)
    ("Austria", "S"):  ["Slovenia", "Italy"],
    ("Austria", "SW"): ["Italy", "Switzerland"],
    ("Austria", "W"):  ["Switzerland"],

    # Switzerland
    ("Switzerland", "N"):  ["Germany"],
    ("Switzerland", "E"):  ["Austria"],
    ("Switzerland", "S"):  ["Italy"],
    ("Switzerland", "SE"): ["Italy"],
    ("Switzerland", "W"):  ["France"],

    # Norway
    ("Norway", "E"):  ["Sweden"],
    ("Norway", "NE"): ["Finland", "Sweden"],
    # (Norway, S) → Denmark REMOVED: across Skagerrak sea border; model
    # strongly prefers Sweden (which is geographically SE).

    # Sweden
    ("Sweden", "W"):  ["Norway"],
    ("Sweden", "NW"): ["Norway"],
    ("Sweden", "E"):  ["Finland"],
    ("Sweden", "NE"): ["Finland"],
    ("Sweden", "S"):  ["Denmark"],   # across Øresund
    ("Sweden", "SW"): ["Denmark"],

    # Denmark
    ("Denmark", "S"):  ["Germany"],
    ("Denmark", "E"):  ["Sweden"],   # across Øresund
    ("Denmark", "NE"): ["Sweden"],
    ("Denmark", "N"):  ["Norway"],   # across Skagerrak

    # Poland
    ("Poland", "W"):  ["Germany"],
    ("Poland", "S"):  ["Slovakia", "Czech Republic"],
    ("Poland", "SW"): ["Czech Republic"],
    ("Poland", "SE"): ["Ukraine"],     # Slovakia is S not SE (already in S primary)
    ("Poland", "E"):  ["Belarus"],
    ("Poland", "NE"): ["Lithuania"],

    # Czech Republic
    ("Czech Republic", "NW"): ["Germany"],
    ("Czech Republic", "W"):  ["Germany"],
    ("Czech Republic", "N"):  ["Poland"],
    ("Czech Republic", "NE"): ["Poland"],
    ("Czech Republic", "E"):  ["Slovakia"],
    ("Czech Republic", "S"):  ["Austria"],

    # Slovakia
    ("Slovakia", "W"):  ["Czech Republic"],
    ("Slovakia", "NW"): ["Czech Republic"],
    ("Slovakia", "N"):  ["Poland"],
    ("Slovakia", "E"):  ["Ukraine"],
    ("Slovakia", "S"):  ["Hungary"],
    ("Slovakia", "SW"): ["Austria"],

    # Russia (European-facing borders only)
    ("Russia", "NW"): ["Finland", "Estonia"],
    ("Russia", "W"):  ["Belarus", "Estonia", "Latvia"],
    ("Russia", "SW"): ["Ukraine", "Belarus"],

    # Ukraine
    ("Ukraine", "N"):  ["Belarus"],
    ("Ukraine", "NE"): ["Russia"],
    ("Ukraine", "E"):  ["Russia"],
    ("Ukraine", "W"):  ["Hungary", "Slovakia", "Poland"],
    ("Ukraine", "NW"): ["Poland"],
    ("Ukraine", "SW"): ["Romania"],

    # Belarus
    ("Belarus", "E"):  ["Russia"],
    ("Belarus", "NE"): ["Russia"],
    ("Belarus", "N"):  ["Latvia", "Lithuania"],
    ("Belarus", "NW"): ["Lithuania"],
    ("Belarus", "W"):  ["Poland"],
    ("Belarus", "SW"): ["Poland"],
    ("Belarus", "S"):  ["Ukraine"],
    ("Belarus", "SE"): ["Ukraine"],

    # Bulgaria
    ("Bulgaria", "N"):  ["Romania"],
    ("Bulgaria", "NE"): ["Romania"],
    ("Bulgaria", "W"):  ["Serbia"],
    ("Bulgaria", "SW"): ["Greece"],
    ("Bulgaria", "S"):  ["Greece"],

    # Serbia
    ("Serbia", "N"):  ["Hungary"],
    ("Serbia", "NE"): ["Romania"],
    ("Serbia", "E"):  ["Bulgaria"],
    ("Serbia", "SE"): ["Bulgaria"],
    # (Serbia, S) → Albania REMOVED: tunnels through Kosovo; model says
    # Macedonia (geographically directly south, not in set).
    # (Serbia, SW) → Albania REMOVED: same reason (Kosovo/Montenegro).
    ("Serbia", "W"):  ["Croatia"],
    ("Serbia", "NW"): ["Croatia"],

    # Croatia
    ("Croatia", "N"):  ["Slovenia", "Hungary"],
    ("Croatia", "NW"): ["Slovenia"],
    ("Croatia", "NE"): ["Hungary"],
    ("Croatia", "E"):  ["Serbia"],
    ("Croatia", "SE"): ["Serbia"],

    # Slovenia
    ("Slovenia", "W"):  ["Italy"],
    ("Slovenia", "SW"): ["Italy"],
    ("Slovenia", "N"):  ["Austria"],
    ("Slovenia", "E"):  ["Hungary"],
    ("Slovenia", "NE"): ["Hungary"],
    ("Slovenia", "S"):  ["Croatia"],
    ("Slovenia", "SE"): ["Croatia"],

    # Finland
    ("Finland", "W"):  ["Sweden"],
    ("Finland", "NW"): ["Norway"],     # Sweden is W not NW (already in W primary)
    ("Finland", "N"):  ["Norway"],
    ("Finland", "E"):  ["Russia"],
    ("Finland", "SE"): ["Russia"],
    ("Finland", "S"):  ["Estonia"],   # across Gulf of Finland

    # Hungary
    ("Hungary", "N"):  ["Slovakia"],
    ("Hungary", "NE"): ["Ukraine"],
    ("Hungary", "E"):  ["Romania"],
    ("Hungary", "SE"): ["Romania"],
    ("Hungary", "S"):  ["Serbia", "Croatia"],
    ("Hungary", "SW"): ["Croatia", "Slovenia"],
    ("Hungary", "W"):  ["Austria", "Slovenia"],
    ("Hungary", "NW"): ["Austria"],

    # Estonia
    ("Estonia", "S"):  ["Latvia"],
    ("Estonia", "E"):  ["Russia"],
    ("Estonia", "N"):  ["Finland"],   # across Gulf
    ("Estonia", "NW"): ["Finland"],

    # Latvia
    ("Latvia", "N"):  ["Estonia"],
    ("Latvia", "S"):  ["Lithuania"],
    ("Latvia", "E"):  ["Russia"],
    ("Latvia", "SE"): ["Belarus"],

    # Lithuania
    ("Lithuania", "N"):  ["Latvia"],
    ("Lithuania", "E"):  ["Belarus"],
    ("Lithuania", "SE"): ["Belarus"],
    ("Lithuania", "S"):  ["Poland"],
    # (Lithuania, SW) → Russia REMOVED: Kaliningrad exclave; model
    # consistently says Latvia/Poland/Belarus — it doesn't know Kaliningrad
    # is Russian territory.
    # (Lithuania, W) → Russia REMOVED: same reason.

    # Greece
    ("Greece", "N"):  ["Bulgaria"],
    ("Greece", "NE"): ["Bulgaria"],
    ("Greece", "NW"): ["Albania"],
    ("Greece", "W"):  ["Albania"],

    # Albania
    ("Albania", "S"):  ["Greece"],
    ("Albania", "SE"): ["Greece"],
    # (Albania, NE) → Serbia REMOVED: tunnels through Kosovo; model says
    # Macedonia or Kosovo (neither in set).
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

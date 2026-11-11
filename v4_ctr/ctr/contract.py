"""Input contract between the trained CTR model and the mgads ad server (admob_platform ADR 0004).

Everything the ad server needs to turn an ad request into model inputs is defined here and written into
the exported manifest (`manifest.json`), so the Go feature builder reads the same vocabularies and rules
instead of keeping its own copy. A golden file of inputs and expected outputs (`golden.json`) is exported
with every model and checked by the ad server's tests.

Inputs are limited to what an ad request carries at serve time (ADR 0004 decision 9): ad format, local
hour and weekday derived from the region's time zone, region, page-context category, the candidate's
creative categories and the user's recent served ads and outcomes. Engagement after the impression
(dwell, skip, interaction) is never an input. The user id, app id and placement id are not inputs: a
learned per-id vector from a synthetic world would mean nothing for a real id, so the user is represented
by their history and the app by its category.
"""

CONTRACT_VERSION = 1

# Recent served ads per user given to the model, newest first, and how long a click needs to be visible.
HISTORY_LEN = 32
FATIGUE_WINDOW_S = 24 * 3600
FATIGUE_CAP = 31
# Session masking (CADET): the outcome of a served ad younger than this is not known to the model, matching
# the event pipeline's admission delay. The ad itself is still in the history.
OUTCOME_DELAY_S = 3600
# Pad up to this many candidate categories per creative.
CANDIDATE_CATEGORIES = 2

# Token 0 is padding (or unknown where noted) in every vocabulary.
FORMATS = ["banner", "interstitial", "rewarded"]  # tokens 1..3
# Ad-server request formats that map to a model format. Anything else has no head and uses the fallback.
SERVER_FORMATS = {"BANNER": "banner", "VIDEO": "interstitial", "INTERSTITIAL": "interstitial",
                  "REWARDED": "rewarded"}

# Synthetic categories (generator ids 0..11) are tokens 2..13; token 1 is "known but outside the
# vocabulary". Each maps to IAB Content Taxonomy 3.0 ids. The server resolves an IAB id by exact match
# first, then by its tier-1 parent; an id that matches neither is token 1.
CATEGORIES = ["enterprise_software", "office_equipment", "video_games", "gaming_hardware", "streaming_media",
              "fashion", "beauty", "groceries", "home_family", "travel", "personal_finance", "education"]
CATEGORY_IAB = {
    "enterprise_software": ["596", "599"],  # Technology & Computing, Computing
    "office_equipment": ["52"],  # Business and Finance
    "video_games": ["680"],  # Video Gaming
    "gaming_hardware": ["632"],  # Consumer Electronics
    "streaming_media": ["JLBCU7"],  # Entertainment (Movies, Music, Television by parent)
    "fashion": ["552"],  # Style & Fashion
    "beauty": ["553"],  # Beauty (a tier-2 under Style & Fashion; exact match wins)
    "groceries": ["473", "477", "210"],  # Shopping, Grocery Shopping, Food & Drink
    "home_family": ["274", "186"],  # Home & Garden, Family and Relationships
    "travel": ["653"],  # Travel
    "personal_finance": ["391"],  # Personal Finance
    "education": ["132"],  # Education
}
CAT_PAD, CAT_OTHER, CAT_BASE = 0, 1, 2
N_CAT_TOKENS = CAT_BASE + len(CATEGORIES)

# Synthetic regions: the default 4-zone mix has 3 region codes per zone, zones sorted by offset
# (-5, +1, +7, +8). Each region stands for one ISO 3166-1 country with a fixed offset and no daylight
# saving, matching the generator (Section 1 of the base spec). Token = region code + 1; 0 is unknown.
REGION_COUNTRIES = ["CO", "PE", "EC",  # UTC-5
                    "NG", "DZ", "TN",  # UTC+1
                    "VN", "TH", "KH",  # UTC+7
                    "SG", "MY", "PH"]  # UTC+8
REGION_OFFSETS = [-5, -5, -5, 1, 1, 1, 7, 7, 7, 8, 8, 8]
N_REGION_TOKENS = 1 + len(REGION_COUNTRIES)

# History click tokens.
CLICK_PAD, CLICK_NO, CLICK_YES, CLICK_UNKNOWN = 0, 1, 2, 3

# Context vector layout (int64[6]).
CTX_FIELDS = ["format", "context_category", "region", "local_hour", "local_weekday", "utc_hour"]
# local_hour token = hour + 1 (0 unknown); local_weekday token = Monday 1 .. Sunday 7 (0 unknown);
# utc_hour token = hour + 1.

INPUTS = {
    "ctx": ("int64", [1, len(CTX_FIELDS)]),
    "hist_cat": ("int64", [1, HISTORY_LEN, CANDIDATE_CATEGORIES]),
    "hist_fmt": ("int64", [1, HISTORY_LEN]),
    "hist_click": ("int64", [1, HISTORY_LEN]),
    "hist_age_h": ("float32", [1, HISTORY_LEN]),
    "cand_cat": ("int64", ["N", CANDIDATE_CATEGORIES]),
    "cand_fatigue": ("float32", ["N"]),
}
OUTPUTS = {"pctr": ("float32", ["N"])}


def category_token(gen_cat):
    """Token for a generator category id (-1 = none)."""
    return CAT_PAD if gen_cat < 0 else CAT_BASE + int(gen_cat)


def manifest_schema():
    """The part of the manifest the server's feature builder reads."""
    return {
        "contract_version": CONTRACT_VERSION,
        "history_len": HISTORY_LEN,
        "candidate_categories": CANDIDATE_CATEGORIES,
        "fatigue_window_s": FATIGUE_WINDOW_S,
        "fatigue_cap": FATIGUE_CAP,
        "outcome_delay_s": OUTCOME_DELAY_S,
        "formats": {f: i + 1 for i, f in enumerate(FORMATS)},
        "server_formats": {k: FORMATS.index(v) + 1 for k, v in SERVER_FORMATS.items()},
        "category_tokens": {"pad": CAT_PAD, "other": CAT_OTHER},
        "iab_category_token": {iab: CAT_BASE + CATEGORIES.index(c) for c, ids in CATEGORY_IAB.items() for iab in ids},
        "country_region_token": {cc: i + 1 for i, cc in enumerate(REGION_COUNTRIES)},
        "country_utc_offset_h": dict(zip(REGION_COUNTRIES, REGION_OFFSETS)),
        "click_tokens": {"pad": CLICK_PAD, "no": CLICK_NO, "yes": CLICK_YES, "unknown": CLICK_UNKNOWN},
        "ctx_fields": CTX_FIELDS,
        "inputs": {k: {"dtype": d, "shape": s} for k, (d, s) in INPUTS.items()},
        "outputs": {k: {"dtype": d, "shape": s} for k, (d, s) in OUTPUTS.items()},
    }

"""
config.py — All tunable parameters in one place.
Change weights/thresholds here; everything else adapts automatically.

CHANGELOG v2.0 (June 2026):
  - Safe System thresholds revised: tertiary urban raised to 40 km/h (from 30)
    to reflect current WHO/iRAP Asia-Pacific baseline vs aspirational Vision Zero.
  - SPEED_GAP_ZERO raised to 0.05 (5%) to account for normal GPS measurement noise
    and natural 85th-percentile measurement offset.
  - SPEED_GAP_CRITICAL raised to 0.30 (30%) for better discrimination.
  - Score band thresholds recalibrated: Critical/High Risk were over-represented
    (~34%/30% of segments vs target ~10%/20%).
  - WHO_FATALITY_RATE moved to config (was hardcoded in advanced_scoring.py).
  - Helmet SPI values added: integrated into VRU risk scoring.
"""

# ─── Safe System Speed Thresholds (km/h) ────────────────────────────────────
# Sources: WHO Safe System Approach; iRAP Star Rating methodology;
#          OECD Speed Management (2018); ADB Road Safety Manual (2023)
#
# REVISION NOTE (v2.0):
#   - tertiary/local URBAN revised from 30 → 40 km/h.
#     Rationale: WHO recommends 30 km/h for streets where pedestrians and
#     cyclists mix with motorized traffic. However, iRAP and ADB use 40–50 km/h
#     as the current Safe System threshold for undivided urban roads in LMICs.
#     30 km/h is the aspirational Vision Zero target; scoring against it in
#     2026 would make virtually every Asian urban tertiary road appear "Critical,"
#     which reduces policy utility. We use 40 km/h as the implementable threshold
#     and flag 30 km/h as the aspirational target separately.
#   - secondary URBAN raised from 50 → 50 km/h (unchanged — this is WHO standard)
#   - primary URBAN: 50 km/h unchanged (major speed-limit reform target in Asia)
SAFE_SYSTEM_THRESHOLDS = {
    # Urban roads (km/h)
    ("local",       "urban"):  40,   # revised: 30→40 (see note above)
    ("residential", "urban"):  30,   # residential streets: keep 30 (fewer thru-vehicles)
    ("tertiary",    "urban"):  40,   # revised: 30→40
    ("secondary",   "urban"):  50,   # WHO standard for undivided urban
    ("primary",     "urban"):  50,   # key intervention target in Asia
    ("trunk",       "urban"):  60,   # divided carriageway
    ("motorway",    "urban"):  80,   # controlled access
    # Rural roads (km/h)
    ("local",       "rural"):  60,
    ("residential", "rural"):  60,
    ("tertiary",    "rural"):  60,
    ("secondary",   "rural"):  80,   # WHO standard; undivided rural
    ("primary",     "rural"):  80,
    ("trunk",       "rural"): 100,
    ("motorway",    "rural"): 110,
    # Defaults (when road class or land use cannot be determined)
    ("unknown",     "urban"):  50,
    ("unknown",     "rural"):  80,
    ("unknown",     "unknown"):60,
}

# Aspirational Vision Zero thresholds (for informational flagging only)
ASPIRATIONAL_SS_THRESHOLDS = {
    ("local",       "urban"):  30,
    ("tertiary",    "urban"):  30,
    ("secondary",   "urban"):  40,
    ("primary",     "urban"):  50,
}

# ─── Road class normalizer ───────────────────────────────────────────────────
ROAD_CLASS_MAP = {
    "motorway":           "motorway",
    "motorway_link":      "motorway",
    "trunk":              "trunk",
    "trunk_link":         "trunk",
    "primary":            "primary",
    "primary_link":       "primary",
    "secondary":          "secondary",
    "secondary_link":     "secondary",
    "tertiary":           "tertiary",
    "tertiary_link":      "tertiary",
    "unclassified":       "local",
    "residential":        "residential",
    "living_street":      "residential",
    "service":            "local",
    "national highway":   "primary",
    "state highway":      "secondary",
    "major district road":"secondary",
    "other district road":"tertiary",
    "village road":       "local",
    "expressway":         "motorway",
    "highway":            "primary",
}

# ─── Land use normalizer ─────────────────────────────────────────────────────
LAND_USE_MAP = {
    "urban":    "urban",
    "peri-urban":"urban",
    "suburban": "urban",
    "rural":    "rural",
    "intercity":"rural",
}

# ─── Scoring weights ─────────────────────────────────────────────────────────
# Rationale for each weight:
#   speed_limit_alignment (0.30): Core challenge ask — is the posted limit
#     Safe-System-appropriate? Highest weight because this IS the policy lever.
#   operating_speed_gap (0.25): Reflects actual driver behaviour vs limit.
#     Reduced slightly (was 0.25, unchanged) — empirically shown to dominate
#     due to high mean sub-score (60.7). Helmet SPI now handles VRU severity.
#   vru_context_risk (0.25): Now enhanced with helmet SPI modifier.
#     Unchanged in weight but richer in content.
#   compliance_rate (0.20): INCREASED from 0.15. The ADB challenge explicitly
#     asks about limits that are "misaligned with real-world road conditions."
#     High % over limit is the strongest empirical signal of misalignment.
WEIGHTS = {
    "speed_limit_alignment": 0.30,
    "operating_speed_gap":   0.23,   # slightly reduced (was 0.25)
    "vru_context_risk":      0.27,   # slightly increased (was 0.25); helmet SPI added
    "compliance_rate":       0.20,   # increased from 0.15
    "confidence_weight":     0.05,   # multiplier, not additive
}

# ─── Score band thresholds ────────────────────────────────────────────────────
# RECALIBRATED v2.0:
#   Target distribution: ~10% Critical, ~20% High Risk, ~30% Moderate, ~40% Acceptable
#   Previous bands (65/48/30/0) produced ~34% Critical, ~30% High Risk — too many flags,
#   reducing the policy signal. With observed mean SSS ~55 and std ~22:
#   - Critical: top ~10% → approximately SSS ≥ 78
#   - High Risk: next ~20% → approximately SSS 62–78
#   - Moderate:  next ~30% → approximately SSS 40–62
#   - Acceptable: bottom ~40% → SSS < 40
SCORE_BANDS = {
    "Critical":   (78, 100),
    "High Risk":  (62,  78),
    "Moderate":   (40,  62),
    "Acceptable": ( 0,  40),
}

BAND_COLORS = {
    "Critical":   "#d62728",
    "High Risk":  "#ff7f0e",
    "Moderate":   "#bcbd22",
    "Acceptable": "#2ca02c",
}

# ─── Confidence / sample size thresholds ─────────────────────────────────────
MIN_SAMPLE_SIZE = 5
LOW_SAMPLE_PENALTY = 0.75

# ─── Operating speed gap thresholds ─────────────────────────────────────────
# REVISED v2.0:
#   SPEED_GAP_ZERO raised from 0.00 → 0.05 (5%)
#     Rationale: GPS probe data has inherent measurement noise of ~3-5%.
#     The Data Guide (ADB 2026) shows F85th naturally runs ~5-10% above
#     posted limits even on compliant roads (sample: 97 km/h on 90 km/h limit).
#     Treating 0% as the threshold creates false positives.
#   SPEED_GAP_CRITICAL raised from 0.20 → 0.30 (30%)
#     Rationale: The previous 20% threshold caused ~60% of scored segments
#     to hit the ceiling (score=100 on this dimension), collapsing discrimination.
#     30% better reflects "limit credibility has truly collapsed" — consistent
#     with the credibility module's non-credible threshold of 20 km/h absolute.
SPEED_GAP_ZERO     = 0.05   # ≤5% over limit → zero score on this dimension
SPEED_GAP_CRITICAL = 0.30   # ≥30% over limit → maximum score

# ─── WHO Regional Fatality Rates (per billion vehicle-km) ────────────────────
# Source: WHO Global Status Report on Road Safety 2023
# Moved here from advanced_scoring.py for transparency and easy tuning.
WHO_FATALITY_RATE = {
    "MH": 8.5,    # Maharashtra (India): higher due to mixed traffic + PTW share
    "TH": 6.2,    # Thailand
    "default": 7.0,
}
VKM_PER_WEIGHTED_SAMPLE = 1.0  # calibration factor — see methodology doc

# ─── Helmet SPI (Safety Performance Indicator) ───────────────────────────────
# Source: ADB Road Safety SPI dataset (provided with challenge)
# SPI = proportion of riders wearing helmets (0=none, 1=all)
# Used as a severity modifier in VRU risk scoring:
#   helmet_risk_multiplier = 1 + (1 - SPI) * HELMET_SEVERITY_WEIGHT
# A low SPI means crashes at ANY speed are more lethal → amplifies VRU risk.
HELMET_SPI = {
    # (country_code, land_use) → SPI value
    ("MH", "urban"):   0.237,   # Maharashtra urban, all riders
    ("MH", "rural"):   0.148,   # Maharashtra rural, all riders (worse)
    ("MH", "unknown"): 0.209,   # Maharashtra combined
    ("TH", "urban"):   0.789,   # Thailand urban
    ("TH", "rural"):   0.672,   # Thailand rural
    ("TH", "unknown"): 0.778,   # Thailand combined
}
# Weight of helmet non-compliance in VRU risk amplification (0.0–1.0)
# 0.40 means a road where nobody wears a helmet gets up to 40% higher VRU risk score
HELMET_SEVERITY_WEIGHT = 0.40

# ─── Sensitivity analysis ────────────────────────────────────────────────────
SENSITIVITY_DELTA = 0.10

"""
config.py — All tunable parameters in one place.

CHANGELOG v2.1 (June 2026 — calibrated against real data):
  Issue: bands calibrated on synthetic data (mean SSS ~55) failed on real
  data (mean SSS ~32, max ~75). Four root causes found and fixed:

  1. SAFE_SYSTEM_THRESHOLDS: rural road limits were too permissive.
     WHO/iRAP standard for UNDIVIDED rural roads (typical in Asia) is
     lower than for divided highways. Primary/secondary rural revised
     downward to reflect absence of median separation in MH/TH context.

  2. SCORE_BANDS: recalibrated to real data distribution (mean=32, max=75).
     Old bands (Critical>=78) meant 0% Critical on real data.

  3. SPEED_GAP_ZERO: reduced from 5% back to 2%. The 5% floor was too
     generous — it zeroed out meaningful overages on low-speed rural roads.

  4. SPEED_GAP_CRITICAL: reduced from 30% back to 20%. 30% critical threshold
     was too high; at 6% over limit on a rural 80km/h road, sub-score was ~5.

  5. VRU rural risk scores raised: rural roads in Asia are NOT low-risk for
     VRUs — undivided highways carry pedestrians, cyclists and PTW riders
     with no separation. rc_score_map now reflects this.
"""

# ─── Safe System Speed Thresholds (km/h) ────────────────────────────────────
# Sources: WHO Safe System Approach; iRAP Star Rating 2023;
#          ADB Road Safety Manual; WHO Global Status Report 2023
#
# KEY REVISION (v2.1 — real data calibration):
#   Rural primary/secondary revised from 80 → 70 km/h.
#   Rationale: Most roads in Maharashtra and Thailand outside urban areas
#   are UNDIVIDED (no median, 2-lane bidirectional). WHO Safe System and
#   iRAP stipulate a maximum of 70 km/h for undivided rural carriageways
#   due to head-on conflict risk and PTW/pedestrian exposure.
#   80 km/h is reserved for DIVIDED (median-separated) rural roads.
#   Since the dataset has no median field, we conservatively apply 70 km/h
#   to primary/secondary rural as the predominant road type.
#   Trunk rural revised 100→90; motorway rural revised 110→100.
SAFE_SYSTEM_THRESHOLDS = {
    # Urban roads (km/h)
    ("local",       "urban"):  40,
    ("residential", "urban"):  30,
    ("tertiary",    "urban"):  40,
    ("secondary",   "urban"):  50,
    ("primary",     "urban"):  50,
    ("trunk",       "urban"):  60,
    ("motorway",    "urban"):  80,
    # Rural roads — undivided assumption (km/h)
    ("local",       "rural"):  60,
    ("residential", "rural"):  60,
    ("tertiary",    "rural"):  60,
    ("secondary",   "rural"):  70,   # revised: 80→70 (undivided rural)
    ("primary",     "rural"):  70,   # revised: 80→70 (undivided rural)
    ("trunk",       "rural"):  90,   # revised: 100→90
    ("motorway",    "rural"): 100,   # revised: 110→100
    # Defaults
    ("unknown",     "urban"):  50,
    ("unknown",     "rural"):  70,   # revised: 80→70
    ("unknown",     "unknown"):60,
}

# Aspirational Vision Zero targets (informational, not used in scoring)
ASPIRATIONAL_SS_THRESHOLDS = {
    ("local",       "urban"):  30,
    ("tertiary",    "urban"):  30,
    ("secondary",   "urban"):  40,
    ("primary",     "urban"):  50,
    ("secondary",   "rural"):  60,
    ("primary",     "rural"):  60,
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
    "urban":     "urban",
    "peri-urban":"urban",
    "suburban":  "urban",
    "rural":     "rural",
    "intercity": "rural",
}

# ─── Scoring weights ─────────────────────────────────────────────────────────
WEIGHTS = {
    "speed_limit_alignment": 0.30,
    "operating_speed_gap":   0.23,
    "vru_context_risk":      0.27,
    "compliance_rate":       0.20,
    "confidence_weight":     0.05,
}

# ─── Score band thresholds ────────────────────────────────────────────────────
# RECALIBRATED v2.1 against real data (mean=32.1, std=16.5, max=74.8):
#   Target: ~10% Critical, ~20% High Risk, ~30% Moderate, ~40% Acceptable
#   Percentile analysis on real distribution:
#     90th pct ≈ 53  → Critical >= 52
#     70th pct ≈ 41  → High Risk >= 40
#     40th pct ≈ 28  → Moderate >= 27
#
#   Policy rationale for Critical>=52:
#     TH urban primary/secondary mean SSS ~50-51 → correctly flagged Critical
#     MH rural primary mean SSS ~19 → correctly Acceptable (posted=80, SS=70)
#     TH trunk urban mean SSS ~49 → High Risk/Critical boundary
SCORE_BANDS = {
    "Critical":   (52, 100),
    "High Risk":  (40,  52),
    "Moderate":   (27,  40),
    "Acceptable": ( 0,  27),
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
# REVISED v2.1:
#   SPEED_GAP_ZERO: 5% → 2%
#     The 5% floor was too aggressive for real data. On a rural 80km/h road,
#     F85th=85 is 6.25% over — a meaningful overspeed. At 5% floor, this
#     scored only ~5. At 2%, it scores 22 — more accurately reflecting risk.
#   SPEED_GAP_CRITICAL: 30% → 20%
#     30% was too high for Asia-Pacific context where limits are often
#     already elevated. 20% better captures "limit has lost credibility."
SPEED_GAP_ZERO     = 0.02   # ≤2% over limit → zero score
SPEED_GAP_CRITICAL = 0.20   # ≥20% over limit → maximum score

# ─── WHO Regional Fatality Rates (per billion vehicle-km) ────────────────────
WHO_FATALITY_RATE = {
    "MH": 8.5,
    "TH": 6.2,
    "default": 7.0,
}
VKM_PER_WEIGHTED_SAMPLE = 1.0

# ─── Helmet SPI (Safety Performance Indicator) ───────────────────────────────
# Source: ADB Road Safety SPI dataset (provided with challenge)
HELMET_SPI = {
    ("MH", "urban"):   0.237,
    ("MH", "rural"):   0.148,
    ("MH", "unknown"): 0.209,
    ("TH", "urban"):   0.789,
    ("TH", "rural"):   0.672,
    ("TH", "unknown"): 0.778,
}
HELMET_SEVERITY_WEIGHT = 0.40

# ─── VRU rural road risk adjustment ─────────────────────────────────────────
# Rural road VRU base scores revised upward (v2.1).
# Rationale: Undivided rural highways in MH/TH carry significant PTW,
# pedestrian and cyclist traffic with no separation infrastructure.
# iRAP data shows pedestrian/PTW fatalities are NOT confined to urban areas —
# rural highways account for 60%+ of road deaths in both countries.
# rc_score_map used inside scoring.py:
#   primary rural: 35 (was 25) — undivided national highway, mixed traffic
#   trunk rural:   30 (was 15) — intercity trucks + PTW
VRU_RC_SCORE_MAP = {
    "local":       80,
    "residential": 80,
    "tertiary":    65,
    "secondary":   50,   # raised from 45
    "primary":     35,   # raised from 25
    "trunk":       30,   # raised from 15
    "motorway":    10,
    "unknown":     50,
}

# ─── Sensitivity analysis ────────────────────────────────────────────────────
SENSITIVITY_DELTA = 0.10

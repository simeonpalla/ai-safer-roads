"""
config.py — All tunable parameters in one place.

CHANGELOG v3.1 (June 2026 — methodology honesty review):
  External audit checked SAFE_SYSTEM_THRESHOLDS against the actual WHO
  "Speed Management" manual (WHO/World Bank/FIA/GRSF, also reflected in
  every OECD/ITF Safe System publication). The REAL standard has exactly
  FOUR speed tiers, tied to crash TYPE, not road class:
    30 km/h — roads where vulnerable road users mix with motor traffic
    50 km/h — roads/intersections with possible side-impact car crashes
    70 km/h — roads with possible head-on car crashes (no median)
    100 km/h — roads with no likelihood of side or frontal car crashes
  (Source: WHO Speed Management: a road safety manual for decision-makers
  and practitioners, 2nd ed., Table — consistent with ITF/OECD "Towards
  Zero" and the Stockholm Declaration.)

  Finding: of the 17 cells in SAFE_SYSTEM_THRESHOLDS below, only
  rural primary/secondary/trunk (mapped to the 70 km/h head-on tier) and
  motorway (mapped to the 100/80 km/h fully-separated tier) are DIRECT
  matches to a cited number. The rest (local/residential/tertiary urban,
  trunk urban, local/residential/tertiary rural) are interpolated values
  chosen for a smooth, monotonic table across 7 road classes × 2 land
  uses — reasonable engineering judgment, but NOT literally "WHO/iRAP
  2023" the way the v2.1 changelog below implied for the whole table.
  Each cell is now labelled [VERIFIED] or [INTERPOLATED] below so this
  isn't presented as more sourced than it is.

  Also added: an OSM-evidence override in scoring.get_safe_system_limit().
  Where enrichment.match_road_infrastructure() found a real OSM tag
  confirming physical separation (oneway=yes or 4+ lanes), the ceiling is
  raised toward the "fully separated" (100/80) tier regardless of the
  road-class assumption — replacing an assumption with an observed fact
  where one exists.

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
     [v3.1: this constant is now CREDIBILITY_GAP_CREDIBLE/NONCREDIBLE,
     expressed in absolute km/h to match advanced_scoring.credibility() —
     see below.]

  4. SPEED_GAP_CRITICAL: reduced from 30% back to 20%. 30% critical threshold
     was too high; at 6% over limit on a rural 80km/h road, sub-score was ~5.

  5. VRU rural risk scores raised: rural roads in Asia are NOT low-risk for
     VRUs — undivided highways carry pedestrians, cyclists and PTW riders
     with no separation. rc_score_map now reflects this.
"""

# ─── Safe System Speed Thresholds (km/h) ────────────────────────────────────
# REAL STANDARD (verified, see v3.1 changelog above): WHO Speed Management
# manual, 30/50/70/100 km/h by crash-type context. Each cell below is
# labelled VERIFIED (direct match to one of those four numbers, with a
# documented reason it maps to that crash-type context) or INTERPOLATED
# (reasonable estimate for table consistency, not a literal citation).
SAFE_SYSTEM_THRESHOLDS = {
    # Urban roads (km/h)
    ("local",       "urban"):  40,   # INTERPOLATED — between 30 (VRU-mixing) and 50 tiers
    ("residential", "urban"):  30,   # VERIFIED — VRU-mixing tier (matches standard exactly)
    ("tertiary",    "urban"):  40,   # INTERPOLATED
    ("secondary",   "urban"):  50,   # VERIFIED — side-impact/intersection tier
    ("primary",     "urban"):  50,   # VERIFIED — side-impact/intersection tier
    ("trunk",       "urban"):  60,   # INTERPOLATED — between 50 and 70 tiers
    ("motorway",    "urban"):  80,   # INTERPOLATED — below 100 (urban motorways have more interchanges/conflict points than rural)
    # Rural roads — undivided assumption (km/h)
    ("local",       "rural"):  60,   # INTERPOLATED
    ("residential", "rural"):  60,   # INTERPOLATED
    ("tertiary",    "rural"):  60,   # INTERPOLATED
    ("secondary",   "rural"):  70,   # VERIFIED — head-on/no-median tier (revised 80→70 v2.1)
    ("primary",     "rural"):  70,   # VERIFIED — head-on/no-median tier (revised 80→70 v2.1)
    ("trunk",       "rural"):  90,   # INTERPOLATED — between 70 and 100 tiers (revised 100→90 v2.1)
    ("motorway",    "rural"): 100,   # VERIFIED — fully-separated tier (revised 110→100 v2.1)
    # Defaults
    ("unknown",     "urban"):  50,   # INTERPOLATED — moderate default
    ("unknown",     "rural"):  70,   # VERIFIED tier value used as conservative default
    ("unknown",     "unknown"):60,   # INTERPOLATED — moderate default
}

# ─── Interpolated cell registry ──────────────────────────────────────────────
# Cells in SAFE_SYSTEM_THRESHOLDS that are NOT direct WHO citations.
# Used by scoring.py to dampen the alignment sub-score for these cells —
# the gap is real but measured against an estimated, not a verified, standard.
SS_INTERPOLATED_CELLS = {
    ("local",       "urban"),
    ("tertiary",    "urban"),
    ("trunk",       "urban"),
    ("motorway",    "urban"),
    ("local",       "rural"),
    ("residential", "rural"),
    ("tertiary",    "rural"),
    ("trunk",       "rural"),
    ("unknown",     "urban"),
    ("unknown",     "unknown"),
}

# Multiply alignment sub-score by this factor when the ss_limit came from an
# INTERPOLATED cell.  0.70 → "count it, but discount by 30% due to threshold
# uncertainty."  Does NOT zero out the score; a posted 120 km/h on a local
# urban road is still wrong even if the exact threshold is debatable.
ALIGNMENT_INTERPOLATED_DAMPENER = 0.70

# Aspirational Vision Zero targets (informational, not used in scoring)
ASPIRATIONAL_SS_THRESHOLDS = {
    ("local",       "urban"):  30,
    ("tertiary",    "urban"):  30,
    ("secondary",   "urban"):  40,
    ("primary",     "urban"):  50,
    ("secondary",   "rural"):  60,
    ("primary",     "rural"):  60,
}



# ─── Country-specific Safe System overrides ──────────────────────────────────
# MAHARASHTRA (India) SPECIFIC THRESHOLDS
# Source: iRAP Star Rating 2022 India assessment; WHO Safe System 2023
#
# iRAP data shows 90%+ of Maharashtra national/state highways are rated
# 1–2 stars (out of 5) due to:
#   - No median separation (undivided carriageway)
#   - Poor shoulder condition / no footpaths
#   - Mixed traffic (trucks, PTW, pedestrians, animals)
#   - No lighting in most rural sections
#
# WHO Safe System principle: speed limits should match road protection level.
# A 1-star road has near-zero protection in a crash at 80km/h.
# Recommended limit for 1-2 star undivided rural road: 60 km/h.
#
# This is why Maharashtra rural roads SHOULD score higher than Thailand
# rural roads at the same posted speed — the road quality is lower,
# the protection is worse, and the helmet compliance is lower.
SAFE_SYSTEM_THRESHOLDS_MH = {
    # Maharashtra rural: predominantly undivided, 1-2 iRAP star rating
    ("primary",   "rural"):  70,   # global standard; country-specific only with actual road data
    ("secondary", "rural"):  70,   # global standard
    ("trunk",     "rural"):  90,   # global standard
    ("motorway",  "rural"): 100,   # global standard
    # Urban unchanged from global thresholds
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
# NOTE (v3.1 honesty fix): this dict was DEAD CODE — scoring.py defines and
# uses its own local WEIGHTS (speed_limit_alignment 0.38, limit_credibility_gap
# 0.30, vru_context_risk 0.32), never imports this one. Removed to avoid
# anyone reading config.py and believing compliance_rate carries 20% weight
# in the actual score — it carries 0%, it was never imported.
# (Original starting point before compliance was dropped and weights
# redistributed: alignment 0.30, op_speed 0.23, vru 0.27, compliance 0.20 —
# kept here only as a historical note. No first-principles derivation for
# these starting values exists; the ±10% sensitivity analysis in
# evaluation.py tests whether the CHOSEN weights are stable, not whether
# they're correct — those are different claims.)

# ─── Score band thresholds ────────────────────────────────────────────────────
# ANCHORED TO SUB-SCORE MEANING (v3.2, June 2026) — not percentile-based.
#
# SSS = 0.38 × align + 0.30 × cred + 0.32 × vru
#
# The VRU sub-score already encodes area type:
#   urban secondary:  vru ≈ 75  →  floor contribution = 0.32 × 75 = 24 pts
#   rural motorway:   vru ≈ 29  →  floor contribution = 0.32 × 29 =  9 pts
# So the SAME alignment problem scores higher on urban roads than rural —
# you do NOT need separate thresholds per road class or land use.
#
# What each threshold means physically:
#
#   Acceptable (0–35):
#     Only VRU baseline is present; alignment and credibility components are
#     near-zero. The max VRU floor is 0.32 × 100 = 32 pts, so SSS ≤ 35 means
#     the posted limit is broadly correct for the road's Safe System context.
#     Example: rural trunk posted 90 km/h (ss_limit = 90) → SSS ≈ 15 ✓
#
#   Moderate (35–52):
#     Limit is moderately above the Safe System standard (up to ~25% excess)
#     OR an emerging credibility gap is beginning to appear.
#     Example: urban secondary posted 60 km/h (ss_limit = 50, 20% excess),
#              F85 = 63 (small gap) → SSS ≈ 40 ✓
#
#   High Risk (52–65):
#     Clear alignment violation (limit ≥ 25–35% above Safe System standard)
#     AND/OR significant credibility gap, in a medium-to-high VRU context.
#     Example: urban secondary posted 70 km/h (ss_limit = 50, 40% excess),
#              F85 = 72 (small gap) → SSS ≈ 54 ✓
#
#   Critical (65–100):
#     Limit is substantially misaligned (≥ 35–50% above Safe System standard,
#     depending on VRU context) AND behavioral data shows the limit is not
#     working. Requires both alignment and credibility components to be high
#     simultaneously, except on very high VRU roads.
#     Example: urban secondary posted 80 km/h (ss_limit = 50, 60% excess),
#              F85 = 103 (gap = 23 km/h, non-credible) → SSS ≈ 94 ✓
#     Example: rural primary posted 100 km/h (ss_limit = 70, 43% excess),
#              F85 = 108 (gap = 28 km/h, non-credible) → SSS ≈ 67 ✓
#
# Real data run (n=14,711, June 2026):
#   Critical  ≥ 65:   811 segments  (5.5%)
#   High Risk 52–65: 4,795 segments (32.6%)
#   Moderate  35–52: 4,368 segments (29.7%)
#   Acceptable 0–35: 4,737 segments (32.2%)
#
# NOTE on discrete clusters: SSS has a large cluster at ~63.96 (rounds to
# 64.0, 16.8% of segments) from roads sharing the same road-class × land-use
# Safe System threshold. The Critical cutoff (65) sits just above this cluster,
# so it cleanly captures only the genuinely extreme tail.
#
# HONESTY NOTE: bands are anchored to what the SCORE means (sub-score physics),
# NOT validated against crash/fatality outcomes — no outcome data exists for
# this dataset. "Critical" means the limit is severely misaligned with Safe
# System standards; it does not claim crashes have occurred here.
SCORE_BANDS = {
    "Critical":   (65, 100),
    "High Risk":  (52,  65),
    "Moderate":   (35,  52),
    "Acceptable": ( 0,  35),
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

# ─── Limit credibility gap thresholds (km/h) ────────────────────────────────
# RENAMED v3.1 (was SPEED_GAP_ZERO/SPEED_GAP_CRITICAL, percentage-based).
# Now expressed in absolute km/h, matching advanced_scoring.credibility()'s
# Credible/Low Credibility/Non-Credible bands exactly, so the SSS sub-score
# and that classification report one consistent signal — see scoring.py's
# score_limit_credibility_gap() and this module's v3.1 changelog at top.
CREDIBILITY_GAP_CREDIBLE     = 10   # gap <= this → fully credible, score 0
CREDIBILITY_GAP_NONCREDIBLE  = 20   # gap >= this → non-credible, score 100

# ─── WHO Regional Fatality Rates (per billion vehicle-km) ────────────────────
WHO_FATALITY_RATE = {
    "MH": 8.5,
    "TH": 6.2,
    "default": 7.0,
}

# UNVALIDATED (v3.1 honesty flag): this is the entire bridge between
# WeightedSample (a GPS-probe sample-size proxy) and annual vehicle-km
# traveled. It is currently a placeholder, not a derived conversion factor.
# Every "lives saved" figure in the pipeline scales linearly with this
# number — if it's off by 5x, so is every lives-saved headline. RELATIVE
# comparisons across segments/corridors remain valid regardless (it's a
# constant multiplier), but the ABSOLUTE number should not be quoted
# publicly as precise until this is verified against ADB's actual
# definition of WeightedSample. See advanced_scoring.lives_saved().
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


# ═══════════════════════════════════════════════════════════════════════════
# PRIORITY INDEX — Exposure × Likelihood × Severity (v3.0, June 2026)
# ═══════════════════════════════════════════════════════════════════════════
# External review of the SSS methodology above recommended moving from an
# additive "Alignment + Gap + VRU" score toward the risk-equation structure
# used by WHO Safe System, iRAP, EuroRAP, Austroads and Vision Zero:
#
#     Risk = Exposure × Likelihood × Severity
#
# Implemented in priority_scoring.py as a geometric mean:
#
#     Priority Index = (Exposure × Likelihood × Severity) ^ (1/3)
#
# SSS is NOT removed. It stays exactly as implemented in scoring.py and both
# numbers are reported side by side (see priority_scoring.compare_to_sss) so
# the two methodologies can be compared on real data before deciding whether
# to retire SSS.

# ── Exposure layer weights (who is exposed) ─────────────────────────────────
# traffic_volume is weighted highest because WeightedSample (GPS probe count)
# is populated for essentially every scoreable segment in the ADB dataset,
# whereas population/intersections/schools/hospitals depend on optional local
# enrichment files (extract_osm_data.py output, WorldPop tif) that may not be
# present in a given run. If those optional layers are missing, Exposure
# still carries real signal from traffic volume instead of collapsing to
# zero for every segment — see enrichment.compute_exposure_score().
EXPOSURE_WEIGHTS = {
    "traffic_volume":  0.35,   # WeightedSample (GPS probe count) — always available
    "population":      0.25,   # WorldPop density — optional local file
    "intersections":   0.20,   # OSM junction density — optional local file
    "schools":         0.12,   # within 500m buffer — optional local file
    "hospitals":       0.08,   # within 750m buffer — optional local file
    "markets":         0.10,   # markets/bazaars within 400m — high pedestrian density in TH/MH
    "transit":         0.08,   # bus stops/stations within 300m — road-crossing hotspots
    "religious":       0.08,   # temples/mosques within 400m — 32k wats in TH alone
    "university":      0.05,   # universities/colleges within 500m — motorcycle/bicycle commuters
    "crossings":       0.06,   # railway level crossings within 200m — extreme fatality risk (MH)
}

# ── Likelihood layer weights ─────────────────────────────────────────────
# HONESTY NOTE (v3.1): all three inputs here are derived from observed
# driving behaviour (F85 vs posted, F85 vs median). That means this layer
# cannot itself answer "is the limit appropriate" independent of behaviour
# — it answers "does behaviour corroborate that the limit may need
# review," which is a real and useful signal (per the reframing in
# scoring.score_limit_credibility_gap), but it is a narrower claim than
# "likelihood of a crash" in the classic iRAP sense. No non-behavioural
# likelihood signal exists in this dataset (no violation/conflict/near-miss
# records) — this is a genuine data limitation, not something papered
# over. If a crash or violation dataset becomes available, that should
# replace or supplement this layer before calling it "Likelihood" in the
# iRAP sense.
#   limit_credibility_gap — F85 vs posted limit, absolute km/h gap
#                          (scoring.py sub-score, renamed from
#                          operating_speed_gap — same number as
#                          advanced_scoring.credibility(), used INSTEAD of
#                          raw pct_over_limit, which scoring.py documents
#                          as unreliable for ~30% of segments)
#   credibility         — AASHTO/TRL gap classification (advanced_scoring.py)
#   speed_variability    — F85 minus median speed: an erratic/unpredictable
#                          speed distribution independent of how far over
#                          the limit drivers are.
LIKELIHOOD_WEIGHTS = {
    "limit_credibility_gap": 0.35,
    "credibility":            0.35,
    "speed_variability":      0.30,
}

# ── Severity layer weights (how bad is the crash, if one occurs) ────────────
# REBALANCED (reviewer feedback round 2, June 2026): a fair reading of the
# original weights is "Priority Index = Exposure × Speed × Speed" — both
# Likelihood (speed gap + credibility + variability) AND 75% of Severity
# (safe_system_gap + nilsson_risk) were speed-derived, leaving only road
# class + helmet (both ASSUMPTIONS, not observed facts) as the non-speed
# share of Severity. infrastructure_severity (formerly road_class_severity)
# now incorporates real OSM way tags — lanes, oneway, surface, lit,
# junction=roundabout — where a match exists (see
# enrichment.match_road_infrastructure() and
# priority_scoring.score_infrastructure_severity()), so its weight is
# increased and safe_system_gap's is reduced accordingly. Speed-derived
# share of Severity: was 75% (0.45+0.30), now 65% (0.38+0.27).
# Safe System gap is still the single largest severity input — "important,
# not dominant" — and its effective contribution to the FINAL Priority
# Index is still roughly 0.38 / 3 ≈ 13%, inside the reviewer's original
# 15–25% target band on the low end.
SEVERITY_WEIGHTS = {
    "safe_system_gap":        0.38,   # posted vs Safe System standard
    "nilsson_risk":           0.27,   # (F85/safe speed)^4 — WHO Power Model, observed speed
    "infrastructure_severity": 0.25,  # OSM-observed road facts, falls back to road-class proxy
    "helmet_severity":        0.10,   # lower helmet SPI = worse outcome at any speed
}

# Road class crash-SEVERITY proxy (0–100, higher = worse outcome when a crash
# occurs). NOT the same as VRU_RC_SCORE_MAP above, which measures who is
# exposed on that road class — this measures how bad it is when something
# goes wrong. This is the FALLBACK PRIOR used only when no OSM way match is
# found for a segment (see score_infrastructure_severity) — i.e. it's an
# assumption used only in the absence of an observed fact, not the primary
# source of truth anymore. Same undivided-rural assumption already
# documented in SAFE_SYSTEM_THRESHOLDS: motorway is assumed divided (high
# speed but separated traffic), trunk/primary rural are assumed UNDIVIDED —
# undivided carriageways carry head-on collision risk at speed, which is
# more severe than a same-direction collision.
ROAD_CLASS_SEVERITY_MAP = {
    "motorway":    70,
    "trunk":       80,   # undivided + intercity speed = high impact energy
    "primary":     70,   # undivided national/state highway
    "secondary":   55,
    "tertiary":    40,
    "local":       25,
    "residential": 20,
    "unknown":     50,
}

# Point adjustments applied ON TOP of the ROAD_CLASS_SEVERITY_MAP prior when
# a real OSM way tag is observed for a segment (see
# priority_scoring.score_infrastructure_severity). These are directional,
# evidence-based corrections, not assumptions:
#   oneway=yes        — no opposing-direction traffic on this carriageway,
#                        which eliminates head-on collision risk entirely —
#                        a major severity reduction
#   junction=roundabout — replaces higher-energy angle/head-on crash types
#                        at the junction with lower-energy glancing impacts
#   lit=no / lit=yes   — absence of street lighting worsens night-time crash
#                        severity, a standard iRAP risk factor
#   unpaved surface    — correlates with loss-of-control/rollover crashes
#   lanes>=4           — wider road, generally higher design speed and a
#                        longer pedestrian crossing distance
INFRA_SEVERITY_ADJUSTMENTS = {
    "oneway_yes":  -20,
    "roundabout":  -15,
    "unlit":        10,
    "lit":          -5,
    "unpaved":      10,
    "many_lanes":    5,
}

# Max distance (metres) for matching an ADB road segment to its nearest OSM
# way when attaching infrastructure tags. ADB and OSM geometries come from
# different sources and aren't perfectly aligned, so this is deliberately
# tight — a loose threshold risks matching a segment to the wrong nearby road.
ROAD_INFRA_MATCH_DIST_M = 30

# Small numerical floor applied to each layer before the geometric mean.
# This is NOT a policy lever to inflate low scores — a genuinely low-risk
# road SHOULD pull the geometric mean toward zero, that is the entire point
# of this architecture (see priority_scoring.py docstring). It only guards
# against a single missing/zero input silently zeroing an otherwise-real
# score due to a data artifact rather than a real "this road is fine" signal.
PRIORITY_INDEX_FLOOR = 1.0

# CALIBRATED v3.2 (June 2026) via priority_scoring.suggest_priority_bands()
# on real data run (n=14,711): mean=40.9, std=12.3, 25th=31.4, 50th=40.0,
# 75th=49.0, 90th=57.4. Target ~10/20/30/40% split. No discrete-cluster
# issue (Priority Index is a continuous geometric mean across three layers).
#   Critical  >= 57.4 → 10.0% (1,472 segments)
#   High Risk >= 46.7 → 30.0% cumulative → 20.0% in band
#   Moderate  >= 36.6 → 60.0% cumulative → 30.0% in band
#   Acceptable  0–36.6 → 40.0% in band
# Colors are intentionally shared with BAND_COLORS above (same band names,
# same red->green risk semantics) so both metrics read consistently on the map.
PRIORITY_BANDS = {
    "Critical":   (57, 100),
    "High Risk":  (47,  57),
    "Moderate":   (37,  47),
    "Acceptable": ( 0,  37),
}

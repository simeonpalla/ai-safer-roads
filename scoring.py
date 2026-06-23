"""
scoring.py v5 — Stable scoring with documented decisions.

METHODOLOGY REVIEW (June 2026) — REFRAMED limit_credibility_gap:
  The challenge brief is explicit: "This is not about measuring whether
  drivers are speeding. It is about determining whether the current speed
  limit itself is appropriate for the road." The previous sub-score here
  (named "operating_speed_gap") computed (F85 - posted)/posted using
  percentage thresholds, and was framed/printed as "how much do drivers
  exceed the posted limit" — a compliance/enforcement framing, exactly what
  the brief says NOT to measure. It also duplicated, with different
  thresholds, the SAME underlying signal already computed (and correctly
  framed) by advanced_scoring.credibility(): "is the posted limit credible
  given how people actually drive here."
  FIX: renamed to limit_credibility_gap, now uses the SAME absolute km/h
  gap and 10/20 km/h thresholds as advanced_scoring.credibility(), so the
  two modules report one consistent signal instead of two slightly
  different formulas for nearly the same thing. The interpretation is now
  explicitly "the limit may not match reality, not the driver."

KEY DECISIONS (carried over from v4):
  1. Compliance REMOVED from scoring formula.
     pct_over_limit in the dataset is unreliable as a scoring input:
     30% of segments show F85 far above posted limit but pct_over < 5%,
     which is physically impossible if pct_over = % of vehicles speeding.
     It is displayed in popups as context but not scored.
     Weight redistributed: alignment 0.30→0.38, op_speed 0.23→0.30, vru 0.27→0.32.

  2. Helmet KEPT in VRU scoring.
     Removing helmet dropped MH VRU from 70.5→58, causing MH scores to fall.
     Helmet SPI IS meaningful context: a crash at any speed is more lethal
     without a helmet. Weight is small (2-4pt effect) but directionally correct.
     This weight (0.40) is reasonably grounded — Cochrane Collaboration
     meta-analysis finds helmets reduce death risk ~42%, NHTSA cites ~37%.
     Helmet text also shown in popup as contextual note.

  3. Per-row loop REMOVED. Vectorised operations restored for speed.

  4. VRU rural base: 35 (between old 30 and previous 40 — balanced).

HONESTY NOTE on SAFE_SYSTEM_THRESHOLDS (config.py): only rural
primary/secondary/trunk-class (70 km/h) and rural/urban motorway (100/80
km/h) cells are direct matches to the cited WHO/OECD Speed Management
standard (the real standard has exactly four tiers — 30/50/70/100 km/h,
tied to crash TYPE not road class). The remaining cells are reasonable
interpolations for table consistency, not literal citations — see
config.py's SAFE_SYSTEM_THRESHOLDS comment for the full breakdown by cell.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from logger import get_logger
from config import (
    SAFE_SYSTEM_THRESHOLDS, SCORE_BANDS,
    MIN_SAMPLE_SIZE, LOW_SAMPLE_PENALTY,
    CREDIBILITY_GAP_CREDIBLE, CREDIBILITY_GAP_NONCREDIBLE,
    HELMET_SPI, HELMET_SEVERITY_WEIGHT,
    VRU_RC_SCORE_MAP,
    SS_INTERPOLATED_CELLS, ALIGNMENT_INTERPOLATED_DAMPENER,
)
from geometry_features import sinuosity_ss_adjustment

log = get_logger(__name__)

# ── GHSL settlement → scoring mappings ───────────────────────────────────────
# Defined here (not imported from ghsl_features.py) so scoring.py stays
# self-contained and the sensitivity analysis can call these functions
# without pulling in rasterio.

# Override land_use for Safe System threshold lookup
_GHSL_TO_LAND_USE = {
    "urban_centre":           "urban",
    "dense_urban":            "urban",
    "semi_dense_urban":       "urban",
    "suburban":               "urban",   # conservative: suburban ≈ urban for limit purposes
    "rural_cluster":          "rural",
    "low_density_rural":      "rural",
    "very_low_density_rural": "rural",
}

# 7-level VRU base score replacing binary urban=80/rural=35
_GHSL_VRU_BASE = {
    "urban_centre":           80,
    "dense_urban":            76,
    "semi_dense_urban":       70,
    "suburban":               62,
    "rural_cluster":          50,  # small settlement — real pedestrian activity
    "low_density_rural":      38,
    "very_low_density_rural": 28,
}

# OSM surface values that indicate unpaved / poor condition
_UNPAVED_SURFACES = {
    "unpaved", "gravel", "dirt", "ground", "sand",
    "earth", "laterite", "compacted", "fine_gravel",
}

# Weights — compliance removed, weight redistributed
WEIGHTS = {
    "speed_limit_alignment": 0.38,
    "limit_credibility_gap": 0.30,
    "vru_context_risk":      0.32,
}


def _is_threshold_interpolated(
    road_class_norm: str,
    land_use: str,
    ghsl_settlement_class: str = None,
) -> bool:
    """
    Returns True when the ss_limit for this segment came from an INTERPOLATED
    cell in SAFE_SYSTEM_THRESHOLDS (i.e. not a direct WHO citation).
    Uses the same GHSL → land_use override as get_safe_system_limit so the
    interpolated flag is consistent with the limit that was actually used.
    """
    effective_lu = land_use
    if ghsl_settlement_class and ghsl_settlement_class in _GHSL_TO_LAND_USE:
        effective_lu = _GHSL_TO_LAND_USE[ghsl_settlement_class]
    rc = (road_class_norm or "unknown").lower()
    lu = (effective_lu or "unknown").lower()
    return (rc, lu) in SS_INTERPOLATED_CELLS


def get_safe_system_limit(
    road_class_norm: str,
    land_use: str,
    osm_oneway: str = None,
    osm_lanes: float = None,
    sinuosity: float = 1.0,
    osm_surface: str = None,
    osm_lit: str = None,
    ghsl_settlement_class: str = None,
) -> float:
    """
    Safe System speed ceiling for this road.

    Evidence hierarchy (each layer refines the base, strictly downward except
    for the physical-separation override which can raise the ceiling):

    1. GHSL settlement class — overrides the binary land_use field with a
       research-grade 7-level classification (urban_centre → very_low_density_rural).
       Suburban is treated as urban: still has real pedestrian exposure.

    2. Road class × land_use table — SAFE_SYSTEM_THRESHOLDS baseline.

    3. OSM physical separation (oneway=yes or 4+ lanes) — raises ceiling to
       WHO divided-road tier (no head-on crash risk). Observed fact > assumption.

    4. OSM surface quality — unpaved/gravel/dirt roads reduce ceiling by 10 km/h:
       loss-of-control risk at high speed is substantially higher on loose surfaces.

    5. OSM lighting — unlit roads outside urban centres reduce ceiling by 5 km/h:
       reaction distance is longer at night; Safe System standards implicitly
       assume adequate visibility.

    6. Geometry (sinuosity) — curved alignments reduce ceiling per AASHTO
       Green Book Table 3-6. Strictly downward, floor 30 km/h.
    """
    # Step 1: GHSL overrides binary land_use where available
    effective_lu = land_use
    if ghsl_settlement_class and ghsl_settlement_class in _GHSL_TO_LAND_USE:
        effective_lu = _GHSL_TO_LAND_USE[ghsl_settlement_class]

    key = (
        road_class_norm.lower() if pd.notna(road_class_norm) else "unknown",
        effective_lu.lower() if pd.notna(effective_lu) else "unknown",
    )

    # Step 2: base from road class × land use table
    if key in SAFE_SYSTEM_THRESHOLDS:
        base = float(SAFE_SYSTEM_THRESHOLDS[key])
    else:
        fallback = ("unknown", key[1])
        base = float(SAFE_SYSTEM_THRESHOLDS.get(fallback, SAFE_SYSTEM_THRESHOLDS[("unknown", "unknown")]))

    # Step 3: OSM-confirmed physical separation → no head-on risk → raise ceiling
    is_divided_confirmed = (
        (pd.notna(osm_oneway) and str(osm_oneway).lower() == "yes") or
        (pd.notna(osm_lanes) and osm_lanes >= 4)
    )
    if is_divided_confirmed and key[0] != "motorway":
        base = max(base, 100.0) if key[1] == "rural" else max(base, 80.0)

    # Step 4: unpaved surface → lower ceiling (loss-of-control risk)
    if pd.notna(osm_surface) and str(osm_surface).lower() in _UNPAVED_SURFACES:
        base = max(base - 10.0, 30.0)

    # Step 5: unlit road outside urban core → lower ceiling
    is_urban_core = effective_lu == "urban" and ghsl_settlement_class in (
        "urban_centre", "dense_urban", "semi_dense_urban", None
    )
    if pd.notna(osm_lit) and str(osm_lit).lower() == "no" and not is_urban_core:
        base = max(base - 5.0, 30.0)

    # Step 6: geometry adjustment — curved roads require lower design speed
    reduction = sinuosity_ss_adjustment(sinuosity if pd.notna(sinuosity) else 1.0)
    if reduction > 0:
        base = max(base - reduction, 30.0)

    return base


def add_safe_system_limits(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    has_osm      = "osm_oneway" in gdf.columns or "osm_lanes" in gdf.columns
    has_sinuosity= "sinuosity" in gdf.columns
    has_surface  = "osm_surface" in gdf.columns
    has_lit      = "osm_lit" in gdf.columns
    has_ghsl     = "ghsl_settlement_class" in gdf.columns

    gdf["ss_limit"] = gdf.apply(
        lambda r: get_safe_system_limit(
            r.get("road_class_norm", "unknown"),
            r.get("land_use", "unknown"),
            r.get("osm_oneway")            if has_osm       else None,
            r.get("osm_lanes")             if has_osm       else None,
            r.get("sinuosity")             if has_sinuosity else 1.0,
            r.get("osm_surface")           if has_surface   else None,
            r.get("osm_lit")               if has_lit       else None,
            r.get("ghsl_settlement_class") if has_ghsl      else None,
        ),
        axis=1,
    )
    # Flag which rows used an INTERPOLATED table cell — used by
    # score_speed_limit_alignment to dampen the sub-score for uncertain thresholds.
    gdf["ss_limit_interpolated"] = gdf.apply(
        lambda r: _is_threshold_interpolated(
            r.get("road_class_norm", "unknown"),
            r.get("land_use", "unknown"),
            r.get("ghsl_settlement_class") if has_ghsl else None,
        ),
        axis=1,
    )
    n_interp = gdf["ss_limit_interpolated"].sum()
    log.info(f"  SS limit: {len(gdf) - n_interp:,} VERIFIED cells, "
             f"{n_interp:,} INTERPOLATED cells (alignment dampened {ALIGNMENT_INTERPOLATED_DAMPENER:.0%})")

    if has_ghsl:
        n_ghsl = gdf["ghsl_settlement_class"].notna().sum()
        log.info(f"  SS limit uses GHSL settlement class for {n_ghsl:,} segments")
    if has_surface:
        n_unpaved = gdf["osm_surface"].isin(_UNPAVED_SURFACES).sum()
        if n_unpaved:
            log.info(f"  SS limit reduced for unpaved surface on {n_unpaved:,} segments")
    if has_sinuosity:
        n_adj = (gdf["sinuosity"] >= 1.20).sum()
        if n_adj:
            log.info(f"  SS limit adjusted for sinuosity on {n_adj:,} curved segments")
    return gdf


def score_speed_limit_alignment(
    posted: pd.Series,
    ss_limit: pd.Series,
    is_interpolated: pd.Series = None,
) -> pd.Series:
    """
    How misaligned is the posted limit vs Safe System standard?
    gap_pct = (posted - ss_limit) / ss_limit
    0 if gap<=0, 100 if gap>=50%, linear between.

    is_interpolated — boolean Series aligned to posted/ss_limit index.
      When True, the ss_limit came from an INTERPOLATED config cell (not a
      direct WHO citation), so the gap is multiplied by ALIGNMENT_INTERPOLATED_DAMPENER
      (0.70) to reflect threshold uncertainty.  A large gap still scores high;
      a borderline gap on an uncertain threshold no longer punishes as hard.

    Example: posted=80, ss=50 → gap=60% → score=100 (capped)
    Example: posted=80, ss=70 → gap=14% → score=28.6
    Example: posted=50, ss=40 (INTERPOLATED) → gap=25% → raw=50 → dampened=35
    """
    gap_pct = (posted - ss_limit) / ss_limit.replace(0, np.nan)
    raw = np.clip(gap_pct / 0.50, 0, 1).fillna(0) * 100

    if is_interpolated is not None:
        dampener = is_interpolated.map({True: ALIGNMENT_INTERPOLATED_DAMPENER, False: 1.0}).fillna(1.0)
        raw = raw * dampener

    return raw


def score_limit_credibility_gap(
    speed_85th: pd.Series,
    speed_limit: pd.Series,
    osm_lanes: pd.Series = None,
    osm_surface: pd.Series = None,
    ghsl_settlement_class: pd.Series = None,
    road_class_norm: pd.Series = None,
    land_use: pd.Series = None,
) -> pd.Series:
    """
    Evidence that the posted limit does not match how the road is used.

    REFRAMED: this is NOT "how much do drivers speed." It is evidence about
    whether the POSTED LIMIT is appropriate for the road. A large gap means
    the limit has likely lost credibility and needs review — not that
    enforcement needs to increase.

    ROAD-QUALITY DAMPENER: on well-built rural roads, a high F85 gap more
    plausibly indicates the posted limit is SET TOO LOW for the road's design
    rather than that drivers are recklessly speeding. Per the ADB FAQ:
    "concluding that a road needs a lower speed limit simply because F85 is
    high could be dangerous if the road design is meant for higher speeds."
    The dampener halves the penalty — it does NOT apply in urban/suburban or
    unpaved settings.

    v3.2 FIX: the original required osm_lanes >= 4 which almost never fires
    (OSM lane tags are sparse in India/Thailand). New primary condition:
    road_class_norm in {primary, trunk, motorway} + rural GHSL/land_use +
    not unpaved.  The 4+ lanes condition still fires as an alternative path.

    gap <= 10 km/h ("Credible")     → 0
    gap >= 20 km/h ("Non-Credible") → 100 (or 50 if road-quality dampener active)
    """
    gap  = (speed_85th - speed_limit).clip(lower=0)
    span = CREDIBILITY_GAP_NONCREDIBLE - CREDIBILITY_GAP_CREDIBLE
    raw  = np.clip((gap - CREDIBILITY_GAP_CREDIBLE) / span, 0, 1).fillna(0) * 100

    has_any_signal = any(
        x is not None for x in
        [osm_lanes, osm_surface, ghsl_settlement_class, road_class_norm, land_use]
    )
    if not has_any_signal:
        return raw

    rural_ghsl = {"low_density_rural", "very_low_density_rural"}
    major_classes = {"primary", "trunk", "motorway"}

    # ── Is the road confirmed rural? ─────────────────────────────────────────
    # Prefer GHSL (research-grade classification) over binary land_use field.
    if ghsl_settlement_class is not None:
        is_rural = ghsl_settlement_class.isin(rural_ghsl)
    elif land_use is not None:
        is_rural = land_use.fillna("").eq("rural")
    else:
        # No rural signal at all — don't apply dampener (conservative)
        return raw

    # ── Is the surface paved? ────────────────────────────────────────────────
    not_unpaved = pd.Series(True, index=raw.index)
    if osm_surface is not None:
        not_unpaved = ~osm_surface.fillna("").isin(_UNPAVED_SURFACES)

    # ── Is the road major class OR wide? ─────────────────────────────────────
    # Primary condition (v3.2): major road class — fires even without osm_lanes.
    is_major = pd.Series(False, index=raw.index)
    if road_class_norm is not None:
        is_major = road_class_norm.fillna("").isin(major_classes)

    # Alternative condition (original): 4+ lanes tagged in OSM.
    has_many_lanes = pd.Series(False, index=raw.index)
    if osm_lanes is not None:
        has_many_lanes = osm_lanes.fillna(0) >= 4

    is_wide_or_major = is_major | has_many_lanes

    # Dampener: rural + paved + (major class or wide)
    is_hq_rural = is_rural & not_unpaved & is_wide_or_major
    raw = raw * np.where(is_hq_rural, 0.5, 1.0)

    return raw


def score_vru_context_risk(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    VRU exposure: settlement context + road class + urban density + helmet SPI.

    GHSL UPGRADE: when ghsl_settlement_class is present, the binary
    urban=80/rural=35 base score is replaced with a 7-level score that
    distinguishes urban centres (80) from rural clusters (50) from truly
    isolated rural roads (28). This directly fixes the FAQ-flagged limitation
    that "LandUse may not reflect recent urban development."

    OSM HIGHWAY TAG: residential/living_street roads get a pedestrian-mixing
    boost (+10) regardless of settlement class — these road types legally share
    the carriageway with pedestrians and cyclists by definition.

    Helmet multiplier kept: crash lethality at any speed is higher without
    a helmet. Net effect on final SSS: ~2-4 pts — small but directionally correct.
    """
    lu   = gdf.get("land_use",        pd.Series([np.nan]*len(gdf), index=gdf.index)).fillna("unknown")
    rc   = gdf.get("road_class_norm", pd.Series([np.nan]*len(gdf), index=gdf.index)).fillna("unknown")
    up   = gdf.get("urban_pct",       pd.Series([np.nan]*len(gdf), index=gdf.index))
    cc   = gdf.get("country_code",    pd.Series([np.nan]*len(gdf), index=gdf.index)).fillna("unknown")
    ghsl = gdf.get("ghsl_settlement_class", None)
    hw   = gdf.get("osm_highway",     pd.Series([""] * len(gdf), index=gdf.index)).fillna("")

    # Land-use base score: GHSL 7-level if available, binary fallback otherwise
    if ghsl is not None and ghsl.notna().any():
        lu_score = ghsl.map(_GHSL_VRU_BASE).fillna(
            lu.map({"urban": 80, "rural": 35, "unknown": 50}).fillna(50)
        )
    else:
        lu_score = lu.map({"urban": 80, "rural": 35, "unknown": 50}).fillna(50)

    rc_score = rc.map(VRU_RC_SCORE_MAP).fillna(50)
    base = 0.60 * lu_score + 0.40 * rc_score

    # OSM highway tag: residential/living_street → pedestrian-mixing boost
    pedestrian_road = hw.isin({"residential", "living_street", "unclassified"})
    base = (base + pedestrian_road.astype(float) * 10).clip(0, 100)

    # OSM lighting: unlit roads outside urban centres → higher pedestrian casualty
    # risk at night. iRAP star-rating explicitly includes lighting as a VRU factor;
    # NHTSA data: ~25% of pedestrian fatalities on inadequately lit roads.
    # Only applied outside urban_centre/dense_urban where street lighting is assumed.
    osm_lit_col = gdf.get("osm_lit", pd.Series([""] * len(gdf), index=gdf.index)).fillna("")
    is_urban_core = (ghsl.isin({"urban_centre", "dense_urban"}) if ghsl is not None and ghsl.notna().any()
                     else lu == "urban")
    unlit_penalty = osm_lit_col.eq("no") & ~is_urban_core
    base = (base + unlit_penalty.astype(float) * 8).clip(0, 100)

    if up.notna().any():
        up_norm = up.clip(0, 100) / 100
        base = (base * (1 + 0.20 * up_norm.fillna(0.5))).clip(0, 100)

    def _helmet_mult(row_cc, row_lu):
        spi = HELMET_SPI.get((row_cc, row_lu), HELMET_SPI.get((row_cc, "unknown"), 0.75))
        return 1.0 + (1.0 - spi) * HELMET_SEVERITY_WEIGHT

    helmet = pd.Series(
        [_helmet_mult(c, l) for c, l in zip(cc, lu)],
        index=gdf.index,
    )
    return (base * helmet).clip(0, 100)


def compute_confidence_weight(sample_size: pd.Series) -> pd.Series:
    s = sample_size.fillna(0)
    return pd.Series(
        np.where(
            s >= 30, 1.00,
            np.where(
                s >= MIN_SAMPLE_SIZE,
                LOW_SAMPLE_PENALTY + (1.0 - LOW_SAMPLE_PENALTY)
                * (s - MIN_SAMPLE_SIZE) / (30 - MIN_SAMPLE_SIZE),
                LOW_SAMPLE_PENALTY,
            ),
        ),
        index=sample_size.index,
    )


def compute_speed_safety_score(
    gdf: gpd.GeoDataFrame,
    weights: dict = None,
) -> gpd.GeoDataFrame:
    """
    Compute SSS for all scoreable segments.
    Three components: speed_limit_alignment, limit_credibility_gap, vru_context_risk.
    Compliance excluded (unreliable field — see module docstring).
    """
    if weights is None:
        weights = WEIGHTS

    gdf  = gdf.copy()
    mask = gdf["scoreable"]

    gdf.loc[mask, "sub_score_limit_alignment"] = score_speed_limit_alignment(
        gdf.loc[mask, "speed_limit"],
        gdf.loc[mask, "ss_limit"],
        is_interpolated = gdf.loc[mask, "ss_limit_interpolated"] if "ss_limit_interpolated" in gdf.columns else None,
    )
    gdf.loc[mask, "sub_score_limit_credibility"] = score_limit_credibility_gap(
        gdf.loc[mask, "speed_85th"],
        gdf.loc[mask, "speed_limit"],
        osm_lanes             = gdf.loc[mask, "osm_lanes"]             if "osm_lanes"             in gdf.columns else None,
        osm_surface           = gdf.loc[mask, "osm_surface"]           if "osm_surface"           in gdf.columns else None,
        ghsl_settlement_class = gdf.loc[mask, "ghsl_settlement_class"] if "ghsl_settlement_class" in gdf.columns else None,
        road_class_norm       = gdf.loc[mask, "road_class_norm"]       if "road_class_norm"       in gdf.columns else None,
        land_use              = gdf.loc[mask, "land_use"]               if "land_use"               in gdf.columns else None,
    )
    gdf.loc[mask, "sub_score_vru_risk"] = score_vru_context_risk(gdf[mask])

    # Store compliance as display-only field (not scored)
    if "pct_over_limit" in gdf.columns:
        gdf.loc[mask, "sub_score_compliance"] = (
            np.sqrt(gdf.loc[mask, "pct_over_limit"].clip(0, 100) / 100) * 100
        )

    gdf.loc[mask, "confidence_weight"] = compute_confidence_weight(
        gdf.loc[mask, "sample_size"]
    )

    w = weights
    total_w = w["speed_limit_alignment"] + w["limit_credibility_gap"] + w["vru_context_risk"]

    gdf.loc[mask, "sss_raw"] = (
        w["speed_limit_alignment"] * gdf.loc[mask, "sub_score_limit_alignment"]
        + w["limit_credibility_gap"]  * gdf.loc[mask, "sub_score_limit_credibility"]
        + w["vru_context_risk"]     * gdf.loc[mask, "sub_score_vru_risk"]
    ) / total_w

    gdf.loc[mask, "sss"] = (
        gdf.loc[mask, "sss_raw"] * gdf.loc[mask, "confidence_weight"]
    ).clip(0, 100)

    gdf.loc[mask, "sss_band"] = gdf.loc[mask].apply(
        lambda r: classify_band(r["sss"], r.get("sub_score_limit_credibility")),
        axis=1,
    )
    gdf["low_data_flag"] = (
        gdf["sample_size"].fillna(0) < MIN_SAMPLE_SIZE
    ) & gdf["scoreable"]
    gdf.loc[mask, "sss_recommendation"] = gdf.loc[mask].apply(
        _generate_recommendation, axis=1
    )

    log.info("\nSSS computed. Score distribution:")
    log.info(gdf.loc[mask, "sss"].describe().round(1).to_string())
    log.info("\nBand distribution:")
    log.info(gdf.loc[mask, "sss_band"].value_counts().to_string())
    return gdf


def compute_alignment_only_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    TIER 1 score — "does the posted limit match the Safe System standard
    for this road class/land use." Needs ONLY speed_limit + ss_limit, no
    F85/median/sample-size data at all, so it covers every segment with
    `alignment_scoreable` True (a strict superset of the full-SSS
    `scoreable` mask — see preprocessing.get_analysis_subset).

    This is the most directly brief-aligned single number in the whole
    pipeline ("is the current speed limit itself appropriate for the
    road") and the most "scalable and replicable across countries" one,
    since it doesn't depend on GPS-probe behavioural data most countries
    won't have. Reported alongside, not instead of, the full Tier 2 SSS.
    """
    gdf  = gdf.copy()
    mask = gdf["alignment_scoreable"]

    gdf.loc[mask, "alignment_only_score"] = score_speed_limit_alignment(
        gdf.loc[mask, "speed_limit"],
        gdf.loc[mask, "ss_limit"],
        is_interpolated = gdf.loc[mask, "ss_limit_interpolated"] if "ss_limit_interpolated" in gdf.columns else None,
    )
    gdf.loc[mask, "alignment_only_band"] = gdf.loc[mask, "alignment_only_score"].apply(classify_band)
    return gdf


def classify_band(score: float, credibility_sub_score: float = None) -> str:
    """
    Assign a band label to a Speed Safety Score.

    CRITICAL GATE (v3.2): a segment can only reach Critical if the
    credibility sub-score is >= 25, meaning there is at least a ~15 km/h
    behavioural gap confirming the limit is not working.  When the limit is
    wrong (high alignment) but drivers respect it (low credibility gap), the
    road is still High Risk — urgent, but not the same as one where both the
    limit AND the behaviour are dangerous.

    credibility_sub_score is optional so classify_band remains usable for
    alignment_only_score (Tier 1) where no behavioural data exists.
    """
    if pd.isna(score):
        return "No Data"
    for band, (lo, hi) in SCORE_BANDS.items():
        if lo <= score < hi:
            if band == "Critical" and credibility_sub_score is not None:
                if credibility_sub_score < 25:
                    return "High Risk"
            return band
    # score == 100 or above upper boundary
    final = "Critical" if score >= max(lo for lo, _ in SCORE_BANDS.values()) else "Acceptable"
    if final == "Critical" and credibility_sub_score is not None and credibility_sub_score < 25:
        return "High Risk"
    return final


def _generate_recommendation(row: pd.Series) -> str:
    posted = row.get("speed_limit", np.nan)
    ss     = row.get("ss_limit", np.nan)
    f85    = row.get("speed_85th", np.nan)
    band   = row.get("sss_band", "")
    lu     = row.get("land_use", "")
    rc     = row.get("road_class_norm", "")
    cc     = row.get("country_code", "")
    parts  = []

    if pd.notna(posted) and pd.notna(ss):
        if posted > ss + 5:
            parts.append(
                f"Posted limit ({posted:.0f} km/h) exceeds Safe System standard "
                f"({ss:.0f} km/h) for this {lu} {rc} road — "
                f"recommend reducing to {ss:.0f} km/h."
            )
        elif posted < ss - 5:
            parts.append(
                f"Posted limit ({posted:.0f} km/h) is below Safe System "
                f"standard ({ss:.0f} km/h) — limit may be overly restrictive."
            )
        else:
            parts.append(
                f"Posted limit ({posted:.0f} km/h) aligns with Safe System "
                f"standard ({ss:.0f} km/h)."
            )

    if pd.notna(f85) and pd.notna(posted) and f85 > posted * 1.10:
        parts.append(
            f"85th percentile speed ({f85:.0f} km/h) significantly exceeds "
            f"posted limit — enforcement or physical traffic calming needed."
        )

    if band in ("Critical", "High Risk"):
        parts.append("Priority segment: recommend immediate site review.")

    # Contextual helmet note (NOT a scoring driver — shown for completeness)
    if cc == "MH":
        parts.append(
            "Context: Maharashtra helmet wearing rate ~21% (SPI=0.209). "
            "Speed interventions should be paired with helmet enforcement."
        )
    elif cc == "TH" and lu == "rural":
        parts.append(
            "Context: Thailand rural helmet wearing rate ~67% — "
            "pair speed intervention with helmet campaign."
        )

    return " ".join(parts) if parts else "No specific action flagged."

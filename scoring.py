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

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    SAFE_SYSTEM_THRESHOLDS, SCORE_BANDS,
    MIN_SAMPLE_SIZE, LOW_SAMPLE_PENALTY,
    CREDIBILITY_GAP_CREDIBLE, CREDIBILITY_GAP_NONCREDIBLE,
    HELMET_SPI, HELMET_SEVERITY_WEIGHT,
    VRU_RC_SCORE_MAP,
)

# Weights — compliance removed, weight redistributed
WEIGHTS = {
    "speed_limit_alignment": 0.38,
    "limit_credibility_gap": 0.30,
    "vru_context_risk":      0.32,
}


def get_safe_system_limit(road_class_norm: str, land_use: str,
                           osm_oneway: str = None, osm_lanes: float = None) -> float:
    """
    Safe System speed ceiling for this road.

    OSM-EVIDENCE OVERRIDE (new): if enrichment.match_road_infrastructure()
    found a real OSM way tag confirming physical separation (oneway=yes,
    or 4+ lanes suggesting a dual carriageway), the road's crash-type
    context shifts toward "fully separated" (the real WHO 100 km/h tier)
    regardless of the road-class assumption — head-on collisions aren't
    possible on a genuinely separated carriageway. This replaces an
    ASSUMPTION (road class implies divided/undivided) with an OBSERVED
    fact where one is available, falling back to the assumption otherwise.
    """
    key = (
        road_class_norm.lower() if pd.notna(road_class_norm) else "unknown",
        land_use.lower() if pd.notna(land_use) else "unknown",
    )
    if key in SAFE_SYSTEM_THRESHOLDS:
        base = float(SAFE_SYSTEM_THRESHOLDS[key])
    else:
        fallback = ("unknown", key[1])
        base = float(SAFE_SYSTEM_THRESHOLDS.get(fallback, SAFE_SYSTEM_THRESHOLDS[("unknown", "unknown")]))

    # OSM-confirmed physical separation → no head-on risk → allow up to the
    # "fully separated" tier (100), but never LOWER than the road-class
    # assumption already gave (this is a one-directional safety-relevant
    # upgrade, not a general re-assignment).
    is_divided_confirmed = (
        (pd.notna(osm_oneway) and str(osm_oneway).lower() == "yes") or
        (pd.notna(osm_lanes) and osm_lanes >= 4)
    )
    if is_divided_confirmed and key[0] != "motorway":
        base = max(base, 100.0) if key[1] == "rural" else max(base, 80.0)
    return base


def add_safe_system_limits(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    has_osm = "osm_oneway" in gdf.columns or "osm_lanes" in gdf.columns
    gdf["ss_limit"] = gdf.apply(
        lambda r: get_safe_system_limit(
            r.get("road_class_norm", "unknown"),
            r.get("land_use", "unknown"),
            r.get("osm_oneway") if has_osm else None,
            r.get("osm_lanes")  if has_osm else None,
        ),
        axis=1,
    )
    return gdf


def score_speed_limit_alignment(posted: pd.Series, ss_limit: pd.Series) -> pd.Series:
    """
    How misaligned is the posted limit vs Safe System standard?
    gap_pct = (posted - ss_limit) / ss_limit
    0 if gap<=0, 100 if gap>=50%, linear between.

    Example: posted=80, ss=50 → gap=60% → score=100 (capped)
    Example: posted=80, ss=70 → gap=14% → score=28.6
    """
    gap_pct = (posted - ss_limit) / ss_limit.replace(0, np.nan)
    return np.clip(gap_pct / 0.50, 0, 1).fillna(0) * 100


def score_limit_credibility_gap(speed_85th: pd.Series, speed_limit: pd.Series) -> pd.Series:
    """
    REFRAMED (was score_operating_speed_gap). This is NOT a measure of
    "how much drivers speed" — it is evidence about whether the POSTED
    LIMIT matches how the road is actually used. A large gap means the
    limit has likely lost credibility and needs review, not that
    enforcement needs to increase.

    Uses the SAME absolute km/h gap and thresholds as
    advanced_scoring.credibility() (10/20 km/h — CREDIBILITY_GAP_CREDIBLE /
    CREDIBILITY_GAP_NONCREDIBLE in config.py), so this sub-score and that
    module's classification report one consistent number instead of two
    different formulas for nearly the same underlying evidence.

    gap <= 10 km/h ("Credible")     → 0
    gap >= 20 km/h ("Non-Credible") → 100
    Under-speed (negative gap) is NOT penalised here — it has a different
    likely cause (poor road condition, heavy trucks) and a different
    intervention; see advanced_scoring.credibility_class for that case.

    Example: F85=103 on 80km/h posted → 23 km/h gap → score=100 (capped)
    Example: F85=85 on 80km/h posted  → 5 km/h gap  → score=0
    """
    gap = (speed_85th - speed_limit).clip(lower=0)
    span = CREDIBILITY_GAP_NONCREDIBLE - CREDIBILITY_GAP_CREDIBLE
    return np.clip((gap - CREDIBILITY_GAP_CREDIBLE) / span, 0, 1).fillna(0) * 100


def score_vru_context_risk(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    VRU exposure: land use + road class + urban density + helmet SPI.

    Helmet multiplier kept: crash lethality at any speed is higher without
    a helmet. MH (SPI=0.148) gets ~31% VRU amplification vs TH (SPI=0.672)
    at ~9%. Net effect on final SSS: ~2-4 pts — small but directionally correct.

    Rural base = 35 (raised from 30): undivided rural highways in Asia carry
    significant pedestrian and PTW traffic with no separation.
    """
    lu = gdf.get("land_use", pd.Series([np.nan] * len(gdf), index=gdf.index)).fillna("unknown")
    rc = gdf.get("road_class_norm", pd.Series([np.nan] * len(gdf), index=gdf.index)).fillna("unknown")
    up = gdf.get("urban_pct", pd.Series([np.nan] * len(gdf), index=gdf.index))
    cc = gdf.get("country_code", pd.Series([np.nan] * len(gdf), index=gdf.index)).fillna("unknown")

    lu_score_map = {"urban": 80, "rural": 35, "unknown": 50}

    lu_score = lu.map(lu_score_map).fillna(50)
    rc_score = rc.map(VRU_RC_SCORE_MAP).fillna(50)
    base = 0.60 * lu_score + 0.40 * rc_score

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
        gdf.loc[mask, "speed_limit"], gdf.loc[mask, "ss_limit"]
    )
    gdf.loc[mask, "sub_score_limit_credibility"] = score_limit_credibility_gap(
        gdf.loc[mask, "speed_85th"], gdf.loc[mask, "speed_limit"]
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

    gdf.loc[mask, "sss_band"] = gdf.loc[mask, "sss"].apply(_classify_band)
    gdf["low_data_flag"] = (
        gdf["sample_size"].fillna(0) < MIN_SAMPLE_SIZE
    ) & gdf["scoreable"]
    gdf.loc[mask, "sss_recommendation"] = gdf.loc[mask].apply(
        _generate_recommendation, axis=1
    )

    print("\nSSS computed. Score distribution:")
    print(gdf.loc[mask, "sss"].describe().round(1))
    print("\nBand distribution:")
    print(gdf.loc[mask, "sss_band"].value_counts())
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
        gdf.loc[mask, "speed_limit"], gdf.loc[mask, "ss_limit"]
    )
    gdf.loc[mask, "alignment_only_band"] = gdf.loc[mask, "alignment_only_score"].apply(_classify_band)
    return gdf


def _classify_band(score: float) -> str:
    if pd.isna(score):
        return "No Data"
    for band, (lo, hi) in SCORE_BANDS.items():
        if lo <= score < hi:
            return band
    return "Critical" if score >= max(lo for lo, _ in SCORE_BANDS.values()) else "Acceptable"


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

"""
scoring.py — Compute Speed Safety Score (0–100) for every road segment.
v2.1: Recalibrated against real data (mean=32, max=75, not synthetic mean=55).

CHANGELOG v2.1:
  - score_vru_context_risk: uses VRU_RC_SCORE_MAP from config (rural scores raised)
  - score_operating_speed_gap: SPEED_GAP_ZERO 5%→2%, SPEED_GAP_CRITICAL 30%→20%
  - SCORE_BANDS: Critical 52, High Risk 40, Moderate 27 (real-data calibrated)
  - helmet SPI severity multiplier retained from v2.0
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    SAFE_SYSTEM_THRESHOLDS, WEIGHTS, SCORE_BANDS,
    MIN_SAMPLE_SIZE, LOW_SAMPLE_PENALTY,
    SPEED_GAP_CRITICAL, SPEED_GAP_ZERO,
    HELMET_SPI, HELMET_SEVERITY_WEIGHT,
    VRU_RC_SCORE_MAP,
)


def get_safe_system_limit(road_class_norm: str, land_use: str) -> float:
    key = (road_class_norm.lower() if pd.notna(road_class_norm) else "unknown",
           land_use.lower()        if pd.notna(land_use)        else "unknown")
    if key in SAFE_SYSTEM_THRESHOLDS:
        return float(SAFE_SYSTEM_THRESHOLDS[key])
    fallback = ("unknown", key[1])
    if fallback in SAFE_SYSTEM_THRESHOLDS:
        return float(SAFE_SYSTEM_THRESHOLDS[fallback])
    return float(SAFE_SYSTEM_THRESHOLDS[("unknown", "unknown")])


def add_safe_system_limits(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["ss_limit"] = gdf.apply(
        lambda r: get_safe_system_limit(
            r.get("road_class_norm", "unknown"),
            r.get("land_use", "unknown")
        ), axis=1
    )
    return gdf


def score_speed_limit_alignment(posted: pd.Series, ss_limit: pd.Series) -> pd.Series:
    """
    How misaligned is the posted limit vs Safe System standard?
    gap_pct = (posted - ss_limit) / ss_limit
    Score: 0 if gap<=0, 100 if gap>=50%, linear between.
    """
    gap_pct = (posted - ss_limit) / ss_limit.replace(0, np.nan)
    score = np.clip(gap_pct / 0.50, 0, 1) * 100
    return score.fillna(0)


def score_operating_speed_gap(speed_85th: pd.Series, speed_limit: pd.Series) -> pd.Series:
    """
    How much do drivers actually exceed the posted limit?
    v2.1: SPEED_GAP_ZERO=2%, SPEED_GAP_CRITICAL=20%
    Example: F85th=90 on 80km/h → 12.5% over → score = (12.5-2)/(20-2)*100 = 58
    """
    gap_pct = (speed_85th - speed_limit) / speed_limit.replace(0, np.nan)
    score = np.clip(
        (gap_pct - SPEED_GAP_ZERO) / (SPEED_GAP_CRITICAL - SPEED_GAP_ZERO),
        0, 1
    ) * 100
    return score.fillna(0)


def score_vru_context_risk(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    VRU exposure from land use, road class, urban density, and helmet SPI.
    v2.1: VRU_RC_SCORE_MAP imported from config (rural scores raised).
    Rural undivided highways carry significant PTW/pedestrian traffic.
    """
    lu = gdf.get("land_use", pd.Series(["unknown"] * len(gdf), index=gdf.index))
    rc = gdf.get("road_class_norm", pd.Series(["unknown"] * len(gdf), index=gdf.index))
    up = gdf.get("urban_pct", pd.Series([np.nan] * len(gdf), index=gdf.index))
    cc = gdf.get("country_code", pd.Series(["unknown"] * len(gdf), index=gdf.index))

    lu_score_map = {"urban": 80, "rural": 35, "unknown": 50}  # rural raised: 30→45

    lu_score = lu.map(lu_score_map).fillna(55)
    rc_score = rc.map(VRU_RC_SCORE_MAP).fillna(50)

    # Blend 60/40
    base_score = 0.60 * lu_score + 0.40 * rc_score

    # Urban density boost
    if up.notna().any():
        up_norm = up.clip(0, 100) / 100
        base_score = base_score * (1 + 0.20 * up_norm.fillna(0.5))
        base_score = base_score.clip(0, 100)

    # Helmet SPI severity multiplier
    def _helmet_multiplier(row_cc, row_lu):
        spi = HELMET_SPI.get((row_cc, row_lu),
              HELMET_SPI.get((row_cc, "unknown"), 0.75))
        return 1.0 + (1.0 - spi) * HELMET_SEVERITY_WEIGHT

    helmet_mult = pd.Series(
        [_helmet_multiplier(c, l) for c, l in zip(cc, lu)],
        index=gdf.index
    )
    base_score = (base_score * helmet_mult).clip(0, 100)

    return base_score


def score_compliance_rate(pct_over_limit: pd.Series) -> pd.Series:
    """% vehicles exceeding limit. sqrt nonlinearity: 25% → ~50 score."""
    p = pct_over_limit.clip(0, 100)
    score = np.sqrt(p / 100) * 100
    return score.fillna(0)


def compute_confidence_weight(sample_size: pd.Series) -> pd.Series:
    s = sample_size.fillna(0)
    weight = np.where(
        s >= 30, 1.00,
        np.where(
            s >= MIN_SAMPLE_SIZE,
            LOW_SAMPLE_PENALTY + (1.0 - LOW_SAMPLE_PENALTY) * (s - MIN_SAMPLE_SIZE) / (30 - MIN_SAMPLE_SIZE),
            LOW_SAMPLE_PENALTY
        )
    )
    return pd.Series(weight, index=sample_size.index)


def compute_speed_safety_score(
    gdf: gpd.GeoDataFrame,
    weights: dict = None,
) -> gpd.GeoDataFrame:
    if weights is None:
        weights = WEIGHTS

    gdf = gdf.copy()
    mask = gdf["scoreable"]

    gdf.loc[mask, "sub_score_limit_alignment"] = score_speed_limit_alignment(
        gdf.loc[mask, "speed_limit"], gdf.loc[mask, "ss_limit"],
    )
    gdf.loc[mask, "sub_score_op_speed_gap"] = score_operating_speed_gap(
        gdf.loc[mask, "speed_85th"], gdf.loc[mask, "speed_limit"],
    )
    gdf.loc[mask, "sub_score_vru_risk"] = score_vru_context_risk(gdf[mask])
    gdf.loc[mask, "sub_score_compliance"] = score_compliance_rate(
        gdf.loc[mask, "pct_over_limit"]
    )
    gdf.loc[mask, "confidence_weight"] = compute_confidence_weight(
        gdf.loc[mask, "sample_size"]
    )

    w = weights
    total_w = (w["speed_limit_alignment"] + w["operating_speed_gap"] +
               w["vru_context_risk"] + w["compliance_rate"])

    gdf.loc[mask, "sss_raw"] = (
        w["speed_limit_alignment"] * gdf.loc[mask, "sub_score_limit_alignment"] +
        w["operating_speed_gap"]   * gdf.loc[mask, "sub_score_op_speed_gap"]   +
        w["vru_context_risk"]      * gdf.loc[mask, "sub_score_vru_risk"]        +
        w["compliance_rate"]       * gdf.loc[mask, "sub_score_compliance"]
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


def _classify_band(score: float) -> str:
    if pd.isna(score):
        return "No Data"
    for band, (lo, hi) in SCORE_BANDS.items():
        if lo <= score < hi:
            return band
    return "Critical" if score >= 52 else "Acceptable"


def _generate_recommendation(row: pd.Series) -> str:
    posted = row.get("speed_limit", np.nan)
    ss     = row.get("ss_limit", np.nan)
    f85    = row.get("speed_85th", np.nan)
    band   = row.get("sss_band", "")
    lu     = row.get("land_use", "")
    rc     = row.get("road_class_norm", "")
    cc     = row.get("country_code", "")

    parts = []

    if pd.notna(posted) and pd.notna(ss):
        if posted > ss + 5:
            parts.append(
                f"Posted limit ({posted:.0f} km/h) exceeds Safe System "
                f"standard ({ss:.0f} km/h) for this {lu} {rc} road — "
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

    if cc == "MH":
        parts.append(
            "Note: Maharashtra helmet wearing rate is low (~21%) — "
            "speed intervention should be paired with helmet enforcement."
        )
    elif cc == "TH" and lu == "rural":
        parts.append(
            "Note: Thailand rural helmet wearing rate (~67%) — "
            "pair speed intervention with helmet campaign."
        )

    if band in ("Critical", "High Risk"):
        parts.append("Priority segment: recommend immediate site review.")

    return " ".join(parts) if parts else "No specific action flagged."

"""
scoring.py — Compute Speed Safety Score (0–100) for every road segment.
v2.0: Updated thresholds, weights, and helmet SPI integration.

Five sub-scores, each normalized 0–100:
  1. speed_limit_alignment  — Is the posted limit Safe-System-appropriate?
  2. operating_speed_gap    — How much do drivers exceed the limit (85th pct)?
  3. vru_context_risk       — How exposed are VRUs? (now includes helmet SPI)
  4. compliance_rate        — What % of vehicles break the limit?
  5. data_confidence        — Applied as a multiplier, not additive.

Final SSS = weighted sum, adjusted for data confidence.

CHANGELOG v2.0:
  - score_operating_speed_gap: SPEED_GAP_ZERO 0%→5%, SPEED_GAP_CRITICAL 20%→30%
  - score_vru_context_risk: helmet SPI severity multiplier integrated
  - WEIGHTS: compliance_rate 0.15→0.20; vru_context_risk 0.25→0.27;
             operating_speed_gap 0.25→0.23
  - SCORE_BANDS: Critical 65→78, High Risk 48→62, Moderate 30→40, Acceptable 0→40
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    SAFE_SYSTEM_THRESHOLDS, WEIGHTS, SCORE_BANDS,
    MIN_SAMPLE_SIZE, LOW_SAMPLE_PENALTY,
    SPEED_GAP_CRITICAL, SPEED_GAP_ZERO,
    HELMET_SPI, HELMET_SEVERITY_WEIGHT,
)


# ─── 1. Safe System threshold lookup ─────────────────────────────────────────

def get_safe_system_limit(road_class_norm: str, land_use: str) -> float:
    """Return Safe System speed threshold (km/h) for a road type."""
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


# ─── 2. Sub-score: Speed Limit Alignment ─────────────────────────────────────

def score_speed_limit_alignment(posted: pd.Series, ss_limit: pd.Series) -> pd.Series:
    """
    How misaligned is the posted limit vs Safe System standard?

    gap_pct = (posted - ss_limit) / ss_limit
    Score:
      gap_pct ≤ 0    → 0   (limit is at or below SS standard)
      gap_pct ≥ 0.5  → 100 (50%+ over SS standard → critical)
      linear in between

    Example: posted=80, ss=50 → gap=60% → score=min(60/50, 1)*100=100
    Example: posted=60, ss=50 → gap=20% → score=40
    """
    gap_pct = (posted - ss_limit) / ss_limit.replace(0, np.nan)
    score = np.clip(gap_pct / 0.50, 0, 1) * 100
    return score.fillna(0)


# ─── 3. Sub-score: Operating Speed Gap ────────────────────────────────────────

def score_operating_speed_gap(speed_85th: pd.Series, speed_limit: pd.Series) -> pd.Series:
    """
    How much do drivers actually exceed the posted limit?

    REVISED v2.0:
      SPEED_GAP_ZERO: 0% → 5%  (normal GPS probe noise floor)
      SPEED_GAP_CRITICAL: 20% → 30%  (credibility collapse threshold)

    gap_pct = (F85th - posted_limit) / posted_limit
    Score:
      gap_pct ≤ 5%   → 0   (within normal measurement/compliance band)
      gap_pct ≥ 30%  → 100 (limit is effectively ignored)
      linear in between

    Example from Data Guide: 97 km/h on 90 km/h limit = 7.8% → score ≈ 10
    Example: 130 km/h on 90 km/h limit = 44% → score = 100
    """
    gap_pct = (speed_85th - speed_limit) / speed_limit.replace(0, np.nan)
    score = np.clip(
        (gap_pct - SPEED_GAP_ZERO) / (SPEED_GAP_CRITICAL - SPEED_GAP_ZERO),
        0, 1
    ) * 100
    return score.fillna(0)


# ─── 4. Sub-score: VRU Context Risk ──────────────────────────────────────────

def score_vru_context_risk(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    VRU exposure from land use, road class, urban density, and helmet SPI.

    REVISED v2.0: Helmet SPI severity multiplier integrated.
    
    Low helmet wearing rates dramatically increase the lethality of any crash
    at speed. Maharashtra (SPI=0.209) vs Thailand (SPI=0.778) creates a
    country-specific severity amplifier: same road, same speed → worse outcome
    in Maharashtra because riders are ~4× less protected.

    Base score (land use + road class) is multiplied by:
      helmet_multiplier = 1 + (1 - SPI) * HELMET_SEVERITY_WEIGHT

    For Maharashtra combined (SPI=0.209): multiplier = 1 + 0.791*0.40 = 1.316
    For Thailand combined (SPI=0.778):    multiplier = 1 + 0.222*0.40 = 1.089
    """
    lu = gdf.get("land_use", pd.Series(["unknown"] * len(gdf), index=gdf.index))
    rc = gdf.get("road_class_norm", pd.Series(["unknown"] * len(gdf), index=gdf.index))
    up = gdf.get("urban_pct", pd.Series([np.nan] * len(gdf), index=gdf.index))
    cc = gdf.get("country_code", pd.Series(["unknown"] * len(gdf), index=gdf.index))

    lu_score_map = {"urban": 80, "rural": 30, "unknown": 50}
    rc_score_map = {
        "local":       80,
        "residential": 80,
        "tertiary":    65,
        "secondary":   45,
        "primary":     25,
        "trunk":       15,
        "motorway":    10,
        "unknown":     50,
    }

    lu_score = lu.map(lu_score_map).fillna(50)
    rc_score = rc.map(rc_score_map).fillna(50)

    # Blend land use and road class 60/40
    base_score = 0.60 * lu_score + 0.40 * rc_score

    # Boost by urban_pct if available
    if up.notna().any():
        up_norm = up.clip(0, 100) / 100
        base_score = base_score * (1 + 0.20 * up_norm.fillna(0.5))
        base_score = base_score.clip(0, 100)

    # ── Helmet SPI severity multiplier (NEW v2.0) ─────────────────────────
    # Look up SPI by (country_code, land_use), fall back to (country_code, unknown)
    def _helmet_multiplier(row_cc, row_lu):
        spi = HELMET_SPI.get((row_cc, row_lu),
              HELMET_SPI.get((row_cc, "unknown"), 0.75))  # default: 75% if unknown
        return 1.0 + (1.0 - spi) * HELMET_SEVERITY_WEIGHT

    helmet_mult = pd.Series(
        [_helmet_multiplier(c, l) for c, l in zip(cc, lu)],
        index=gdf.index
    )
    base_score = (base_score * helmet_mult).clip(0, 100)

    return base_score


# ─── 5. Sub-score: Compliance Rate ───────────────────────────────────────────

def score_compliance_rate(pct_over_limit: pd.Series) -> pd.Series:
    """
    % of vehicles exceeding the posted limit → how unenforced / misaligned is it?

    pct_over_limit is 0–100.
    Mild nonlinearity (sqrt) so 25% gets ~50 score (not just 25).
    High compliance failure = strong evidence limit is set wrong.
    """
    p = pct_over_limit.clip(0, 100)
    score = np.sqrt(p / 100) * 100
    return score.fillna(0)


# ─── 6. Data Confidence Weight ────────────────────────────────────────────────

def compute_confidence_weight(sample_size: pd.Series) -> pd.Series:
    """
    Confidence multiplier based on sample size.

    sample ≥ 30 → 1.00 (full confidence)
    5 ≤ sample < 30 → linear scale 0.75–1.00
    sample < 5 → 0.75 (low data penalty)
    """
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


# ─── 7. Master scoring function ───────────────────────────────────────────────

def compute_speed_safety_score(
    gdf: gpd.GeoDataFrame,
    weights: dict = None,
) -> gpd.GeoDataFrame:
    """
    Compute SSS for all scoreable segments.

    Adds columns:
      sub_score_*        — individual sub-scores (0–100)
      confidence_weight  — data quality multiplier
      sss_raw            — weighted sum before confidence adjustment
      sss                — final Speed Safety Score (0–100)
      sss_band           — Critical / High Risk / Moderate / Acceptable
      sss_recommendation — plain-English policy recommendation
    """
    if weights is None:
        weights = WEIGHTS

    gdf = gdf.copy()
    mask = gdf["scoreable"]

    # Sub-score 1: Speed Limit Alignment
    gdf.loc[mask, "sub_score_limit_alignment"] = score_speed_limit_alignment(
        gdf.loc[mask, "speed_limit"],
        gdf.loc[mask, "ss_limit"],
    )

    # Sub-score 2: Operating Speed Gap
    gdf.loc[mask, "sub_score_op_speed_gap"] = score_operating_speed_gap(
        gdf.loc[mask, "speed_85th"],
        gdf.loc[mask, "speed_limit"],
    )

    # Sub-score 3: VRU Context Risk (with helmet SPI)
    gdf.loc[mask, "sub_score_vru_risk"] = score_vru_context_risk(gdf[mask])

    # Sub-score 4: Compliance Rate
    gdf.loc[mask, "sub_score_compliance"] = score_compliance_rate(
        gdf.loc[mask, "pct_over_limit"]
    )

    # Confidence weight
    gdf.loc[mask, "confidence_weight"] = compute_confidence_weight(
        gdf.loc[mask, "sample_size"]
    )

    # Weighted sum
    w = weights
    total_w = (
        w["speed_limit_alignment"] +
        w["operating_speed_gap"] +
        w["vru_context_risk"] +
        w["compliance_rate"]
    )

    gdf.loc[mask, "sss_raw"] = (
        w["speed_limit_alignment"] * gdf.loc[mask, "sub_score_limit_alignment"] +
        w["operating_speed_gap"]   * gdf.loc[mask, "sub_score_op_speed_gap"]   +
        w["vru_context_risk"]      * gdf.loc[mask, "sub_score_vru_risk"]        +
        w["compliance_rate"]       * gdf.loc[mask, "sub_score_compliance"]
    ) / total_w

    # Apply confidence multiplier
    gdf.loc[mask, "sss"] = (
        gdf.loc[mask, "sss_raw"] * gdf.loc[mask, "confidence_weight"]
    ).clip(0, 100)

    # Score band classification
    gdf.loc[mask, "sss_band"] = gdf.loc[mask, "sss"].apply(_classify_band)

    # Low data flag
    gdf["low_data_flag"] = (
        gdf["sample_size"].fillna(0) < MIN_SAMPLE_SIZE
    ) & gdf["scoreable"]

    # Policy recommendation text
    gdf.loc[mask, "sss_recommendation"] = gdf.loc[mask].apply(
        _generate_recommendation, axis=1
    )

    print("\nSSS computed. Score distribution:")
    print(gdf.loc[mask, "sss"].describe().round(1))
    print("\nBand distribution:")
    print(gdf.loc[mask, "sss_band"].value_counts())

    return gdf


# ─── 8. Band classification ───────────────────────────────────────────────────

def _classify_band(score: float) -> str:
    if pd.isna(score):
        return "No Data"
    for band, (lo, hi) in SCORE_BANDS.items():
        if lo <= score < hi:
            return band
    return "Critical" if score >= 78 else "Acceptable"


# ─── 9. Policy recommendation text ───────────────────────────────────────────

def _generate_recommendation(row: pd.Series) -> str:
    """Generate plain-English policy recommendation for each segment."""
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

    # Helmet-specific note
    if cc == "MH":
        parts.append(
            "Note: Maharashtra helmet wearing rate is low (~21%) — "
            "any intervention should be paired with helmet enforcement."
        )
    elif cc == "TH" and lu == "rural":
        parts.append(
            "Note: Thailand rural helmet wearing rate (~67%) remains below "
            "Safe System target — pair speed intervention with helmet campaign."
        )

    if band in ("Critical", "High Risk"):
        parts.append("Priority segment: recommend immediate site review.")

    return " ".join(parts) if parts else "No specific action flagged."

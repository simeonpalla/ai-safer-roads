"""
advanced_scoring.py — Science-grounded enhancements on top of base SSS.

Five modules:
  A. nilsson()          → WHO Power Model fatal/injury risk ratios
  B. credibility()      → Is the speed limit actually respected?
  C. recommend_limit()  → Evidence-based recommended speed limit
  D. lives_saved()      → Estimated fatality reduction from intervention
  E. detect_corridors() → Policy-actionable intervention zones

Call run_advanced_scoring(gdf) to run all five.
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint

warnings.filterwarnings("ignore")

# ── WHO regional fatality rates per billion vehicle-km ────────────────────────
# Source: WHO Global Status Report on Road Safety 2023
WHO_FATALITY_RATE = {"MH": 8.5, "TH": 6.2, "default": 7.0}

# ── Traffic volume proxy ──────────────────────────────────────────────────────
# WeightedSample = GPS probe-observation count (ADB dataset)
# Calibrated so study-area total ≈ 200–400 lives/year (consistent with
# Maharashtra ~13k + Thailand ~17k national totals, study = fraction of network)
VKM_PER_WEIGHTED_SAMPLE = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# A. NILSSON POWER MODEL
# Nilsson G. (2004). Traffic Safety Dimensions and the Power Model.
# Lund Institute of Technology. Cited in WHO Global Road Safety Report.
# ═══════════════════════════════════════════════════════════════════════════════

def nilsson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fatal crash risk = (observed_speed / safe_speed)^4
    Serious injury  = (observed_speed / safe_speed)^3

    observed_speed = F85th percentile (what drivers actually do)
    safe_speed     = Safe System threshold for this road type

    ratio = 1.0 → at baseline
    ratio = 2.0 → double the fatal crash risk
    ratio = 6.5 → Thailand urban secondary (80 km/h limit, 100 km/h actual)
    """
    gdf  = gdf.copy()
    mask = gdf["scoreable"] & gdf["speed_85th"].notna() & gdf["ss_limit"].notna()
    obs  = gdf.loc[mask, "speed_85th"]
    safe = gdf.loc[mask, "ss_limit"].replace(0, np.nan)

    gdf.loc[mask, "nilsson_fatal_ratio"]      = (obs / safe) ** 4
    gdf.loc[mask, "nilsson_injury_ratio"]     = (obs / safe) ** 3
    gdf.loc[mask, "nilsson_fatal_pct_excess"] = (
        (gdf.loc[mask, "nilsson_fatal_ratio"] - 1) * 100
    ).clip(lower=0)

    def _label(r):
        if pd.isna(r):  return "No data"
        if r <= 1.1:    return "At or near Safe System baseline"
        if r <= 2.0:    return f"{r:.1f}x baseline — Elevated"
        if r <= 4.0:    return f"{r:.1f}x baseline — High"
        return          f"{r:.1f}x baseline — Critical"

    gdf.loc[mask, "nilsson_interpretation"] = (
        gdf.loc[mask, "nilsson_fatal_ratio"].apply(_label)
    )

    n2 = (gdf.loc[mask, "nilsson_fatal_ratio"] > 2).sum()
    n4 = (gdf.loc[mask, "nilsson_fatal_ratio"] > 4).sum()
    mx = gdf.loc[mask, "nilsson_fatal_ratio"].max()
    print(f"  Nilsson fatal ratio range "
          f"{gdf.loc[mask,'nilsson_fatal_ratio'].min():.2f} – {mx:.2f}")
    print(f"  Segments with >2x fatal risk: {n2:,}")
    print(f"  Segments with >4x fatal risk: {n4:,}")
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# B. SPEED LIMIT CREDIBILITY
# 85th percentile rule — AASHTO Green Book; TRL Speed Limit Appraisal Framework
# ═══════════════════════════════════════════════════════════════════════════════

def credibility(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    A limit is credible when 85th pct speed ≤ posted limit + 10 km/h.
    Non-credible limits are dangerous: drivers learn to ignore ALL signage.

    Credible        → gap ≤ 10 km/h
    Low Credibility → gap 11–20 km/h
    Non-Credible    → gap > 20 km/h   (limit effectively ignored)
    Under-Speed     → 85th < limit - 10 (poor road condition / heavy trucks)
    """
    gdf  = gdf.copy()
    mask = gdf["scoreable"] & gdf["speed_85th"].notna() & gdf["speed_limit"].notna()
    gap  = gdf.loc[mask, "speed_85th"] - gdf.loc[mask, "speed_limit"]
    gdf.loc[mask, "credibility_gap"] = gap.round(1)

    def _class(g):
        if pd.isna(g):  return "No data"
        if g < -10:     return "Under-Speed"
        if g <= 10:     return "Credible"
        if g <= 20:     return "Low Credibility"
        return          "Non-Credible"

    def _action(g):
        if pd.isna(g):  return ""
        if g < -10:     return "Investigate road condition / traffic composition"
        if g <= 10:     return "Maintain enforcement"
        if g <= 20:     return "Increase enforcement or add physical calming"
        return          "Limit reform required — signage is not working"

    gdf.loc[mask, "credibility_class"]       = gap.apply(_class)
    gdf.loc[mask, "credibility_intervention"]= gap.apply(_action)

    print(f"\n  Credibility breakdown:")
    counts = gdf.loc[mask, "credibility_class"].value_counts()
    total  = counts.sum()
    for label, n in counts.items():
        print(f"    {label:<20} {n:>6,}  ({100*n/total:.1f}%)")
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# C. RECOMMENDED SPEED LIMIT
# Lower of: Safe System ceiling AND floor(85th pct / 10) * 10
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_limit(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    recommended = min(Safe System ceiling, floor(85th/10)*10)
    The 85th pct sets the behavioural baseline; Safe System is the hard cap.

    Change effort:
      No change needed    → recommended >= posted
      Minor  (<=10 km/h) → small adjustment
      Moderate (11-20)   → consultation / signage programme
      Major  (>20 km/h)  → political reform required
    """
    gdf  = gdf.copy()
    mask = (gdf["scoreable"] & gdf["speed_85th"].notna() &
            gdf["ss_limit"].notna() & gdf["speed_limit"].notna())

    f85    = gdf.loc[mask, "speed_85th"]
    ss     = gdf.loc[mask, "ss_limit"]
    posted = gdf.loc[mask, "speed_limit"]

    behavioural = (np.floor(f85 / 10) * 10).clip(lower=20)
    recommended = np.minimum(ss, behavioural).clip(lower=20)

    gdf.loc[mask, "recommended_limit"]   = recommended
    gdf.loc[mask, "limit_change_needed"] = (posted - recommended).round(1)

    def _effort(d):
        if pd.isna(d): return "Unknown"
        if d <= 0:     return "No change needed"
        if d <= 10:    return "Minor (<=10 km/h)"
        if d <= 20:    return "Moderate (11-20 km/h)"
        return         "Major (>20 km/h)"

    gdf.loc[mask, "change_effort"] = (
        gdf.loc[mask, "limit_change_needed"].apply(_effort)
    )

    print(f"\n  Limit change effort breakdown:")
    counts = gdf.loc[mask, "change_effort"].value_counts()
    total  = counts.sum()
    for label, n in counts.items():
        print(f"    {label:<25} {n:>6,}  ({100*n/total:.1f}%)")

    needs  = (gdf.loc[mask, "limit_change_needed"] > 0).sum()
    avg    = gdf.loc[mask & (gdf["limit_change_needed"] > 0),
                     "limit_change_needed"].mean()
    print(f"\n  Segments needing limit reduction: {needs:,}")
    print(f"  Average required reduction: {avg:.1f} km/h")
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# D. LIVES SAVED ESTIMATE
# Nilsson Power Model + WHO regional fatality rates
# Elvik R. (2009) Meta-analysis of speed and safety. Accident Analysis.
# ═══════════════════════════════════════════════════════════════════════════════

def lives_saved(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Estimates annual lives saved if speed limits corrected to recommended values.

    Method:
      1. Traffic proxy: WeightedSample * VKM_PER_WEIGHTED_SAMPLE = annual vkm
      2. Current fatalities = vkm / 1e9 * WHO regional rate
      3. Risk reduction = 1 - (recommended/observed)^4  [Nilsson]
      4. Lives saved = current_fatalities * risk_reduction

    Uncertainty: central ± factor of 2 (50% / 200% bounds)
    NOTE: ORDER-OF-MAGNITUDE proxy. All assumptions documented.
    """
    gdf = gdf.copy()

    ws_col = ("weighted_sample" if "weighted_sample" in gdf.columns else
              "sample_size"      if "sample_size"      in gdf.columns else None)
    if ws_col is None:
        print("  No traffic proxy column found — skipping lives saved.")
        return gdf

    mask = (gdf["scoreable"] & gdf["speed_85th"].notna() &
            gdf["recommended_limit"].notna() & gdf[ws_col].notna())

    ws      = gdf.loc[mask, ws_col].clip(lower=0)
    obs     = gdf.loc[mask, "speed_85th"]
    rec     = gdf.loc[mask, "recommended_limit"]
    country = gdf.loc[mask, "country_code"]

    vkm  = ws * VKM_PER_WEIGHTED_SAMPLE
    gdf.loc[mask, "est_annual_vkm"] = vkm

    rate = country.map(WHO_FATALITY_RATE).fillna(WHO_FATALITY_RATE["default"])
    current = (vkm / 1e9) * rate
    gdf.loc[mask, "est_current_fatalities"] = current.round(4)

    risk_ratio   = ((rec / obs) ** 4).clip(upper=1.0)
    risk_reduct  = (1 - risk_ratio).clip(lower=0)
    central      = current * risk_reduct

    gdf.loc[mask, "est_lives_saved"]   = central.round(4)
    gdf.loc[mask, "lives_saved_lower"] = (central * 0.5).round(4)
    gdf.loc[mask, "lives_saved_upper"] = (central * 2.0).round(4)

    print(f"\n  ── Lives Saved Estimates (study area, annual) ──")
    print(f"  Est. current annual fatalities (proxy): {current.sum():.1f}")
    print(f"  Est. lives saved if limits corrected:   {central.sum():.1f}")
    print(f"  Uncertainty range:                      "
          f"{central.sum()*0.5:.1f} – {central.sum()*2.0:.1f}")
    print(f"  NOTE: These are order-of-magnitude estimates.")
    print(f"        Assumptions documented in methodology.")
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# E. HIGH-RISK CORRIDOR DETECTION
# Attribute-based grouping — province + road class + sss band
# ═══════════════════════════════════════════════════════════════════════════════

def detect_corridors(
    gdf: gpd.GeoDataFrame,
    min_sss: float = 50.0,
    buffer_m: float = 50.0,   # unused, kept for API compat
    min_segments: int = 3,
) -> gpd.GeoDataFrame:
    """
    Group high-risk segments into policy-actionable intervention zones.

    WHY NOT spatial buffer:
      Road networks are physically connected — any buffer merges everything
      into one national-scale blob. Useless for ministry-level planning.

    METHOD: attribute grouping
      Group by country + region + road_class + sss_band
      Thailand: region = province_id (77 provinces)
      Maharashtra: region = land_use + road_class (no province data)

      Each group = one corridor a ministry can act on:
        "Thailand | Bangkok | Primary | Critical"
        "Maharashtra | urban_secondary | High Risk"

    Representative geometry: convex hull of segment centroids per group.
    """
    from config import SCORE_BANDS

    mask      = gdf["scoreable"] & gdf["sss"].notna() & (gdf["sss"] >= min_sss)
    high_risk = gdf[mask].copy()

    if len(high_risk) == 0:
        print(f"\n  No segments with SSS >= {min_sss} found.")
        return gpd.GeoDataFrame()

    # Build region label per segment
    def _region(row):
        pid = row.get("province_id", None)
        if pid is not None and str(pid) not in ("nan", "None", ""):
            return str(pid)
        lu = str(row.get("land_use", "unknown"))
        rc = str(row.get("road_class_norm", "unknown"))
        return f"{lu}_{rc}"

    high_risk["_region"] = high_risk.apply(_region, axis=1)

    group_keys = ["country_code", "_region", "road_class_norm", "sss_band"]
    group_keys = [k for k in group_keys if k in high_risk.columns]

    # Aggregate stats per group
    agg = {
        "sss":            "mean",
        "segment_id":     "count",
        "speed_limit":    "mean",
        "speed_85th":     "mean",
        "pct_over_limit": "mean",
    }
    opt = {
        "nilsson_fatal_ratio": "max",
        "est_lives_saved":     "sum",
        "lives_saved_lower":   "sum",
        "lives_saved_upper":   "sum",
        "recommended_limit":   "mean",
        "limit_change_needed": "mean",
        "credibility_class":   lambda x: x.mode().iloc[0] if len(x) else "—",
        "change_effort":       lambda x: x.mode().iloc[0] if len(x) else "—",
        "land_use":            lambda x: x.mode().iloc[0] if len(x) else "—",
    }
    for col, func in opt.items():
        if col in high_risk.columns:
            agg[col] = func

    grouped = (high_risk.groupby(group_keys)
                        .agg(agg)
                        .reset_index()
                        .rename(columns={"segment_id": "n_segments"}))
    grouped  = grouped[grouped["n_segments"] >= min_segments].copy()

    if len(grouped) == 0:
        print("  No corridor groups with enough segments.")
        return gpd.GeoDataFrame()

    # Build representative geometry: convex hull of centroids per group
    hr4326 = high_risk.to_crs(epsg=4326)
    hr4326["_cx"] = hr4326.geometry.centroid.x
    hr4326["_cy"] = hr4326.geometry.centroid.y

    geom_map = {}
    for keys, sub in hr4326.groupby(group_keys):
        k = keys if isinstance(keys, tuple) else (keys,)
        pts = MultiPoint(list(zip(sub["_cx"], sub["_cy"])))
        geom_map[k] = pts.convex_hull if len(sub) > 2 else pts.centroid

    records = []
    for _, row in grouped.iterrows():
        k    = tuple(row[kk] for kk in group_keys)
        geom = geom_map.get(k, None)
        records.append({**row.to_dict(), "geometry": geom})

    corridors = gpd.GeoDataFrame(records, crs="EPSG:4326")
    corridors = corridors.dropna(subset=["geometry"])

    # Human-readable label
    corridors["corridor_label"] = (
        corridors["country_code"].astype(str) + " | " +
        corridors["_region"].astype(str)       + " | " +
        corridors["road_class_norm"].astype(str)
    )

    # Area proxy (convex hull km²)
    corridors_m = corridors.to_crs(epsg=3857)
    corridors["area_km2"] = (corridors_m.geometry.area / 1e6).round(2)

    # Priority rank
    rank_col  = "est_lives_saved" if "est_lives_saved" in corridors.columns else "sss"
    corridors = (corridors.sort_values(rank_col, ascending=False)
                          .reset_index(drop=True))
    corridors["priority_rank"] = range(1, len(corridors) + 1)
    corridors["corridor_id"]   = range(1, len(corridors) + 1)

    print(f"\n  ── High-Risk Corridor Groups (SSS >= {min_sss}) ──")
    print(f"  Total corridor groups: {len(corridors)}")
    print(f"  High-risk segments covered: {corridors['n_segments'].sum():,}")

    show = ["priority_rank", "corridor_label", "n_segments", "sss"]
    if "nilsson_fatal_ratio" in corridors.columns: show.append("nilsson_fatal_ratio")
    if "est_lives_saved"     in corridors.columns: show.append("est_lives_saved")
    if "change_effort"       in corridors.columns: show.append("change_effort")
    show = [c for c in show if c in corridors.columns]
    print(f"\n  Top 10 corridors:")
    print(corridors[show].head(10).round(2).to_string(index=False))

    return corridors


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER — run all five modules
# ═══════════════════════════════════════════════════════════════════════════════

def run_advanced_scoring(gdf: gpd.GeoDataFrame) -> tuple:
    print("\n" + "="*60)
    print("  ADVANCED SCORING — 5 MODULES")
    print("="*60)

    print("\n[A] Nilsson Power Model (WHO fatal risk ratios)...")
    gdf = nilsson(gdf)

    print("\n[B] Speed Limit Credibility...")
    gdf = credibility(gdf)

    print("\n[C] Recommended Speed Limits...")
    gdf = recommend_limit(gdf)

    print("\n[D] Lives Saved Estimates (Nilsson + WHO fatality rates)...")
    gdf = lives_saved(gdf)

    print("\n[E] High-Risk Corridor Detection...")
    corridors = detect_corridors(gdf, min_sss=50.0, min_segments=3)

    print("\n" + "="*60)
    print("  Advanced scoring complete.")
    print("="*60)

    return gdf, corridors

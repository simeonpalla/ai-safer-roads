"""
preprocessing.py — Load, clean, and harmonize both country datasets.
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

from config import ROAD_CLASS_MAP, LAND_USE_MAP

warnings.filterwarnings("ignore")


MH_COLUMN_ALIASES = {
    "DISSOLVE_ID":          "segment_id",
    "class":                "road_class_raw",
    "subtype":              "road_subtype",
    "names_primary":        "road_name",
    "UrbanPC":              "urban_pct",
    "SampleSize_avg":       "sample_size",
    "RoadLength":           "road_length_m",
    "WeightedSample":       "weighted_sample",
    "Percent_":             "percentile_rank",
    "Percentile":           "percentile",
    "SpeedLimit":           "speed_limit_raw",
    "RoadClass":            "road_class",
    "LandUse":              "land_use_raw",
    "NumberOverLimit":      "n_over_limit",
    "MedianSpeed":          "median_speed",
    "F85thPercentileSpeed": "speed_85th",
    "PercentOverLimit":     "pct_over_limit",
    "RankedPercentile":     "ranked_percentile",
    "SpeedLimitFloor":      "speed_limit_floor",
    "PercentileBand":       "percentile_band",
    "AnalysisStatus":       "analysis_status",
    "StreetImageLink":      "image_url",
    "ExcludeFromSpeedSPI":  "exclude_flag",
    "Pass":                 "pass_flag",
    "Sample_Size_Total":    "sample_size_total",
}

TH_COLUMN_ALIASES = {
    "OvertureID":            "segment_id",
    "english_ro":            "road_name",
    "SampleSize_avg":        "sample_size",
    "RoadLength":            "road_length_m",
    "WeightedSample":        "weighted_sample",
    "Percent_":              "percentile_rank",
    "Percentile":            "percentile",
    "SpeedLimit":            "speed_limit_raw",
    "RoadClass":             "road_class",
    "LandUse":               "land_use_raw",
    "NumberOverLimit":       "n_over_limit",
    "MedianSpeed":           "median_speed",
    "F85thPercentileSpeed":  "speed_85th",
    "PercentOverLimit":      "pct_over_limit",
    "RankedPercentile":      "ranked_percentile",
    "SpeedLimitFloor":       "speed_limit_floor",
    "PercentileBand":        "percentile_band",
    "ForAnalysis":           "for_analysis",
    "InvPercentile":         "inv_percentile",
    "ProvinceID":            "province_id",
    "AnalysisStatus":        "analysis_status",
    "ExcludeFromSpeedSPI":   "exclude_flag",
    "StreetImageLink":       "image_url",
    "NO_OF_Result_Segments": "n_result_segments",
    "SampleSizeTotal":       "sample_size_total",
}


def _normalize_road_class(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.lower().str.strip()
        .map(ROAD_CLASS_MAP).fillna("unknown")
    )


def _normalize_land_use(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.lower().str.strip()
        .map(LAND_USE_MAP).fillna("unknown")
    )


def _parse_speed_limit(series: pd.Series) -> pd.Series:
    """Convert any SpeedLimit representation to numeric km/h."""
    s = series.copy().astype(str).str.strip()
    numeric = s.str.extract(r"(\d+\.?\d*)")[0].astype(float)
    mph_mask = s.str.lower().str.contains("mph", na=False)
    numeric.loc[mph_mask] = numeric.loc[mph_mask] * 1.60934
    return numeric


def _derive_has_speed_data(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Derive has_speed_data purely from whether core speed columns are populated.
    This is MORE reliable than checking AnalysisStatus strings or ForAnalysis flags,
    which vary between dataset versions.
    """
    speed_cols = ["speed_85th", "median_speed", "pct_over_limit"]
    present = [c for c in speed_cols if c in gdf.columns]
    if not present:
        return pd.Series(False, index=gdf.index)
    # A segment has speed data if ANY of the core columns is non-null
    has_data = pd.Series(False, index=gdf.index)
    for col in present:
        has_data = has_data | gdf[col].notna()
    return has_data


def load_maharashtra(filepath: str) -> gpd.GeoDataFrame:
    print(f"Loading Maharashtra data from: {filepath}")
    gdf = gpd.read_file(filepath)

    # Debug: print raw column info
    print(f"  Raw columns: {list(gdf.columns)}")
    print(f"  AnalysisStatus values: {gdf['AnalysisStatus'].value_counts().to_dict()}")

    gdf = gdf.rename(columns={k: v for k, v in MH_COLUMN_ALIASES.items() if k in gdf.columns})
    gdf["country"] = "India (Maharashtra)"
    gdf["country_code"] = "MH"

    # Speed limit
    if "speed_limit_raw" in gdf.columns:
        gdf["speed_limit"] = _parse_speed_limit(gdf["speed_limit_raw"])
    # SpeedLimitFloor fallback removed — data user guide says ignore this field.

    # Road class + land use
    if "road_class" in gdf.columns:
        gdf["road_class_norm"] = _normalize_road_class(gdf["road_class"])
    if "land_use_raw" in gdf.columns:
        gdf["land_use"] = _normalize_land_use(gdf["land_use_raw"])

    # Urban pct fallback for land use
    if "urban_pct" in gdf.columns:
        gdf["land_use_from_urban_pct"] = np.where(
            gdf["urban_pct"] > 50, "urban", "rural"
        )
        if "land_use" in gdf.columns:
            gdf["land_use"] = gdf["land_use"].where(
                gdf["land_use"] != "unknown", gdf["land_use_from_urban_pct"]
            )
        else:
            gdf["land_use"] = gdf["land_use_from_urban_pct"]

    # ── KEY FIX: derive has_speed_data from actual column content ──────────
    gdf["has_speed_data"] = _derive_has_speed_data(gdf)

    n_speed = gdf["has_speed_data"].sum()
    n_limit = gdf["speed_limit"].notna().sum() if "speed_limit" in gdf.columns else 0
    n_85th  = gdf["speed_85th"].notna().sum() if "speed_85th" in gdf.columns else 0
    print(f"  → {len(gdf):,} total | {n_speed:,} with speed data | "
          f"{n_limit:,} with speed limit | {n_85th:,} with 85th pct speed")
    return gdf


def load_thailand(filepath: str) -> gpd.GeoDataFrame:
    print(f"Loading Thailand data from: {filepath}")
    # Default layer is ADB_Results_D4
    gdf = gpd.read_file(filepath)

    print(f"  Raw columns: {list(gdf.columns)}")

    # Check ForAnalysis values before rename
    if "ForAnalysis" in gdf.columns:
        print(f"  ForAnalysis unique values: {gdf['ForAnalysis'].value_counts().to_dict()}")
        print(f"  ForAnalysis dtype: {gdf['ForAnalysis'].dtype}")
        print(f"  ForAnalysis non-null: {gdf['ForAnalysis'].notna().sum()}")

    gdf = gdf.rename(columns={k: v for k, v in TH_COLUMN_ALIASES.items() if k in gdf.columns})
    gdf["country"] = "Thailand"
    gdf["country_code"] = "TH"

    # Speed limit
    if "speed_limit_raw" in gdf.columns:
        gdf["speed_limit"] = pd.to_numeric(gdf["speed_limit_raw"], errors="coerce")
    # SpeedLimitFloor fallback removed — data user guide says ignore this field.

    if "road_class" in gdf.columns:
        gdf["road_class_norm"] = _normalize_road_class(gdf["road_class"])
    if "land_use_raw" in gdf.columns:
        gdf["land_use"] = _normalize_land_use(gdf["land_use_raw"])

    # ── KEY FIX: derive has_speed_data from actual column content ──────────
    gdf["has_speed_data"] = _derive_has_speed_data(gdf)

    # ForAnalysis in the TH GeoJSON contains speed-limit-like values (30/50/80/90 km/h),
    # NOT a binary 0/1 flag. The column-content check in _derive_has_speed_data() is the
    # correct authority for has_speed_data; no secondary check needed here.

    n_speed = gdf["has_speed_data"].sum()
    n_limit = gdf["speed_limit"].notna().sum() if "speed_limit" in gdf.columns else 0
    n_85th  = gdf["speed_85th"].notna().sum() if "speed_85th" in gdf.columns else 0
    print(f"  → {len(gdf):,} total | {n_speed:,} with speed data | "
          f"{n_limit:,} with speed limit | {n_85th:,} with 85th pct speed")
    return gdf


def load_helmet_data(filepath: str) -> pd.DataFrame:
    print(f"Loading helmet data from: {filepath}")
    return pd.read_excel(filepath)


def merge_datasets(mh: gpd.GeoDataFrame, th: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    KEEP_COLS = [
        "segment_id", "country", "country_code", "road_name",
        "road_class", "road_class_norm", "land_use",
        "speed_limit", "speed_limit_floor",
        "median_speed", "speed_85th",
        "pct_over_limit", "n_over_limit",
        "sample_size", "sample_size_total", "weighted_sample",
        "ranked_percentile", "percentile_band",
        "analysis_status", "exclude_flag", "has_speed_data",
        "image_url", "geometry",
    ]
    for col in ["urban_pct", "province_id", "road_length_m"]:
        if col in mh.columns or col in th.columns:
            KEEP_COLS.append(col)

    def _keep(gdf):
        cols = [c for c in KEEP_COLS if c in gdf.columns]
        return gdf[cols].copy()

    mh = mh.to_crs(epsg=4326)
    th = th.to_crs(epsg=4326)

    combined = gpd.GeoDataFrame(
        pd.concat([_keep(mh), _keep(th)], ignore_index=True),
        crs="EPSG:4326"
    )

    # Diagnostic
    print(f"\nCombined dataset: {len(combined):,} total segments")
    print(f"  has_speed_data=True: {combined['has_speed_data'].sum():,}")
    print(f"  speed_limit non-null: {combined['speed_limit'].notna().sum():,}")
    print(f"  speed_85th non-null: {combined['speed_85th'].notna().sum():,}")
    print(f"  median_speed non-null: {combined['median_speed'].notna().sum():,}")
    return combined


def get_analysis_subset(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Mark segments as scoreable, at two tiers.

    TIER 2 — "scoreable" (full score): requires posted limit AND 85th-pct
    speed AND general speed-data presence, all NONZERO. This is the
    behaviourally-confirmed SSS most of this pipeline reports.

    TIER 1 — "alignment_scoreable" (NEW, v3.1): requires ONLY a usable
    posted limit (no behavioural/GPS data needed at all). This produces
    sub_score_limit_alignment alone — "does the posted limit match the
    Safe System standard for this road class/land use" — which doesn't
    need F85 or median speed. It exists because the brief explicitly asks
    for a methodology that's "scalable and replicable across countries,"
    and many ADB member countries won't have rich GPS-probe behavioural
    data the way this MH/TH demo dataset does. A method that hard-depends
    on behavioural data isn't very replicable.

    HONEST CAVEAT for THIS dataset specifically: Tier 1 will NOT
    dramatically raise coverage here, because in both MH and TH most of
    the ~79% unscored segments are missing speed_limit too, not just
    behavioural fields (ADB's AnalysisStatus/ForAnalysis flags appear to
    gate most fields together, not selectively). Tier 1 still adds real
    value: it's the right architecture for other countries' data, and it
    picks up whatever segments in THIS dataset have a limit but lack full
    behavioural confirmation.

    BUG FIX (reviewer feedback, June 2026): some source rows use a literal
    0 as a placeholder for "no data" instead of NaN — e.g. speed_limit=0,
    median_speed=0, AND speed_85th=0 all at once on the same row. A posted
    speed limit of 0 km/h doesn't exist on a real road, and an 85th-
    percentile speed of exactly 0 km/h for an entire sampled segment would
    only happen if a road were permanently gridlocked — for an entire
    segment's sample, this is for all practical purposes never genuine.
    The previous .notna()-only check let these placeholder rows through as
    "scoreable", and they then surfaced in the map/CSV/AI layer labelled
    "Acceptable" with a real-looking SSS/Priority Index, which is wrong:
    they should be excluded, not scored as calm roads.
    median_speed is NOT included in this zero-check by itself — a real,
    very low (but nonzero) median can legitimately occur on a congested
    segment, so median_speed==0 alone does not invalidate a row.
    """
    gdf = gdf.copy()

    has_speed_data = gdf["has_speed_data"].fillna(False)
    has_limit      = gdf["speed_limit"].notna() & (gdf["speed_limit"] > 0)
    has_85th       = gdf["speed_85th"].notna()  & (gdf["speed_85th"]  > 0)
    # median_speed is desirable but not a hard requirement
    has_median     = gdf["median_speed"].notna() if "median_speed" in gdf.columns \
                     else pd.Series(True, index=gdf.index)

    # Honour ADB's ExcludeFromSpeedSPI flag — MH is 0/1 numeric, TH may be "YES"/"NO" string
    if "exclude_flag" in gdf.columns:
        excl = gdf["exclude_flag"]
        not_excluded = ~(
            (excl == 1) |
            (excl.astype(str).str.upper().isin(["YES", "1", "TRUE"]))
        )
        not_excluded = not_excluded.fillna(True)  # NaN = no flag = not excluded
    else:
        not_excluded = pd.Series(True, index=gdf.index)

    # ExcludeFromSpeedSPI is ADB's flag for speed-behaviour data quality issues.
    # Tier 2 (full SSS uses F85/median speed) → respect the flag.
    # Tier 1 (alignment-only, posted limit vs Safe System standard, no speed data) → ignore it;
    # a bad speed sample doesn't make the posted limit invalid.
    gdf["scoreable"]           = has_speed_data & has_limit & has_85th & not_excluded
    gdf["alignment_scoreable"] = has_limit  # Tier 1 — posted limit alone, flag irrelevant

    # Diagnostic breakdown
    n_zero_limit = int((gdf["speed_limit"] == 0).sum()) if "speed_limit" in gdf.columns else 0
    n_zero_85th  = int((gdf["speed_85th"]  == 0).sum()) if "speed_85th"  in gdf.columns else 0
    print(f"\nScoreable condition breakdown:")
    print(f"  has_speed_data:   {has_speed_data.sum():,}")
    print(f"  has_limit (>0):   {has_limit.sum():,}  "
          f"({n_zero_limit:,} excluded as speed_limit==0 placeholder)")
    print(f"  has_85th (>0):    {has_85th.sum():,}  "
          f"({n_zero_85th:,} excluded as speed_85th==0 placeholder)")
    print(f"  has_median:       {has_median.sum():,}")
    print(f"  TIER 2 (scoreable, full SSS):        {gdf['scoreable'].sum():,} / {len(gdf):,} "
          f"({100*gdf['scoreable'].mean():.1f}%)")
    print(f"  TIER 1 (alignment_scoreable, limit only): {gdf['alignment_scoreable'].sum():,} / {len(gdf):,} "
          f"({100*gdf['alignment_scoreable'].mean():.1f}%)")
    n_tier1_only = (gdf["alignment_scoreable"] & ~gdf["scoreable"]).sum()
    print(f"  Tier 1 adds {n_tier1_only:,} segments beyond Tier 2 "
          f"(have a posted limit but lack full behavioural confirmation)")

    return gdf

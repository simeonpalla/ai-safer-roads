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
    if "speed_limit_floor" in gdf.columns and "speed_limit" in gdf.columns:
        # Fill any remaining NaN speed_limit from floor
        gdf["speed_limit"] = gdf["speed_limit"].fillna(
            pd.to_numeric(gdf["speed_limit_floor"], errors="coerce")
        )

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
    if "speed_limit_floor" in gdf.columns:
        gdf["speed_limit"] = gdf.get("speed_limit", pd.Series(dtype=float)).fillna(
            pd.to_numeric(gdf["speed_limit_floor"], errors="coerce")
        )

    if "road_class" in gdf.columns:
        gdf["road_class_norm"] = _normalize_road_class(gdf["road_class"])
    if "land_use_raw" in gdf.columns:
        gdf["land_use"] = _normalize_land_use(gdf["land_use_raw"])

    # ── KEY FIX: derive has_speed_data from actual column content ──────────
    gdf["has_speed_data"] = _derive_has_speed_data(gdf)

    # Also set using ForAnalysis if available (as secondary check)
    if "for_analysis" in gdf.columns:
        fa = pd.to_numeric(gdf["for_analysis"], errors="coerce")
        gdf["has_speed_data"] = gdf["has_speed_data"] | (fa == 1)
        print(f"  for_analysis == 1 count: {(fa == 1).sum()}")

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
        "analysis_status", "has_speed_data",
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
    Mark segments as scoreable.
    Requires: has speed data AND speed_limit AND speed_85th.
    median_speed is optional (used in scoring but not required).
    """
    gdf = gdf.copy()

    has_speed_data = gdf["has_speed_data"].fillna(False)
    has_limit      = gdf["speed_limit"].notna()
    has_85th       = gdf["speed_85th"].notna()
    # median_speed is desirable but not a hard requirement
    has_median     = gdf["median_speed"].notna() if "median_speed" in gdf.columns \
                     else pd.Series(True, index=gdf.index)

    gdf["scoreable"] = has_speed_data & has_limit & has_85th

    # Diagnostic breakdown
    print(f"\nScoreable condition breakdown:")
    print(f"  has_speed_data:   {has_speed_data.sum():,}")
    print(f"  has_limit:        {has_limit.sum():,}")
    print(f"  has_85th:         {has_85th.sum():,}")
    print(f"  has_median:       {has_median.sum():,}")
    print(f"  ALL (scoreable):  {gdf['scoreable'].sum():,} / {len(gdf):,} "
          f"({100*gdf['scoreable'].mean():.1f}%)")

    return gdf

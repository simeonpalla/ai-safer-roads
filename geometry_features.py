"""
geometry_features.py — Road geometry analysis for safe speed adjustment.

WHY THIS MATTERS:
  The Safe System speed limits in config.py are derived from road class × land use.
  That table treats a straight rural primary highway identically to a winding
  mountain road of the same class — both get 70 km/h SS limit. Road geometry
  says otherwise: sight distance on a curve limits the speed at which a driver
  can see and react to a hazard. AASHTO Green Book and iRAP both specify that
  design speed must account for horizontal curvature — a highly sinuous alignment
  should carry a lower design speed ceiling than a straight one.

  This module computes two geometry-derived features:
    sinuosity       — actual length / crow-flies distance. 1.0 = perfectly
                      straight. 1.2+ = moderately curved. 1.5+ = sharply curved.
    bearing_stddev  — standard deviation of per-vertex bearing changes (degrees).
                      High variance = unpredictable/frequent direction changes.

  Both feed into a downward adjustment of ss_limit in scoring.get_safe_system_limit().
  The adjustment is ONLY downward (a curved road never becomes SAFER than its
  class default would imply) and is capped so that ss_limit never drops below 30.

SOURCES:
  - AASHTO Policy on Geometric Design of Highways & Streets (Green Book) — Table 3-6:
    design speed decreases with increasing degree of curvature.
  - iRAP Road Assessment Programme protocol — sinuosity is an explicit attribute
    in iRAP's star rating methodology for design risk.
  - WHO Speed Management manual: "Speeds in excess of design speed increase crash
    risk exponentially on curved alignments."
"""

import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point


# Speed reductions applied to the SS limit for curved segments (km/h)
# Grounded in AASHTO table: design speed ~ 10-20 km/h lower per curvature tier.
SINUOSITY_SS_REDUCTION = {
    # (sinuosity_min, sinuosity_max): km/h_reduction
    (1.20, 1.50): 10,   # moderately curved
    (1.50, 2.00): 20,   # sharply curved (switchbacks, hill roads)
    (2.00, float("inf")): 25,  # severely sinuous
}


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _sinuosity(geom) -> float:
    """
    Sinuosity = actual path length / straight-line endpoint distance.
    Returns 1.0 for point/null geometries or segments shorter than 10 m.
    Capped at 5.0 to avoid noise from very short or self-intersecting segments.
    """
    if geom is None or geom.is_empty:
        return 1.0
    try:
        coords = list(geom.coords)
    except Exception:
        return 1.0
    if len(coords) < 2:
        return 1.0

    # Crow-flies distance
    c_start, c_end = coords[0], coords[-1]
    straight = _haversine_m(c_start[0], c_start[1], c_end[0], c_end[1])
    if straight < 10.0:
        return 1.0

    # Actual path length (sum of segment haversine distances)
    actual = sum(
        _haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        for i in range(len(coords) - 1)
    )
    return float(min(actual / straight, 5.0))


def _bearing_stddev(geom) -> float:
    """
    Standard deviation of bearing changes between consecutive vertices (degrees).
    0 = perfectly straight. Higher = more frequent/sharp direction changes.
    """
    if geom is None or geom.is_empty:
        return 0.0
    try:
        coords = list(geom.coords)
    except Exception:
        return 0.0
    if len(coords) < 3:
        return 0.0

    bearings = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        b = math.degrees(math.atan2(dx, dy)) % 360
        bearings.append(b)

    if len(bearings) < 2:
        return 0.0

    diffs = []
    for i in range(len(bearings) - 1):
        d = abs(bearings[i + 1] - bearings[i])
        diffs.append(min(d, 360 - d))  # normalise to [0, 180]

    return float(np.std(diffs))


def compute_geometry_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add sinuosity and bearing_stddev columns to a GeoDataFrame.
    Both are computed from the existing road geometry; no new data required.
    """
    if "geometry" not in gdf.columns or gdf["geometry"].isna().all():
        print("  geometry_features: no geometry column — skipping")
        gdf["sinuosity"] = 1.0
        gdf["bearing_stddev"] = 0.0
        return gdf

    gdf = gdf.copy()
    print("  Computing road geometry features (sinuosity, bearing_stddev)...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["sinuosity"]     = gdf["geometry"].apply(_sinuosity)
        gdf["bearing_stddev"] = gdf["geometry"].apply(_bearing_stddev)

    # Report distribution
    s = gdf["sinuosity"]
    print(f"  Sinuosity: mean={s.mean():.3f} | p50={s.median():.3f} | "
          f"p90={s.quantile(0.90):.3f} | max={s.max():.3f}")
    n_curved = (s >= 1.20).sum()
    n_sharply = (s >= 1.50).sum()
    print(f"  Curved segments (>=1.20): {n_curved:,}  |  Sharply curved (>=1.50): {n_sharply:,}")
    return gdf


def sinuosity_ss_adjustment(sinuosity: float) -> float:
    """
    Returns the km/h reduction to apply to a road's Safe System limit
    based on its sinuosity.  Always ≥ 0 (never increases the limit).
    """
    if pd.isna(sinuosity) or sinuosity < 1.20:
        return 0.0
    for (lo, hi), reduction in sorted(SINUOSITY_SS_REDUCTION.items()):
        if lo <= sinuosity < hi:
            return float(reduction)
    # Should not reach here (last bucket has inf upper bound)
    return float(max(SINUOSITY_SS_REDUCTION.values()))

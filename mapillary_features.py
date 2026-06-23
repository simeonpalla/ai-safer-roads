"""
mapillary_features.py — Extract road infrastructure features from Mapillary API.

WHY THIS MATTERS:
  Road safety datasets capture the physical road and its posted limits.
  They say nothing about whether the road has traffic signs, lane markings,
  or pedestrian infrastructure — and crucially, whether ANY street-level
  imagery even exists for that road.

  Mapillary's crowd-sourced and commercial imagery covers billions of road
  kilometres. Their API exposes:
    - trafficsign detections: speed limit signs, pedestrian warning signs, etc.
    - mvd_fast detections: barriers, lane markings, road furniture
    - panoptic detections: pixel-level scene understanding (road, sidewalk, building)

  This module queries the map_features API v4 per road segment and computes:

    mapillary_covered         — True if any Mapillary imagery exists for segment
    trafficsign_count         — number of traffic sign detections in segment bbox
    trafficsign_density       — signs per km (normalised for segment length)
    infra_visibility_score    — 0–100 composite: how well is this road's infrastructure
                                 digitally visible? 0 = invisible (no imagery at all),
                                 100 = well-documented with many detected objects.

  The "infrastructure blindspot" finding is itself policy-relevant:
    - Rural roads with ZERO Mapillary coverage AND high SSS scores are doubly
      concerning — they're dangerous AND invisible to digital monitoring systems.
    - Urban roads with coverage but zero traffic signs may have poor signage.

  This replaces the assumption "urban roads have better signage" with evidence.

MAPILLARY API v4 NOTES:
  Endpoint: GET https://graph.mapillary.com/map_features
  Auth:     ?access_token=<TOKEN>
  Params:   fields=id,object_type  &  bbox=west,south,east,north
  Bbox max: 0.01 sq degrees per call (enforced server-side)
  Rate limit: 60,000 requests/hour

  Practical bandwidth constraint (empirically discovered):
    - 0.01°×0.01° cells (0.0001 sq deg, ~1km²): completes in 3–15s.
      Bangkok dense corridors return 1,400–1,700 features per cell.
    - 0.09°×0.09° cells (0.0081 sq deg, ~10km²): server timeout >30s.
      A single Bangkok 0.09° cell would contain ~100,000+ features;
      the server cannot retrieve them within reasonable timeout limits.
    → Use 0.01° grid. This gives ~100× more cells but each is fast.

  API v4 returns coarse object_type categories, not detailed sign subtypes:
    "trafficsign"  — any traffic sign detected
    "mvd_fast"     — road objects (barriers, bollards, signals) via fast detection
    "panoptic"     — pixel-level semantic segmentation (presence of imagery)

  Detailed sign subtypes (regulatory--maximum-speed-limit--50, etc.) are not
  returned via the public API; they require Mapillary enterprise access.

USAGE:
  Set MAPILLARY_TOKEN in environment or pass directly to enrich_with_mapillary().
  Results are cached to enrichment_data/mapillary_cache/features_cache.json to
  avoid re-querying on subsequent pipeline runs.
"""

import os
import time
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

warnings.filterwarnings("ignore")

MAPILLARY_BASE = "https://graph.mapillary.com"

# Mapillary API v4 returns these three coarse object_type categories
OBJECT_TYPES = {"trafficsign", "mvd_fast", "panoptic"}


# ── API client ────────────────────────────────────────────────────────────────

def _query_map_features(bbox_wsen: tuple, token: str,
                         timeout: int = 20, max_retries: int = 2) -> list:
    """
    Query Mapillary map_features for a bounding box.
    bbox_wsen = (west_lon, south_lat, east_lon, north_lat)
    Returns list of feature dicts with object_type field.
    Returns [] on any error; never raises.
    """
    west, south, east, north = bbox_wsen
    # Clamp bbox to Mapillary's 0.01 sq-degree limit
    lon_span = min(east - west, 0.099)
    lat_span = min(north - south, 0.099)
    east  = west + lon_span
    north = south + lat_span

    url = f"{MAPILLARY_BASE}/map_features"
    params = {
        "access_token": token,
        "fields": "id,object_type",
        "bbox": f"{west},{south},{east},{north}",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            elif resp.status_code == 500:
                err = resp.json().get("error", {}).get("message", "")
                if "too large" in err.lower():
                    return []   # bbox problem — skip
                break
            else:
                break
        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1)
    return []


def _aggregate_features(features: list, segment_length_km: float = 0.1) -> dict:
    """
    Aggregate Mapillary feature counts into infrastructure scores.

    API v4 returns three coarse object_type values:
      "trafficsign"  — any traffic sign (speed limits, warning signs, etc.)
      "mvd_fast"     — road objects detected via fast multi-view detection
      "panoptic"     — panoptic segmentation features (scene understanding)

    Infrastructure visibility score (0-100):
      0   = no features at all (no imagery coverage)
      30  = only panoptic features (imagery exists but no specific objects detected)
      60  = some mvd_fast objects present
      100 = multiple traffic signs + road objects (well-signed, documented road)
    """
    n_total     = len(features)
    n_sign      = sum(1 for f in features if f.get("object_type") == "trafficsign")
    n_mvd       = sum(1 for f in features if f.get("object_type") == "mvd_fast")
    n_panoptic  = sum(1 for f in features if f.get("object_type") == "panoptic")

    if n_total == 0:
        return {
            "mapillary_n_features":      0,
            "mapillary_n_trafficsigns":  0,
            "mapillary_n_mvd":           0,
            "trafficsign_density":       0.0,
            "infra_visibility_score":    0.0,
        }

    sign_density = n_sign / max(segment_length_km, 0.01)

    # Visibility score: starts at 30 if ANY imagery, goes up with objects
    base = 30.0
    sign_contrib = min(n_sign / 5, 1.0) * 40.0   # up to 40 pts for 5+ signs
    mvd_contrib  = min(n_mvd / 10, 1.0) * 20.0   # up to 20 pts for 10+ mvd objects
    pan_contrib  = min(n_panoptic / 20, 1.0) * 10.0  # up to 10 pts for panoptic depth
    visibility = float(np.clip(base + sign_contrib + mvd_contrib + pan_contrib, 0, 100))

    return {
        "mapillary_n_features":      n_total,
        "mapillary_n_trafficsigns":  n_sign,
        "mapillary_n_mvd":           n_mvd,
        "trafficsign_density":       float(round(sign_density, 3)),
        "infra_visibility_score":    float(round(visibility, 1)),
    }


# Grid size for batched API queries — 0.01° × 0.01° = 0.0001 sq deg.
# NOTE: 0.09° × 0.09° cells (~10km) cause server timeouts in dense urban areas like
# Bangkok where a single cell may contain 100,000+ detections. 0.01° cells (~1km)
# complete in 3–15s (tested: Sukhumvit 1,422 features, Silom 1,576 features).
_GRID_DEG = 0.01


# ── Main enrichment function ──────────────────────────────────────────────────

def enrich_with_mapillary(
    gdf: gpd.GeoDataFrame,
    token: str = None,
    cache_dir: str = "enrichment_data/mapillary_cache",
    delay_s: float = 0.03,
    segment_mask: "pd.Series | None" = None,
    max_new_queries: int = None,
) -> gpd.GeoDataFrame:
    """
    Enrich road segments with Mapillary infrastructure features.

    Uses a grid-based query strategy: divides the study area into 0.01°×0.01°
    cells (~1 km), queries each unique cell once, then assigns coverage to all
    segments whose midpoint falls in that cell.

    segment_mask — optional boolean Series aligned to gdf.index.  When provided,
        only grid cells containing True-masked segments are queried (targeted mode:
        e.g. Critical + High Risk bands only).  Coverage results are still written
        to ALL segments that share a queried cell, not just the masked ones.
        Pass None to query cells for every segment (full-dataset mode).

    Columns added:
      mapillary_covered         — True if any features found in grid cell
      mapillary_n_features      — total features (all types) in cell
      mapillary_n_trafficsigns  — number of traffic sign detections
      mapillary_n_mvd           — number of mvd_fast object detections
      infra_visibility_score    — 0–100 (0=no imagery, 100=well-documented road)

    Segments with no Mapillary coverage get infra_visibility_score=0.
    This is intentional: invisible roads are a finding, not a missing value.

    Grid cache stored at: {cache_dir}/small_grid_cache.json
    """
    if token is None:
        token = os.environ.get("MAPILLARY_TOKEN", "")
    if not token:
        print("  mapillary_features: no token — skipping (set MAPILLARY_TOKEN)")
        for col in ["mapillary_covered", "mapillary_n_features",
                    "mapillary_n_trafficsigns", "mapillary_n_mvd",
                    "infra_visibility_score"]:
            gdf[col] = 0
        gdf["mapillary_covered"] = False
        return gdf

    if "geometry" not in gdf.columns or gdf["geometry"].isna().all():
        print("  mapillary_features: no geometry — skipping")
        return gdf

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    grid_cache_file = cache_path / "small_grid_cache.json"

    grid_cache = {}
    if grid_cache_file.exists():
        with open(grid_cache_file) as f:
            grid_cache = json.load(f)

    gdf = gdf.copy()
    for col in ["mapillary_covered", "mapillary_n_features",
                "mapillary_n_trafficsigns", "mapillary_n_mvd",
                "infra_visibility_score"]:
        gdf[col] = 0.0
    gdf["mapillary_covered"] = False

    # Compute grid cell for each segment midpoint
    bounds = gdf["geometry"].bounds
    midx = (bounds["minx"] + bounds["maxx"]) / 2
    midy = (bounds["miny"] + bounds["maxy"]) / 2
    gdf["_grid_lon"] = (midx / _GRID_DEG).astype(int) * _GRID_DEG
    gdf["_grid_lat"] = (midy / _GRID_DEG).astype(int) * _GRID_DEG

    # Determine which grid cells to query
    if segment_mask is not None:
        work_mask = segment_mask.reindex(gdf.index, fill_value=False)
        n_target = work_mask.sum()
        print(f"  Targeted mode: querying cells for {n_target:,} priority segments")
    else:
        work_mask = pd.Series(True, index=gdf.index)

    unique_cells = gdf.loc[work_mask, ["_grid_lon","_grid_lat"]].drop_duplicates()
    n_new = sum(1 for _, r in unique_cells.iterrows()
                if f"{r['_grid_lon']:.3f},{r['_grid_lat']:.3f}" not in grid_cache)
    n_will_query = min(n_new, max_new_queries) if max_new_queries is not None else n_new
    print(f"\n  Grid cells: {len(unique_cells):,} unique  ({len(grid_cache):,} cached, "
          f"{n_new:,} new)")
    if n_new > 0:
        print(f"  Querying {n_will_query:,} new cells "
              f"(capped at {max_new_queries} per run — re-run to fill remaining {n_new - n_will_query:,})"
              if max_new_queries and n_new > max_new_queries else
              f"  Est. query time: ~{n_will_query * (delay_s + 0.4) / 60:.1f} min")

    new_queries = 0
    for i, (_, cell) in enumerate(unique_cells.iterrows()):
        west  = cell["_grid_lon"]
        south = cell["_grid_lat"]
        east  = west + _GRID_DEG
        north = south + _GRID_DEG
        key   = f"{west:.3f},{south:.3f}"

        if key in grid_cache:
            continue

        if max_new_queries is not None and new_queries >= max_new_queries:
            break

        features = _query_map_features((west, south, east, north), token, timeout=8)
        n_sign = sum(1 for f in features if f.get("object_type") == "trafficsign")
        n_mvd  = sum(1 for f in features if f.get("object_type") == "mvd_fast")
        n_pan  = sum(1 for f in features if f.get("object_type") == "panoptic")
        grid_cache[key] = {
            "n_total": len(features), "n_sign": n_sign,
            "n_mvd": n_mvd, "n_panoptic": n_pan,
            "covered": len(features) > 0,
        }
        new_queries += 1
        if delay_s > 0:
            time.sleep(delay_s)

        if (i + 1) % 20 == 0:
            n_cov = sum(1 for v in grid_cache.values() if v.get("covered"))
            print(f"  [{i+1:,}/{len(unique_cells):,}] queried={new_queries:,}  cells_covered={n_cov:,}")
            with open(grid_cache_file, "w") as f:
                json.dump(grid_cache, f)

    if new_queries > 0:
        with open(grid_cache_file, "w") as f:
            json.dump(grid_cache, f)

    # Assign grid coverage to segments
    covered = 0
    for _, cell in unique_cells.iterrows():
        key  = f"{cell['_grid_lon']:.3f},{cell['_grid_lat']:.3f}"
        data = grid_cache.get(key, {})
        if not data.get("covered"):
            continue
        mask = (gdf["_grid_lon"] == cell["_grid_lon"]) & (gdf["_grid_lat"] == cell["_grid_lat"])
        gdf.loc[mask, "mapillary_covered"]        = True
        gdf.loc[mask, "mapillary_n_features"]     = data["n_total"]
        gdf.loc[mask, "mapillary_n_trafficsigns"] = data["n_sign"]
        gdf.loc[mask, "mapillary_n_mvd"]          = data["n_mvd"]
        base = 30.0
        sc   = min(data["n_sign"] / 5, 1.0) * 40.0
        mc   = min(data["n_mvd"] / 10, 1.0) * 20.0
        pc   = min(data.get("n_panoptic", 0) / 20, 1.0) * 10.0
        gdf.loc[mask, "infra_visibility_score"] = float(np.clip(base + sc + mc + pc, 0, 100))
        covered += mask.sum()

    # Drop temp columns
    gdf.drop(columns=["_grid_lon", "_grid_lat"], inplace=True, errors="ignore")

    total = work_mask.sum()
    print(f"\n  Mapillary coverage: {covered:,} / {total:,} segments "
          f"({100*covered/total:.1f}%)")
    if covered > 0:
        vis = gdf.loc[gdf["mapillary_covered"], "infra_visibility_score"]
        print(f"  infra_visibility_score (covered): mean={vis.mean():.1f}  p50={vis.median():.1f}")

    n_blind = ((gdf["infra_visibility_score"] == 0) & work_mask).sum()
    print(f"  Infrastructure blindspots (score=0, no Mapillary data): {n_blind:,} segments")
    return gdf


def apply_mapillary_to_scoring(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply Mapillary-derived infrastructure visibility to scoring:

    1. LOW infra_visibility_score on a HIGH-SSS segment amplifies the priority
       index — a dangerous road that is also invisible to digital monitoring
       deserves MORE urgency, not less.

    2. HIGH trafficsign_density on a segment lightly reduces the limit credibility
       sub-score penalty (well-signed roads tend to have more enforced limits).

    3. Coverage gap flag: segments with infra_visibility_score = 0 AND SSS > 50
       are tagged as "double priority" — dangerous AND unmonitored.

    Only modifies sub-scores where Mapillary was queried (all segments are
    queried when the token is provided, score=0 for uncovered ones).
    """
    gdf = gdf.copy()

    if "infra_visibility_score" not in gdf.columns:
        return gdf

    vis = gdf["infra_visibility_score"].fillna(0)
    sss = gdf.get("sss", pd.Series(np.nan, index=gdf.index)).fillna(0)

    # Tag double-priority segments: dangerous AND invisible
    gdf["mapillary_blindspot"] = (vis == 0) & (sss >= 50)
    n_blind = gdf["mapillary_blindspot"].sum()
    if n_blind:
        print(f"  Mapillary blindspots (SSS>=50 + no imagery): {n_blind:,} segments")

    # Boost priority_index for blindspot segments (if column exists)
    if "priority_index" in gdf.columns:
        mask = gdf["mapillary_blindspot"] & gdf["priority_index"].notna()
        if mask.any():
            gdf.loc[mask, "priority_index"] = (
                gdf.loc[mask, "priority_index"] * 1.10
            ).clip(0, 100)
            print(f"  Priority index boosted 10% on {mask.sum():,} blindspot segments")

    # High visibility + high sign density slightly reduces credibility_gap pressure
    if "sub_score_limit_credibility" in gdf.columns:
        mask = (vis > 60) & gdf["sub_score_limit_credibility"].notna()
        if mask.any():
            # grid-based enrichment produces mapillary_n_trafficsigns not trafficsign_density
            tc_col = "trafficsign_density" if "trafficsign_density" in gdf.columns else "mapillary_n_trafficsigns"
            sign_dens = gdf.loc[mask, tc_col].fillna(0)
            reduction = np.clip(sign_dens / 20, 0, 0.10)  # max 10% reduction
            gdf.loc[mask, "sub_score_limit_credibility"] = (
                gdf.loc[mask, "sub_score_limit_credibility"] * (1 - reduction)
            ).clip(0, 100)

    covered = gdf["mapillary_covered"].fillna(False).sum()
    print(f"  Mapillary scoring adjustments applied ({covered:,} covered, "
          f"{n_blind:,} blindspots flagged)")
    return gdf

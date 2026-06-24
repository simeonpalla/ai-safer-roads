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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

warnings.filterwarnings("ignore")

MAPILLARY_BASE = "https://graph.mapillary.com"

# Mapillary API v4 returns these three coarse object_type categories
OBJECT_TYPES = {"trafficsign", "mvd_fast", "panoptic"}

# Mapillary traffic sign taxonomy prefixes used for CV feature extraction
_SPEED_LIMIT_PREFIXES = ("regulatory--maximum-speed-limit--",
                         "complementary--maximum-speed-limit--")
_PED_CROSSING_TOKENS  = ("pedestrian-crossing", "crosswalk")
_STREET_LAMP_TOKENS   = ("street-light", "street-lamp")
_GUARDRAIL_TOKENS     = ("guardrail", "barrier-concrete", "barrier-jersey",
                         "barrier-water", "barrier-other")


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


# ── Thread workers (module-level for picklability) ────────────────────────────

def _fetch_standard_cell(args: tuple) -> tuple:
    """Worker: query one grid cell for standard map features (trafficsign/mvd/panoptic)."""
    west, south, token = args
    east  = west + _GRID_DEG
    north = south + _GRID_DEG
    features = _query_map_features((west, south, east, north), token, timeout=8)
    key = f"{west:.3f},{south:.3f}"
    n_sign = sum(1 for f in features if f.get("object_type") == "trafficsign")
    n_mvd  = sum(1 for f in features if f.get("object_type") == "mvd_fast")
    n_pan  = sum(1 for f in features if f.get("object_type") == "panoptic")
    return key, {"n_total": len(features), "n_sign": n_sign, "n_mvd": n_mvd,
                 "n_panoptic": n_pan, "covered": len(features) > 0}


def _fetch_cv_cell(args: tuple) -> tuple:
    """Worker: query one grid cell with object_value for CV feature extraction."""
    west, south, token = args
    east  = west + _GRID_DEG
    north = south + _GRID_DEG
    key   = f"{west:.3f},{south:.3f}"
    url    = f"{MAPILLARY_BASE}/map_features"
    params = {"access_token": token, "fields": "id,object_type,object_value",
              "bbox": f"{west},{south},{east},{north}"}
    features = []
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                features = resp.json().get("data", [])
                break
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            elif resp.status_code in (400, 500):
                break
        except (requests.Timeout, requests.RequestException):
            if attempt == 0:
                time.sleep(1)
    cv_data = _aggregate_cv_features(features)
    cv_data["covered"] = len(features) > 0
    return key, cv_data


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
    delay_s: float = 0.0,
    segment_mask: "pd.Series | None" = None,
    max_new_queries: int = None,
    max_workers: int = 8,
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

    pending = [
        (f"{r['_grid_lon']:.3f},{r['_grid_lat']:.3f}", r["_grid_lon"], r["_grid_lat"])
        for _, r in unique_cells.iterrows()
        if f"{r['_grid_lon']:.3f},{r['_grid_lat']:.3f}" not in grid_cache
    ]
    if max_new_queries is not None:
        pending = pending[:max_new_queries]

    new_queries = len(pending)
    if new_queries > 0:
        workers = min(max_workers, new_queries)
        print(f"  Querying {new_queries:,} new cells in parallel (workers={workers}) ...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(_fetch_standard_cell, (west, south, token)): key
                for key, west, south in pending
            }
            completed = 0
            for future in as_completed(future_map):
                key, data = future.result()
                grid_cache[key] = data
                completed += 1
                if completed % 50 == 0:
                    n_cov = sum(1 for v in grid_cache.values() if v.get("covered"))
                    print(f"  [{completed:,}/{new_queries:,}] cells_covered={n_cov:,}")
                    with open(grid_cache_file, "w") as f:
                        json.dump(grid_cache, f)

        with open(grid_cache_file, "w") as f:
            json.dump(grid_cache, f)

    # Vectorised assignment — build a string key per row once, then map from
    # cache dicts. O(n) vs the previous O(cells × rows) loop of .loc writes.
    gdf["_ck"] = (gdf["_grid_lon"].map("{:.3f}".format) + "," +
                  gdf["_grid_lat"].map("{:.3f}".format))

    vis_map = {}; ntot_map = {}; nsign_map = {}; nmvd_map = {}
    for key, data in grid_cache.items():
        if not data.get("covered"):
            continue
        ns = data.get("n_sign", 0)
        nm = data.get("n_mvd",  0)
        np_ = data.get("n_panoptic", 0)
        vis_map[key]   = float(np.clip(30 + min(ns/5,1)*40 + min(nm/10,1)*20 + min(np_/20,1)*10, 0, 100))
        ntot_map[key]  = data.get("n_total", 0)
        nsign_map[key] = ns
        nmvd_map[key]  = nm

    gdf["mapillary_covered"]        = gdf["_ck"].isin(vis_map)
    gdf["infra_visibility_score"]   = gdf["_ck"].map(vis_map).fillna(0.0)
    gdf["mapillary_n_features"]     = gdf["_ck"].map(ntot_map).fillna(0)
    gdf["mapillary_n_trafficsigns"] = gdf["_ck"].map(nsign_map).fillna(0)
    gdf["mapillary_n_mvd"]          = gdf["_ck"].map(nmvd_map).fillna(0)
    covered = int(gdf["mapillary_covered"].sum())

    gdf.drop(columns=["_ck", "_grid_lon", "_grid_lat"], inplace=True, errors="ignore")

    total = work_mask.sum()
    print(f"\n  Mapillary coverage: {covered:,} / {total:,} segments "
          f"({100*covered/total:.1f}%)")
    if covered > 0:
        vis = gdf.loc[gdf["mapillary_covered"], "infra_visibility_score"]
        print(f"  infra_visibility_score (covered): mean={vis.mean():.1f}  p50={vis.median():.1f}")

    n_blind = ((gdf["infra_visibility_score"] == 0) & work_mask).sum()
    print(f"  Infrastructure blindspots (score=0, no Mapillary data): {n_blind:,} segments")
    return gdf


# ── CV feature helpers ────────────────────────────────────────────────────────

def _parse_speed_kmh(value: str) -> int | None:
    """Extract speed km/h from Mapillary object_value taxonomy string.
    'regulatory--maximum-speed-limit--90--g1' → 90
    Returns None if value does not contain a speed limit.
    """
    if not value:
        return None
    v = value.lower()
    for prefix in _SPEED_LIMIT_PREFIXES:
        if prefix in v:
            tail = v.split(prefix)[-1].split("--")[0]
            try:
                speed = int(tail)
                if 5 <= speed <= 140:
                    return speed
            except ValueError:
                pass
    return None


def _aggregate_cv_features(features: list) -> dict:
    """Parse object_value from Mapillary map_features response.

    object_value is available in the public Mapillary v4 API when
    fields=id,object_type,object_value is requested. If the API does not
    return it (e.g. enterprise-only restriction), all counts default to 0.
    """
    from collections import Counter
    speeds, ped, lamp, guard = [], 0, 0, 0

    for f in features:
        val = (f.get("object_value") or f.get("value") or "").lower()
        if not val:
            continue
        spd = _parse_speed_kmh(val)
        if spd:
            speeds.append(spd)
        if any(t in val for t in _PED_CROSSING_TOKENS):
            ped += 1
        if any(t in val for t in _STREET_LAMP_TOKENS):
            lamp += 1
        if any(t in val for t in _GUARDRAIL_TOKENS):
            guard += 1

    detected_speed = Counter(speeds).most_common(1)[0][0] if speeds else None
    return {
        "detected_speed": detected_speed,
        "n_speed_signs":  len(speeds),
        "n_ped_crossing": ped,
        "n_street_lamp":  lamp,
        "n_guardrail":    guard,
    }


def enrich_with_mapillary_cv(
    gdf: gpd.GeoDataFrame,
    token: str = None,
    cache_dir: str = "enrichment_data/mapillary_cache",
    delay_s: float = 0.0,
    segment_mask: "pd.Series | None" = None,
    max_new_queries: int = None,
    max_workers: int = 8,
) -> gpd.GeoDataFrame:
    """Query Mapillary map_features with object_value to extract CV detections.

    Uses a SEPARATE cache (cv_grid_cache.json) from enrich_with_mapillary()
    so the standard feature cache is not invalidated.

    Designed to run AFTER enrich_with_mapillary() — targeted to high-risk
    segments (Critical + High Risk) by default via segment_mask.

    New columns:
      mapillary_detected_speed  — modal detected speed limit sign (km/h), NaN if none
      mapillary_speed_mismatch  — |detected_speed − posted_limit|, NaN if no sign
      mapillary_ped_crossing    — pedestrian crossing detections in grid cell
      mapillary_street_lamp     — street lamp detections in grid cell
      mapillary_guardrail       — guardrail/barrier detections in grid cell
    """
    if token is None:
        token = os.environ.get("MAPILLARY_TOKEN", "")
    if not token:
        print("  [CV] No Mapillary token — skipping CV enrichment")
        return gdf

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cv_cache_file = cache_path / "cv_grid_cache.json"

    cv_cache = {}
    if cv_cache_file.exists():
        with open(cv_cache_file) as f:
            cv_cache = json.load(f)

    gdf = gdf.copy()
    for col in ["mapillary_detected_speed", "mapillary_speed_mismatch",
                "mapillary_ped_crossing", "mapillary_street_lamp", "mapillary_guardrail"]:
        gdf[col] = np.nan

    bounds = gdf["geometry"].bounds
    midx = (bounds["minx"] + bounds["maxx"]) / 2
    midy = (bounds["miny"] + bounds["maxy"]) / 2
    gdf["_cv_lon"] = (midx / _GRID_DEG).astype(int) * _GRID_DEG
    gdf["_cv_lat"] = (midy / _GRID_DEG).astype(int) * _GRID_DEG

    work_mask = (segment_mask.reindex(gdf.index, fill_value=False)
                 if segment_mask is not None
                 else pd.Series(True, index=gdf.index))

    unique_cells = gdf.loc[work_mask, ["_cv_lon", "_cv_lat"]].drop_duplicates()
    n_new = sum(
        1 for _, r in unique_cells.iterrows()
        if f"{r['_cv_lon']:.3f},{r['_cv_lat']:.3f}" not in cv_cache
    )
    n_will = min(n_new, max_new_queries) if max_new_queries is not None else n_new
    print(f"\n  [CV] Grid cells: {len(unique_cells):,}  ({len(cv_cache):,} cached, {n_new:,} new)")
    if n_new > 0:
        est = n_will * (delay_s + 0.5) / 60
        print(f"  [CV] Querying {n_will:,} new cells — est. {est:.1f} min")

    pending = [
        (f"{r['_cv_lon']:.3f},{r['_cv_lat']:.3f}", r["_cv_lon"], r["_cv_lat"])
        for _, r in unique_cells.iterrows()
        if f"{r['_cv_lon']:.3f},{r['_cv_lat']:.3f}" not in cv_cache
    ]
    if max_new_queries is not None:
        pending = pending[:max_new_queries]

    new_queries = len(pending)
    if new_queries > 0:
        workers = min(max_workers, new_queries)
        est = new_queries / (workers * 2)  # rough: 2 cells/sec/worker
        print(f"  [CV] Querying {new_queries:,} cells in parallel (workers={workers}) "
              f"— est. {est:.0f}s")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(_fetch_cv_cell, (west, south, token)): key
                for key, west, south in pending
            }
            completed = 0
            for future in as_completed(future_map):
                key, cv_data = future.result()
                cv_cache[key] = cv_data
                completed += 1
                if completed % 50 == 0:
                    print(f"  [CV] {completed:,}/{new_queries:,} complete")
                    with open(cv_cache_file, "w") as f:
                        json.dump(cv_cache, f)

        with open(cv_cache_file, "w") as f:
            json.dump(cv_cache, f)

    # Vectorised assignment
    gdf["_ck"] = (gdf["_cv_lon"].map("{:.3f}".format) + "," +
                  gdf["_cv_lat"].map("{:.3f}".format))

    spd_map = {}; ped_map = {}; lamp_map = {}; guard_map = {}
    for key, data in cv_cache.items():
        if not data:
            continue
        ds = data.get("detected_speed")
        if ds is not None:
            spd_map[key] = float(ds)
        ped_map[key]   = float(data.get("n_ped_crossing", 0))
        lamp_map[key]  = float(data.get("n_street_lamp",  0))
        guard_map[key] = float(data.get("n_guardrail",    0))

    gdf["mapillary_detected_speed"] = gdf["_ck"].map(spd_map)
    gdf["mapillary_ped_crossing"]   = gdf["_ck"].map(ped_map)
    gdf["mapillary_street_lamp"]    = gdf["_ck"].map(lamp_map)
    gdf["mapillary_guardrail"]      = gdf["_ck"].map(guard_map)
    if "speed_limit" in gdf.columns and spd_map:
        gdf["mapillary_speed_mismatch"] = (
            gdf["speed_limit"] - gdf["mapillary_detected_speed"]
        ).abs()

    gdf.drop(columns=["_ck", "_cv_lon", "_cv_lat"], inplace=True, errors="ignore")

    n_speed = gdf["mapillary_detected_speed"].notna().sum()
    n_ped   = (gdf["mapillary_ped_crossing"].fillna(0) > 0).sum()
    n_lamp  = (gdf["mapillary_street_lamp"].fillna(0) > 0).sum()
    print(f"  [CV] Speed signs: {n_speed:,} segments  |  Ped crossings: {n_ped:,}  |  "
          f"Street lamps: {n_lamp:,}")
    if n_speed == 0 and new_queries > 0:
        print("  [CV] Note: object_value returned no speed data — "
              "public API may not expose this field. CV columns will be NaN.")
    return gdf


def apply_mapillary_cv_to_scoring(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply CV-derived Mapillary detections to scoring sub-scores.

    Called separately from apply_mapillary_to_scoring() to avoid double-
    applying the standard visibility adjustments.

    Adjustments:
    1. Speed sign mismatch: if detected sign differs from posted by >15 km/h,
       amplify the credibility gap sub-score by up to 15%.
    2. Pedestrian crossing detected: amplify VRU sub-score by 10%.
    3. Street lamp detected + osm_lit missing/unknown: treat road as lit
       (synthesise osm_lit='yes') for downstream use.
    """
    gdf = gdf.copy()

    if "mapillary_speed_mismatch" not in gdf.columns:
        return gdf

    mismatch = gdf["mapillary_speed_mismatch"].fillna(0)
    ped      = gdf["mapillary_ped_crossing"].fillna(0)
    lamp     = gdf["mapillary_street_lamp"].fillna(0)

    # 1. Speed sign mismatch → credibility amplifier (max +15%)
    if "sub_score_limit_credibility" in gdf.columns:
        high_mismatch = mismatch > 15
        if high_mismatch.any():
            boost = np.clip((mismatch - 15) / 20, 0, 0.15)  # 0 at 15 km/h → 0.15 at 35+ km/h
            gdf.loc[high_mismatch, "sub_score_limit_credibility"] = (
                gdf.loc[high_mismatch, "sub_score_limit_credibility"] * (1 + boost[high_mismatch])
            ).clip(0, 100)
            n = int(high_mismatch.sum())
            print(f"  [CV] Credibility amplified on {n:,} segments (sign mismatch >15 km/h)")

    # 2. Pedestrian crossing detected → VRU amplifier (+10%)
    if "sub_score_vru_risk" in gdf.columns:
        has_ped = ped > 0
        if has_ped.any():
            gdf.loc[has_ped, "sub_score_vru_risk"] = (
                gdf.loc[has_ped, "sub_score_vru_risk"] * 1.10
            ).clip(0, 100)
            print(f"  [CV] VRU score amplified on {int(has_ped.sum()):,} segments "
                  f"(pedestrian crossing detected)")

    # 3. Street lamp detected → synthesise lit flag where osm_lit is unknown
    if "osm_lit" in gdf.columns:
        unknown_lit = gdf["osm_lit"].isna() | gdf["osm_lit"].isin(["", "unknown"])
        synth = unknown_lit & (lamp > 0)
        if synth.any():
            gdf.loc[synth, "osm_lit"] = "yes"
            print(f"  [CV] osm_lit synthesised as 'yes' on {int(synth.sum()):,} segments "
                  f"(street lamp detected, osm_lit was unknown)")

    # Recompute SSS if sub-scores were modified
    if "sub_score_limit_credibility" in gdf.columns:
        from scoring import WEIGHTS
        from config import SCORE_BANDS
        mask = gdf["scoreable"] & gdf["sss"].notna()
        if mask.any():
            gdf.loc[mask, "sss"] = (
                WEIGHTS["speed_limit_alignment"]  * gdf.loc[mask, "sub_score_limit_alignment"] +
                WEIGHTS["limit_credibility_gap"]  * gdf.loc[mask, "sub_score_limit_credibility"] +
                WEIGHTS["vru_context_risk"]        * gdf.loc[mask, "sub_score_vru_risk"]
            ).clip(0, 100)
            def _band(s):
                for name, (lo, hi) in SCORE_BANDS.items():
                    if lo <= s <= hi:
                        return name
                return "Acceptable"
            gdf.loc[mask, "sss_band"] = gdf.loc[mask, "sss"].map(_band)

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

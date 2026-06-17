"""
enrichment.py v4 — Exposure Score using spatial buffers (not centroids).

CHANGES from v3 (per Priority Index reviewer feedback, June 2026):
  - FIXED: population and traffic_volume now use percentile-rank
    normalization (within country_code) instead of log1p/divide-by-max,
    which was anchoring the whole 0–1 scale to a single extreme outlier
    and compressing most segments into a narrow band (observed on real
    data: Exposure mean=49.6, max only 93.1 across 15,121 segments — see
    _percentile_normalize() docstring for the full explanation).
  - NEW: match_road_infrastructure() attaches real OSM way tags (lanes,
    oneway, surface, lit, junction) to each road segment via nearest-way
    spatial join. This is consumed by priority_scoring.py's Severity layer
    to replace the static ROAD_CLASS_SEVERITY_MAP assumption with observed
    facts where a match exists, falling back to the assumption otherwise.
    Requires running extract_osm_data.py first to generate
    enrichment_data/road_infra/road_infra_{MH,TH}.geojson.

CHANGES from v2:
  - WeightedSample (traffic volume) folded into the Exposure Score, not just
    used downstream in advanced_scoring.lives_saved(). This is also now the
    most-weighted component, since it's reliably populated for ~100% of
    scoreable segments unlike the optional OSM/WorldPop layers.
  - Any exposure component with zero signal (no enrichment_data/ files
    present) is dropped and its weight redistributed, with a printed
    warning, instead of silently producing exposure_score=0 everywhere.

CHANGES from v1:
  - Proximity uses road geometry buffer, not segment centroid
    A 10km road passing near a school is correctly detected
  - Added intersection density from OSM PBF extract
  - Combined into single Exposure Score (population + intersections + sensitive sites)
  - Removed download_worldpop / download_osm params (local files only)

EXPOSURE SCORE formula (weights in config.EXPOSURE_WEIGHTS):
  = 0.35 × traffic_volume_score (WeightedSample, percentile-normalized within country)
  + 0.25 × population_score     (WorldPop density, percentile-normalized within country)
  + 0.20 × intersection_score   (junctions per km, normalized)
  + 0.12 × school_score         (within 500m buffer)
  + 0.08 × hospital_score       (within 750m buffer)

This is used two ways downstream:
  1. Legacy: priority_score = SSS × (1 + 0.20 × exposure_norm)   — see enrich_segments()
  2. New:    one of the three Exposure×Likelihood×Severity inputs to the
             Priority Index — see priority_scoring.py. Both are kept side by
             side so you can compare before deciding which to use.
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore")

SCHOOL_BUFFER_M    = 500
HOSPITAL_BUFFER_M  = 750
INTERSECTION_BUFFER_M = 1000
WORLDPOP_BUFFER_M  = 500


def _load_amenities(folder: str, label: str) -> gpd.GeoDataFrame:
    """Load all shapefiles/GeoJSON from folder into one GeoDataFrame."""
    folder = Path(folder)
    if not folder.exists():
        print(f"  [{label}] Folder not found: {folder}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    files = list(folder.glob("**/*.shp")) + list(folder.glob("**/*.geojson"))
    if not files:
        print(f"  [{label}] No files in {folder}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    parts = []
    for f in files:
        try:
            gdf = gpd.read_file(f)
            gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:4326")
            parts.append(gdf[["geometry"]])
            print(f"    Loaded {len(gdf):,} {label} from {f.name}")
        except Exception as e:
            print(f"    Failed {f.name}: {e}")

    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    combined = combined[combined.geometry.geom_type.isin(["Point", "MultiPoint"])]
    print(f"  [{label}] Total: {len(combined):,} points")
    return combined


def _nearest_distance(
    gdf: gpd.GeoDataFrame,
    amenities: gpd.GeoDataFrame,
    label: str,
    max_dist_m: float = 50_000,
) -> pd.Series:
    """
    Actual distance (metres) from each road segment to its NEAREST amenity
    point, uncapped by any buffer radius. This is what calculation
    transparency needs: "near_school: True" tells a reviewer nothing about
    HOW near; "nearest school: 320m" does. Complements (does not replace)
    the buffer-intersect boolean, which is still used for printed coverage
    stats and as a simple yes/no popup line.

    max_dist_m is a sanity cap, not a buffer — distances beyond it are
    treated as "no usable nearby data" (NaN) rather than a misleadingly
    precise number for a region with no loaded amenity points at all.
    """
    if amenities is None or len(amenities) == 0:
        return pd.Series(np.nan, index=gdf.index)

    gdf_m  = gdf.to_crs(epsg=3857)
    amen_m = amenities.to_crs(epsg=3857)

    # Distance to nearest point from the segment's CENTROID — a buffer-vs-
    # geometry intersection test (used for near_school/near_hospital) isn't
    # the right basis for "how far is it", a point-to-point distance is.
    centroids = gpd.GeoDataFrame(
        {"orig_idx": gdf.index},
        geometry=gdf_m.geometry.centroid.values,
        crs=3857,
    ).reset_index(drop=True)

    try:
        joined = gpd.sjoin_nearest(
            centroids,
            amen_m[["geometry"]].reset_index(drop=True),
            max_distance=max_dist_m,
            distance_col="dist_m",
        )
    except Exception as e:
        print(f"  [{label}] Nearest-distance match failed ({e})")
        return pd.Series(np.nan, index=gdf.index)

    joined = joined.sort_values("dist_m").drop_duplicates(subset="orig_idx").set_index("orig_idx")
    dist = pd.Series(np.nan, index=gdf.index)
    dist.loc[joined.index] = joined["dist_m"]
    return dist


def _proximity_decay_score(dist_m: pd.Series, buffer_m: float) -> pd.Series:
    """
    Continuous 0-1 proximity score from a real distance, replacing the
    previous binary near_school/near_hospital (in/out of buffer) as the
    EXPOSURE FORMULA's input. A school 10m away and a school 490m away
    were previously scored identically ("within buffer" = 1.0); a school
    510m away was scored identically to one 50km away ("outside buffer" =
    0.0). Linear decay to 0 at 2x the buffer radius is a more honest
    reflection of exposure than a hard step function, while still using
    the same buffer_m as the meaningful reference distance.
    Missing distance (no amenity data loaded) → 0, not NaN, so it doesn't
    break the weighted-average exposure formula; compute_exposure_score
    already redistributes weight away from components with no signal at
    all (see its docstring) when the WHOLE column is unavailable.
    """
    return (1 - (dist_m.fillna(2 * buffer_m) / (2 * buffer_m))).clip(0, 1)


def _buffer_spatial_join(
    gdf: gpd.GeoDataFrame,
    amenities: gpd.GeoDataFrame,
    buffer_m: float,
    label: str,
) -> pd.Series:
    """
    Buffer each road GEOMETRY (not centroid) by buffer_m metres,
    then spatial join to find which segments intersect an amenity.
    Returns boolean Series.
    """
    if amenities is None or len(amenities) == 0:
        return pd.Series(False, index=gdf.index)

    gdf_m  = gdf.to_crs(epsg=3857)
    amen_m = amenities.to_crs(epsg=3857)

    # Buffer the full road geometry
    buf = gpd.GeoDataFrame(
        {"orig_idx": gdf.index},
        geometry=gdf_m.geometry.buffer(buffer_m).values,
        crs=3857,
    ).reset_index(drop=True)

    joined = gpd.sjoin(
        buf,
        amen_m[["geometry"]].reset_index(drop=True),
        how="left",
        predicate="intersects",
    )
    hits  = set(joined.loc[joined["index_right"].notna(), "orig_idx"])
    mask  = pd.Series(False, index=gdf.index)
    mask[list(hits)] = True
    print(f"  [{label}] {mask.sum():,} road segments within {buffer_m:.0f}m buffer")
    return mask


def _load_intersections(data_dir: str) -> gpd.GeoDataFrame:
    """
    Load intersection points extracted by extract_osm_data.py.
    Falls back gracefully if file not found.
    """
    candidates = [
        Path(data_dir) / "intersections" / "intersections_MH.geojson",
        Path(data_dir) / "intersections" / "intersections_TH.geojson",
        Path(data_dir) / "intersections_combined.geojson",
    ]
    parts = []
    for p in candidates:
        if p.exists():
            try:
                gdf = gpd.read_file(p)
                parts.append(gdf[["geometry"]])
                print(f"    Loaded {len(gdf):,} intersections from {p.name}")
            except Exception:
                pass
    if not parts:
        print("  [Intersections] No intersection files found — skipping")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    return combined


def _load_road_infra(data_dir: str) -> gpd.GeoDataFrame:
    """Load OSM road infrastructure ways extracted by extract_osm_data.py."""
    candidates = [
        Path(data_dir) / "road_infra" / "road_infra_MH.geojson",
        Path(data_dir) / "road_infra" / "road_infra_TH.geojson",
    ]
    parts = []
    for p in candidates:
        if p.exists():
            try:
                infra = gpd.read_file(p)
                parts.append(infra)
                print(f"    Loaded {len(infra):,} road ways from {p.name}")
            except Exception as e:
                print(f"    Failed {p.name}: {e}")
    if not parts:
        print("  [Road Infra] No road_infra files found — Severity will fall "
              "back to the ROAD_CLASS_SEVERITY_MAP assumption for all segments. "
              "Run extract_osm_data.py to generate these from your PBF files.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    return combined


def match_road_infrastructure(
    gdf: gpd.GeoDataFrame,
    data_dir: str,
    max_dist_m: float = 30,
) -> gpd.GeoDataFrame:
    """
    Match each ADB road segment to its nearest OSM way (within max_dist_m)
    and attach the OSM way's infrastructure tags: lanes, oneway, surface,
    lit, junction. These feed priority_scoring.score_infrastructure_severity()
    as OBSERVED facts about the road — replacing the static
    ROAD_CLASS_SEVERITY_MAP assumption ("trunk roads are probably
    undivided") with what the road actually is, where a match exists.

    ADB segment geometry and OSM way geometry come from different source
    datasets and aren't guaranteed to align exactly, so this is a nearest-
    neighbour match capped at max_dist_m rather than an exact intersection.
    Segments with no OSM way within range are left with NaN tags, and
    priority_scoring falls back to the road-class assumption for them —
    same graceful-degradation pattern as every other enrichment layer here.
    """
    gdf = gdf.copy()
    gdf["osm_lanes"] = np.nan
    for c in ["osm_oneway", "osm_surface", "osm_lit", "osm_junction"]:
        gdf[c] = pd.Series(pd.NA, index=gdf.index, dtype="object")

    infra = _load_road_infra(data_dir)
    if len(infra) == 0:
        return gdf

    gdf_m   = gdf.to_crs(epsg=3857)
    infra_m = infra.to_crs(epsg=3857)
    keep_cols = [c for c in ["lanes", "oneway", "surface", "lit", "junction"] if c in infra_m.columns]

    try:
        joined = gpd.sjoin_nearest(
            gdf_m[["geometry"]].reset_index().rename(columns={"index": "orig_idx"}),
            infra_m[["geometry"] + keep_cols],
            max_distance=max_dist_m,
            distance_col="dist_m",
        )
    except Exception as e:
        print(f"  [Road Infra] Spatial match failed ({e}) — falling back to "
              f"road-class assumption for all segments")
        return gdf

    if len(joined) == 0:
        print(f"  [Road Infra] No segments matched an OSM way within {max_dist_m:.0f}m")
        return gdf

    # Keep only the single nearest match per segment (ties/duplicates dropped)
    joined = joined.sort_values("dist_m").drop_duplicates(subset="orig_idx").set_index("orig_idx")

    for c in keep_cols:
        gdf.loc[joined.index, f"osm_{c}"] = joined[c]

    n_matched = len(joined)
    print(f"  [Road Infra] {n_matched:,} / {len(gdf):,} segments matched to an "
          f"OSM way within {max_dist_m:.0f}m ({100*n_matched/max(len(gdf),1):.1f}%)")
    return gdf


def score_intersection_density(
    gdf: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    buffer_m: float = INTERSECTION_BUFFER_M,
) -> pd.Series:
    """
    Count intersections within buffer_m of each road geometry.
    Normalize by road length (intersections per km).
    Score 0–100: 0=isolated road, 100=dense urban junction network.
    """
    result = pd.Series(0.0, index=gdf.index)
    if len(intersections) == 0:
        return result

    gdf_m  = gdf.to_crs(epsg=3857)
    int_m  = intersections.to_crs(epsg=3857)

    # Road length in km
    road_len_km = gdf_m.geometry.length / 1000
    road_len_km = road_len_km.replace(0, 0.1)  # avoid division by zero

    buf_gdf = gpd.GeoDataFrame(
        {"orig_idx": gdf.index, "len_km": road_len_km.values},
        geometry=gdf_m.geometry.buffer(buffer_m).values,
        crs=3857,
    ).reset_index(drop=True)

    joined = gpd.sjoin(
        buf_gdf,
        int_m[["geometry"]].reset_index(drop=True),
        how="left",
        predicate="intersects",
    )
    counts = joined.groupby("orig_idx").size().reindex(gdf.index, fill_value=0)
    lens   = road_len_km

    ints_per_km = counts / lens
    # Normalize: 0 = 0, 5+ per km = 100
    result = (ints_per_km / 5.0).clip(0, 1) * 100
    print(f"  [Intersections] Mean density: {ints_per_km.mean():.2f}/km, "
          f"score mean: {result.mean():.1f}")
    return result


def sample_worldpop(
    gdf: gpd.GeoDataFrame,
    data_dir: str,
) -> pd.Series:
    """Sample WorldPop density along road geometry buffer."""
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
    except ImportError:
        print("  [WorldPop] pip install rasterio")
        return pd.Series(np.nan, index=gdf.index)

    tif_candidates = {
        "MH": [
            Path(data_dir) / "worldpop_MH_2020_1km.tif",
            Path(data_dir).parent / "openSourceData" / "ind_ppp_2020.tif",
        ],
        "TH": [
            Path(data_dir) / "worldpop_TH_2020_1km.tif",
            Path(data_dir).parent / "openSourceData" / "tha_ppp_2020.tif",
        ],
    }

    result = pd.Series(np.nan, index=gdf.index)
    gdf_m  = gdf.to_crs(epsg=3857)

    for cc, candidates in tif_candidates.items():
        tif_path = next((p for p in candidates if p.exists()), None)
        if tif_path is None:
            print(f"  [WorldPop] No TIF found for {cc}")
            continue

        sub_idx = gdf[gdf["country_code"] == cc].index
        print(f"  [WorldPop] {cc}: sampling {len(sub_idx):,} segments from {tif_path.name}...")

        with rasterio.open(tif_path) as src:
            for idx in sub_idx:
                try:
                    # Buffer road GEOMETRY (not centroid)
                    buf    = gdf_m.loc[idx, "geometry"].buffer(WORLDPOP_BUFFER_M)
                    buf_wgs= gpd.GeoSeries([buf], crs=3857).to_crs(src.crs).iloc[0]
                    out, _ = rio_mask(src, [buf_wgs], crop=True, nodata=src.nodata)
                    vals   = out.flatten()
                    vals   = vals[(vals != src.nodata) & (vals > 0)]
                    result[idx] = float(vals.mean()) if len(vals) > 0 else 0.0
                except Exception:
                    pass

        sampled = result[sub_idx].notna().sum()
        print(f"    → {sampled:,} / {len(sub_idx):,} sampled")

    return result


def _percentile_normalize(series: pd.Series, group: pd.Series = None) -> pd.Series:
    """
    Percentile-rank normalization to 0–1.

    BUG FIX (reviewer feedback, June 2026): the previous log1p(x)/log1p(max)
    scheme anchors the whole 0–1 scale to a single extreme outlier (e.g. one
    very dense WorldPop pixel at 67,357 ppl/km² in the real run). Since log
    compression means most "ordinary" values are already a large fraction
    of log(max), this pushed the bulk of segments toward the high end of
    the scale regardless of their real relative exposure, compressing the
    Exposure Score distribution (observed: mean 49.6, max only 93.1 across
    15,121 real segments — barely any segment used the top of the range,
    and most segments ended up clustered in a narrow middle band instead of
    spreading across it). Percentile rank guarantees a genuinely
    discriminative spread for any column with real variation, and is
    immune to a single outlier setting the scale for everyone else.

    If group is provided, ranks are computed SEPARATELY within each group
    (e.g. country_code) rather than globally — otherwise a systematic scale
    difference between countries (e.g. different GPS probe density/
    collection methodology behind WeightedSample, or different WorldPop
    raster resolution) would bias one country's roads to score uniformly
    higher than the other's regardless of their actual relative exposure
    within that country.
    """
    s = series.fillna(0)
    if group is None:
        if s.nunique() <= 1:
            return pd.Series(0.0, index=series.index)
        return s.rank(pct=True, method="average")

    result = pd.Series(0.0, index=series.index)
    for _, idx in group.fillna("unknown").groupby(group.fillna("unknown")).groups.items():
        sub = s.loc[idx]
        result.loc[idx] = 0.0 if sub.nunique() <= 1 else sub.rank(pct=True, method="average")
    return result


def compute_exposure_score(
    pop: pd.Series,
    intersections: pd.Series,
    schools: pd.Series,
    hospitals: pd.Series,
    traffic_volume: pd.Series = None,
    country_code: pd.Series = None,
) -> pd.Series:
    """
    Combine all exposure layers into a single 0–100 Exposure Score.

    Weights come from config.EXPOSURE_WEIGHTS. traffic_volume (WeightedSample,
    the ADB GPS probe-count proxy) is folded in here per reviewer feedback —
    "ADB clearly invested effort in building WeightedSample... yet your
    current score barely uses it." It's also the one exposure input that's
    reliably populated for ~100% of scoreable segments, unlike
    population/intersections/schools/hospitals, which depend on optional
    local enrichment files (extract_osm_data.py, WorldPop tif) that may not
    exist in every run.

    population and traffic_volume are percentile-rank normalized (see
    _percentile_normalize, within country_code if provided) rather than
    log/divide-by-max, specifically to avoid the score-compression issue
    flagged in review — see that function's docstring.

    If a component has NO signal at all (e.g. the enrichment_data/ folder is
    empty so every segment gets population=0, intersections=0), it is
    dropped and its weight is redistributed proportionally across the
    remaining components — rather than silently producing an Exposure Score
    of 0 for every segment, which would make the downstream Priority Index
    meaningless without anyone noticing.
    """
    from config import EXPOSURE_WEIGHTS

    pop_norm = _percentile_normalize(pop, group=country_code)
    int_norm  = (intersections.fillna(0) / 100).clip(0, 1)
    # schools/hospitals are now CONTINUOUS 0-1 distance-decay scores (see
    # _proximity_decay_score), not the old in/out-of-buffer booleans —
    # already 0-1, just need NaN→0 for segments with no amenity data at all.
    sch_norm  = schools.astype(float).fillna(0)
    hosp_norm = hospitals.astype(float).fillna(0)

    components = {
        "population":    pop_norm,
        "intersections":  int_norm,
        "schools":        sch_norm,
        "hospitals":      hosp_norm,
    }

    if traffic_volume is not None:
        tv_norm = _percentile_normalize(traffic_volume, group=country_code)
        components["traffic_volume"] = tv_norm

    # Drop components with literally no signal (every value 0) — a real
    # all-zero exposure layer would be suspicious; far more likely it's an
    # enrichment file that was never built.
    active  = {k: v for k, v in components.items() if not (v.fillna(0) == 0).all()}
    dropped = sorted(set(components) - set(active))
    if dropped:
        print(f"  [Exposure] No signal for: {', '.join(dropped)} — "
              f"weight redistributed across remaining components")

    weights = {k: EXPOSURE_WEIGHTS.get(k, 0) for k in active}
    total_w = sum(weights.values())
    if total_w <= 0:
        print("  [Exposure] WARNING: no exposure signal available at all — "
              "Exposure Score (and anything that multiplies by it, like "
              "Priority Index) will be 0 for every segment. Add WorldPop/OSM "
              "files to enrichment_data/, or check that weighted_sample is "
              "present in your dataset.")
        return pd.Series(0.0, index=pop.index)

    exposure = sum(active[k] * (weights[k] / total_w) for k in active) * 100
    return exposure.clip(0, 100)


def enrich_segments(
    gdf: gpd.GeoDataFrame,
    data_dir: str = "enrichment_data",
) -> gpd.GeoDataFrame:
    """
    Main entry point.

    Adds columns:
        pop_density_500m         — WorldPop along road buffer
        near_school               — bool: road within 500m of school (display only)
        near_hospital             — bool: road within 750m of hospital (display only)
        dist_to_school_m          — distance to NEAREST school, metres (calc. transparency)
        dist_to_hospital_m        — distance to NEAREST hospital, metres (calc. transparency)
        school_proximity_score    — 0–100: continuous decay, feeds Exposure
        hospital_proximity_score  — 0–100: continuous decay, feeds Exposure
        intersection_score        — 0–100: junction density
        exposure_score            — 0–100: composite exposure
        priority_score            — SSS × (1 + 0.2 × exposure_norm): final priority
    """
    print("\n" + "=" * 60)
    print("  EXPOSURE ENRICHMENT — LOCAL OPEN DATA")
    print("=" * 60)
    print(f"  Data directory: {Path(data_dir).absolute()}")
    Path(data_dir).mkdir(exist_ok=True)
    gdf = gdf.copy()

    # 1. WorldPop
    print("\n[1/5] Population density (WorldPop)...")
    pop = sample_worldpop(gdf, data_dir)
    gdf["pop_density_500m"] = pop
    if pop.notna().any():
        print(f"  Mean: {pop.mean():.0f} ppl/km²  Max: {pop.max():.0f} ppl/km²")

    # 2. Schools
    print("\n[2/5] Schools (HOTOSM)...")
    schools_gdf = _load_amenities(f"{data_dir}/schools", "Schools")
    near_school = _buffer_spatial_join(gdf, schools_gdf, SCHOOL_BUFFER_M, "Schools")
    gdf["near_school"] = near_school
    dist_school = _nearest_distance(gdf, schools_gdf, "Schools")
    gdf["dist_to_school_m"] = dist_school.round(0)
    school_proximity = _proximity_decay_score(dist_school, SCHOOL_BUFFER_M)
    if dist_school.notna().any():
        print(f"  [Schools] Median distance to nearest school: {dist_school.median():.0f}m")

    # 3. Hospitals
    print("\n[3/5] Hospitals (HOTOSM)...")
    hosp_gdf = _load_amenities(f"{data_dir}/hospitals", "Hospitals")
    near_hosp = _buffer_spatial_join(gdf, hosp_gdf, HOSPITAL_BUFFER_M, "Hospitals")
    gdf["near_hospital"] = near_hosp
    dist_hosp = _nearest_distance(gdf, hosp_gdf, "Hospitals")
    gdf["dist_to_hospital_m"] = dist_hosp.round(0)
    hospital_proximity = _proximity_decay_score(dist_hosp, HOSPITAL_BUFFER_M)
    if dist_hosp.notna().any():
        print(f"  [Hospitals] Median distance to nearest hospital: {dist_hosp.median():.0f}m")

    # 4. Intersections
    print("\n[4/5] Intersection density (OSM extract)...")
    intersections_gdf = _load_intersections(data_dir)
    int_score = score_intersection_density(gdf, intersections_gdf)
    gdf["intersection_score"] = int_score

    # 5. Road infrastructure (lanes/oneway/surface/lit/junction) — feeds
    # priority_scoring's Severity layer, not Exposure. Run here because all
    # the OSM-loading/spatial-join machinery already lives in this module.
    print("\n[5/5] Road infrastructure tags (OSM extract)...")
    gdf = match_road_infrastructure(gdf, data_dir)

    # Composite Exposure Score
    # Schools/hospitals now use CONTINUOUS distance-decay scores
    # (school_proximity / hospital_proximity) rather than the raw in/out-
    # of-buffer booleans — a school 10m away and one 490m away were
    # previously scored identically. near_school/near_hospital booleans
    # are kept as columns for the simple popup yes/no line and the printed
    # buffer-coverage stat above, but no longer feed the Exposure formula.
    gdf["school_proximity_score"]   = (school_proximity * 100).round(1)
    gdf["hospital_proximity_score"] = (hospital_proximity * 100).round(1)

    traffic_volume = gdf["weighted_sample"] if "weighted_sample" in gdf.columns else None
    if traffic_volume is None:
        print("  [Exposure] weighted_sample column not found — traffic volume "
              "excluded from Exposure Score (weight redistributed)")
    country_code = gdf["country_code"] if "country_code" in gdf.columns else None
    exposure = compute_exposure_score(pop, int_score, school_proximity, hospital_proximity,
                                       traffic_volume, country_code)
    gdf["exposure_score"] = exposure

    # Store each normalized component (0-100) for CALCULATION TRANSPARENCY
    # in the map popup — so "Exposure: 57.0" isn't a black box, every input
    # that produced it is visible and labelled with its weight/basis.
    gdf["exposure_component_population"] = (_percentile_normalize(pop, group=country_code) * 100).round(1)
    if traffic_volume is not None:
        gdf["exposure_component_traffic"] = (_percentile_normalize(traffic_volume, group=country_code) * 100).round(1)

    # Priority Score = SSS × exposure amplifier
    if "sss" in gdf.columns:
        exp_norm = exposure / 100
        gdf["priority_score"] = (
            gdf["sss"] * (1 + 0.20 * exp_norm)
        ).clip(0, 100)
        mask = gdf["scoreable"] & gdf["sss"].notna()
        print(f"\n  Exposure Score: mean={exposure[mask].mean():.1f}, max={exposure[mask].max():.1f}")
        print(f"  Priority Score: mean={gdf.loc[mask,'priority_score'].mean():.1f}, "
              f"max={gdf.loc[mask,'priority_score'].max():.1f}")
        print(f"  (Priority = SSS boosted by up to 20% for high-exposure roads)")

    print("=" * 60)
    return gdf

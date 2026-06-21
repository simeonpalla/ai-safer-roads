"""
viirs_features.py — Nighttime lights enrichment from VIIRS/Black Marble satellite data.

WHY THIS MATTERS:
  Road safety datasets capture the physical road and its posted limits.
  They do not capture WHO is on the road, especially at night.

  In South and Southeast Asia, informal commercial activity — roadside markets,
  stalls, informal loading zones — typically operates at night or early morning.
  These locations have disproportionate pedestrian exposure: people crossing
  poorly-lit roads between stalls, vendors unloading on the carriageway,
  no formal pedestrian infrastructure. But none of this appears in land-use tags.

  VIIRS (Visible Infrared Imaging Radiometer Suite) measures radiance from
  space at night. The annual Black Marble composite strips out moonlight and
  atmospheric effects to isolate human-produced light. High radiance in an
  otherwise rural area → informal activity → elevated nighttime VRU exposure.

  This module adds a `ntl_radiance` column (raw) and a normalised
  `ntl_exposure_score` (0–100) that feeds into the VRU risk sub-score as
  an independent exposure weight.

SATELLITE SOURCE:
  NASA Black Marble VNP46A4 — Annual Composite, 500m resolution.
  Free download (NASA Earthdata login required):
    https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A4/

  Alternative free source (no login, global annual composite):
    Earth Observation Group, Colorado School of Mines:
    https://eogdata.mines.edu/products/vnl/
    File: VNL_v22_npp_2022_global_vcmslcfg_c202303062300.average_masked.dat.tif.gz
    Size: ~200 MB (compressed global)

  QUICK REGIONAL DOWNLOAD (South Asia, ~30 MB via GDAL VSI):
    This module attempts to read directly from the EOG server using GDAL's
    /vsicurl/ mechanism — only the pixels needed for your study area are
    transferred.  No local download needed if your internet connection works.

HOW IT INTEGRATES:
  1. compute_ntl_scores(gdf) — adds ntl_radiance + ntl_exposure_score columns
  2. apply_ntl_to_scoring(gdf) — boosts sub_score_vru_risk on high-ntl segments
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────

# Local cache path — download the GeoTIFF here to avoid repeated remote reads.
LOCAL_VIIRS_PATHS = [
    "enrichment_data/viirs/VNL_v22_npp_2022_global_vcmslcfg.average_masked.tif",
    "enrichment_data/viirs/ntl_south_asia.tif",
    "enrichment_data/viirs/viirs_ntl.tif",
    "data/viirs_ntl.tif",
]

# Remote source (COG-compatible, read via /vsicurl/ if rasterio+GDAL available)
# Uses EOG's latest global composite — only the bounding box you need is fetched.
REMOTE_VIIRS_URL = (
    "https://eogdata.mines.edu/nighttime_light/annual/v22/2022/"
    "VNL_v22_npp_2022_global_vcmslcfg_c202303062300.average_masked.dat.tif"
)

# Percentile thresholds for normalisation (avoids extreme-value distortion)
NTL_CLIP_PERCENTILE_HIGH = 99   # values above 99th percentile clipped to 100
NTL_MIN_MEANINGFUL = 0.3        # nanoWatts/cm²/sr — below this = true dark area


def _find_viirs_file(base_dir: str = ".") -> str | None:
    """Return path to VIIRS GeoTIFF if it exists, else None."""
    for rel in LOCAL_VIIRS_PATHS:
        p = Path(base_dir) / rel
        if p.exists():
            return str(p)
    return None


def _sample_raster_at_points(raster_path: str, points_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Sample a GeoTIFF at point locations, returning an array of float values.
    Points must be in the same CRS as the raster (EPSG:4326 for VIIRS global).
    Returns NaN for points outside the raster extent.
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        raise ImportError("rasterio is required for VIIRS sampling: pip install rasterio")

    values = np.full(len(points_gdf), np.nan)
    with rasterio.open(raster_path) as src:
        xs = points_gdf.geometry.x.values
        ys = points_gdf.geometry.y.values
        # Check that points are within raster bounds
        bounds = src.bounds
        in_bounds = (
            (xs >= bounds.left) & (xs <= bounds.right) &
            (ys >= bounds.bottom) & (ys <= bounds.top)
        )
        if in_bounds.any():
            rows, cols = rowcol(src.transform, xs[in_bounds], ys[in_bounds])
            rows = np.clip(rows, 0, src.height - 1)
            cols = np.clip(cols, 0, src.width - 1)
            data = src.read(1)
            nodata = src.nodata
            sampled = data[rows, cols].astype(float)
            if nodata is not None:
                sampled[sampled == nodata] = np.nan
            sampled[sampled < 0] = np.nan
            values[in_bounds] = sampled
    return values


def _normalise_ntl(raw: np.ndarray) -> np.ndarray:
    """
    Normalise raw NTL radiance to 0–100 score.
    Values below NTL_MIN_MEANINGFUL → 0.
    Values at 99th percentile → 100.
    """
    out = np.where(np.isnan(raw), 0.0, raw)
    out = np.where(out < NTL_MIN_MEANINGFUL, 0.0, out)
    p99 = np.nanpercentile(out[out > 0], 99) if (out > 0).any() else 1.0
    out = np.clip(out / max(p99, 0.001) * 100, 0, 100)
    return out


def compute_ntl_scores(
    gdf: gpd.GeoDataFrame,
    viirs_path: str = None,
    base_dir: str = ".",
    use_remote: bool = True,
) -> gpd.GeoDataFrame:
    """
    Add nighttime lights scores to each road segment.

    Adds columns:
      ntl_radiance        — raw VIIRS radiance at segment midpoint (nW/cm²/sr)
      ntl_exposure_score  — normalised 0–100 (0 = dark, 100 = brightest area)
      ntl_available       — True if a valid VIIRS reading was obtained

    Falls back gracefully if VIIRS data is not available (all scores = 0).

    Args:
      gdf:         GeoDataFrame with road segments (must have geometry in EPSG:4326)
      viirs_path:  Path to VIIRS GeoTIFF; auto-detected if None
      base_dir:    Base directory to search for VIIRS files
      use_remote:  If True and no local file found, try remote COG via /vsicurl/
    """
    gdf = gdf.copy()
    gdf["ntl_radiance"]       = np.nan
    gdf["ntl_exposure_score"] = 0.0
    gdf["ntl_available"]      = False

    if "geometry" not in gdf.columns or gdf["geometry"].isna().all():
        print("  viirs_features: no geometry — skipping")
        return gdf

    # Resolve VIIRS file path
    if viirs_path is None:
        viirs_path = _find_viirs_file(base_dir)

    if viirs_path is None and use_remote:
        # Try remote COG read via GDAL VSI.
        # Disable SSL verification for GDAL/curl so corporate firewalls / outdated
        # certs on the EOG server don't silently block the read.
        os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
        os.environ["CURL_CA_BUNDLE"] = ""
        viirs_path = "/vsicurl/" + REMOTE_VIIRS_URL
        print(f"  VIIRS: no local file found — trying remote COG (internet required)...")
        print(f"  URL: {REMOTE_VIIRS_URL}")

    if viirs_path is None:
        print("  VIIRS: no data source available — ntl_exposure_score set to 0")
        print("  To enable: download VNL file from eogdata.mines.edu and place in:")
        print("    enrichment_data/viirs/viirs_ntl.tif")
        return gdf

    # Compute segment midpoints
    try:
        # Reproject to WGS84 for sampling
        gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
        midpoints = gdf_wgs["geometry"].centroid
        pts = gpd.GeoDataFrame({"geometry": midpoints}, crs="EPSG:4326")
    except Exception as e:
        print(f"  VIIRS: centroid computation failed — {e}")
        return gdf

    print(f"  Sampling VIIRS NTL at {len(pts):,} segment midpoints...")
    try:
        raw = _sample_raster_at_points(viirs_path, pts)
        valid = ~np.isnan(raw)
        print(f"  Valid readings: {valid.sum():,} / {len(raw):,}")

        if valid.any():
            # Cap extreme outliers (urban core lights) and normalise
            normalised = _normalise_ntl(raw)
            gdf["ntl_radiance"]       = raw
            gdf["ntl_exposure_score"] = normalised
            gdf["ntl_available"]      = valid

            # Report distribution
            scores = normalised[normalised > 0]
            if len(scores) > 0:
                print(f"  ntl_exposure_score (non-zero): "
                      f"mean={scores.mean():.1f}  p50={np.median(scores):.1f}  "
                      f"p90={np.percentile(scores, 90):.1f}  max={scores.max():.1f}")
                high_ntl = (normalised > 60).sum()
                print(f"  High-NTL segments (score>60): {high_ntl:,} "
                      f"({100*high_ntl/len(normalised):.1f}%) — nighttime exposure hotspots")
        else:
            print("  VIIRS: no valid readings in study area — check file coverage")

    except ImportError as e:
        print(f"  VIIRS: {e}")
    except Exception as e:
        print(f"  VIIRS: sampling failed — {e}")
        if "vsicurl" in str(viirs_path):
            print("  (Remote read failed — download file manually for offline use)")
            print(f"  Download from: {REMOTE_VIIRS_URL}")

    return gdf


def apply_ntl_to_scoring(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Boost VRU risk sub-score on high-NTL segments where nighttime activity
    elevates exposure beyond what land_use proxy captures.

    Logic:
      - Low NTL (rural dark)  → no adjustment (proxy was right: few pedestrians)
      - Medium NTL (50–75)    → 10% boost to VRU risk
      - High NTL (75–100)     → 20% boost (informal market zone likely)

    Only applied where ntl_available = True; uncovered segments unchanged.
    """
    gdf = gdf.copy()

    if "ntl_exposure_score" not in gdf.columns or "sub_score_vru_risk" not in gdf.columns:
        return gdf

    available = gdf.get("ntl_available", pd.Series(False, index=gdf.index)).fillna(False)
    if not available.any():
        return gdf

    mask = available & gdf["sub_score_vru_risk"].notna()
    if not mask.any():
        return gdf

    ntl = gdf.loc[mask, "ntl_exposure_score"].fillna(0)
    # Boost factor: 0 for ntl<50, linear 0→20% for ntl 50→100
    boost = np.clip((ntl - 50) / 50, 0, 1) * 0.20
    gdf.loc[mask, "sub_score_vru_risk"] = (
        gdf.loc[mask, "sub_score_vru_risk"] * (1 + boost)
    ).clip(0, 100)

    n_boosted = (boost > 0).sum()
    print(f"  NTL: VRU risk boosted on {n_boosted:,} high-nighttime-exposure segments")
    return gdf


def print_ntl_download_instructions():
    """Print step-by-step download instructions for the VIIRS NTL data."""
    print("""
VIIRS Nighttime Lights — Download Instructions
==============================================

Option A: Auto-download (EOG, no login required)
  1. Install GDAL support: pip install rasterio[all]
  2. Run pipeline — the module will attempt /vsicurl/ remote read automatically.

Option B: Manual download (recommended for repeated runs)
  1. Visit: https://eogdata.mines.edu/products/vnl/
  2. Download: VNL_v22_npp_2022_global_vcmslcfg_c202303062300.average_masked.dat.tif.gz
     (Global annual composite, ~200 MB compressed)
  3. Decompress (7-zip or gzip -d) and place the .tif file at:
     enrichment_data/viirs/viirs_ntl.tif

Option C: NASA Black Marble (higher quality, requires free account)
  1. Create free account at: https://urs.earthdata.nasa.gov/
  2. Visit: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A4/
  3. Select year 2022, tile covering South Asia (h24v06, h25v06, h25v07)
  4. Convert HDF5 to GeoTIFF with:
     gdal_translate HDF5:{file}://HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data_Fields/NearNadir_Composite_Snow_Free_avg_rad output.tif
  5. Place at enrichment_data/viirs/viirs_ntl.tif
""")

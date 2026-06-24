"""
ghsl_features.py — GHSL Global Human Settlement Layer (SMOD) integration.

Samples the GHS-SMOD settlement classification raster to road segment
centroids, providing a research-grade 7-level settlement context to replace
the unreliable binary urban/rural land-use flag.

WHY THIS MATTERS:
  The ADB dataset FAQ explicitly states: "LandUse (Urban/Rural) is an estimate
  rather than ground truth... may not reflect recent urban development."
  GHSL SMOD is the standard reference for settlement classification used by
  the UN, EU JRC, and WHO road-safety research — its 7 levels distinguish
  a dense city centre from a small rural cluster, which the binary flag cannot.

SMOD classification codes (GHS-SMOD R2023A):
  30 = Urban Centre          (continuous dense urban fabric)
  23 = Dense Urban Cluster
  22 = Semi-Dense Urban Cluster
  21 = Suburban / Peri-Urban
  13 = Rural Cluster         (small settlement, some foot traffic)
  12 = Low Density Rural
  11 = Very Low Density Rural
  10 = Water
   0 = No data

Data source:  https://human-settlement.emergency.copernicus.eu/
Expected file: enrichment_data/ghsl/GHS_SMOD_E2025.tif
  (single-file global download, ~250 MB; rasterio clips to study area)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from logger import get_logger

log = get_logger(__name__)

GHSL_FILE = Path("enrichment_data/ghsl/GHS_SMOD_E2025.tif")

# Integer SMOD code → settlement class label
SMOD_CODE_MAP = {
    30: "urban_centre",
    23: "dense_urban",
    22: "semi_dense_urban",
    21: "suburban",
    13: "rural_cluster",
    12: "low_density_rural",
    11: "very_low_density_rural",
    10: "water",
    0:  "no_data",
}

# Settlement class → equivalent land_use for Safe System threshold lookup.
# Suburban is mapped to "urban" — the conservative choice: suburban roads
# still have pedestrian exposure and deserve the tighter ceiling.
SMOD_TO_LAND_USE = {
    "urban_centre":           "urban",
    "dense_urban":            "urban",
    "semi_dense_urban":       "urban",
    "suburban":               "urban",
    "rural_cluster":          "rural",
    "low_density_rural":      "rural",
    "very_low_density_rural": "rural",
    "water":                  "unknown",
    "no_data":                "unknown",
}

# Settlement class → VRU exposure weight (0–1 scalar applied to the base
# VRU score). Replaces the binary urban=80 / rural=35 look-up in
# score_vru_context_risk(). Rural cluster gets 0.55 (not 0.35) because
# small settlements have real pedestrian activity at road level.
SMOD_VRU_WEIGHT = {
    "urban_centre":           1.00,
    "dense_urban":            0.95,
    "semi_dense_urban":       0.85,
    "suburban":               0.70,
    "rural_cluster":          0.55,
    "low_density_rural":      0.35,
    "very_low_density_rural": 0.25,
    "water":                  0.10,
    "no_data":                0.50,   # unknown — assume mid-range
}


def compute_ghsl_settlement(
    gdf: gpd.GeoDataFrame,
    base_dir: str = ".",
    ghsl_file: Path = None,
) -> gpd.GeoDataFrame:
    """
    Sample the GHSL SMOD raster at each segment's centroid.

    Adds columns:
        ghsl_settlement_code  (int)   — raw SMOD integer code
        ghsl_settlement_class (str)   — human-readable class label
        ghsl_land_use         (str)   — "urban" / "rural" / "unknown"
                                        (refined replacement for `land_use`)

    Gracefully skips if the raster file is not present — pipeline continues
    using the original binary land_use field.
    """
    gdf = gdf.copy()

    tif_path = ghsl_file or (Path(base_dir) / GHSL_FILE)
    if not tif_path.exists():
        log.warning(
            f"  [GHSL] File not found: {tif_path}\n"
            f"  Download GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.tif from\n"
            f"  https://human-settlement.emergency.copernicus.eu/ and save to\n"
            f"  {tif_path}\n"
            f"  Skipping GHSL — pipeline will use the original binary land_use field."
        )
        return gdf

    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
    except ImportError:
        log.warning("  [GHSL] rasterio not installed — pip install rasterio. Skipping.")
        return gdf

    log.info(f"  [GHSL] Sampling settlement classes from {tif_path.name}...")

    # Centroids in WGS-84
    centroids = gdf.geometry.to_crs(epsg=4326).centroid
    lons = centroids.x.values
    lats = centroids.y.values

    try:
        with rasterio.open(str(tif_path)) as src:
            # The GHSL SMOD raster uses Mollweide (ESRI:54009).
            # to_epsg() returns None for non-EPSG projections, so check
            # is_geographic instead — if the raster CRS is projected (not
            # lat/lon), always reproject coordinates before sampling.
            if src.crs and not src.crs.is_geographic:
                xs, ys = warp_transform(
                    "EPSG:4326",
                    src.crs,
                    lons.tolist(),
                    lats.tolist(),
                )
            else:
                xs, ys = lons.tolist(), lats.tolist()

            coords = list(zip(xs, ys))
            sampled = np.array(
                [v[0] for v in src.sample(coords)],
                dtype=float,
            )
    except Exception as e:
        log.warning(f"  [GHSL] Raster sampling failed: {e}. Skipping.")
        return gdf

    # Replace nodata sentinel (often 255 or -32768) with 0
    nodata = 0.0
    try:
        with rasterio.open(str(tif_path)) as src:
            nodata = src.nodata if src.nodata is not None else 0.0
    except Exception:
        pass
    sampled = np.where((sampled == nodata) | np.isnan(sampled), 0, sampled).astype(int)

    codes  = pd.array(sampled, dtype="Int64")
    labels = pd.array([SMOD_CODE_MAP.get(int(c), "no_data") for c in sampled], dtype="object")
    lu_ref = pd.array([SMOD_TO_LAND_USE.get(str(l), "unknown") for l in labels], dtype="object")

    gdf["ghsl_settlement_code"]  = codes
    gdf["ghsl_settlement_class"] = labels
    gdf["ghsl_land_use"]         = lu_ref

    # Summary
    counts = pd.Series(labels).value_counts()
    log.info("  [GHSL] Settlement class distribution:")
    for cls, n in counts.items():
        pct = 100 * n / len(gdf)
        log.info(f"    {cls:<26} {n:>6,}  ({pct:.1f}%)")

    n_changed = (
        (gdf["ghsl_land_use"] != gdf.get("land_use", pd.Series("unknown", index=gdf.index)))
        & gdf["ghsl_land_use"].isin(["urban", "rural"])
    ).sum()
    log.info(f"  [GHSL] {n_changed:,} segments reclassified vs. original binary land_use field")

    return gdf

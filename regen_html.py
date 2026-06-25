"""
regen_html.py — Rebuild the interactive map HTML from existing scored outputs.
No model rerun needed: loads speed_safety_scores_all.gpkg + corridors.gpkg
from the latest (or specified) run folder and calls build_interactive_map().

Usage:
    python regen_html.py                          # uses latest run folder
    python regen_html.py outputs/run_20260625_105536
"""
import sys
import glob
import os
import geopandas as gpd
import pandas as pd
from pathlib import Path

from visualization import build_interactive_map


def find_latest_run(base="outputs") -> Path:
    runs = sorted(Path(base).glob("run_*"), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run folders found under {base}/")
    return runs[0]


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = find_latest_run()

    print(f"Loading from: {run_dir}")

    gpkg_all = run_dir / "speed_safety_scores_all.gpkg"
    if not gpkg_all.exists():
        # Fall back to MH + TH separate files
        mh = run_dir / "speed_safety_scores_MH.gpkg"
        th = run_dir / "speed_safety_scores_TH.gpkg"
        if not mh.exists() or not th.exists():
            raise FileNotFoundError(f"No gpkg found in {run_dir}")
        gdf = pd.concat([gpd.read_file(mh), gpd.read_file(th)], ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        print(f"  Loaded MH ({len(gpd.read_file(mh)):,}) + TH ({len(gpd.read_file(th)):,}) separately")
    else:
        gdf = gpd.read_file(gpkg_all)
        print(f"  Loaded {len(gdf):,} segments from speed_safety_scores_all.gpkg")

    corridors = None
    corr_gpkg = run_dir / "speed_safety_corridors.gpkg"
    if corr_gpkg.exists():
        corridors = gpd.read_file(corr_gpkg)
        print(f"  Loaded {len(corridors):,} corridors")

    out_path = str(run_dir / "speed_safety_map.html")
    print(f"\nBuilding map -> {out_path}")
    build_interactive_map(gdf, corridors=corridors, output_path=out_path)
    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    main()

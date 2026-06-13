"""
main.py — End-to-end Speed Safety Score pipeline.

Usage:
    python main.py                    # run with real data files
    python main.py --demo             # run with synthetic data (no files needed)
    python main.py --no-eval          # skip sensitivity analysis (faster)
    python main.py --no-map           # skip map generation
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

from preprocessing    import load_maharashtra, load_thailand, load_helmet_data, \
                             merge_datasets, get_analysis_subset
from scoring          import add_safe_system_limits, compute_speed_safety_score
from advanced_scoring import run_advanced_scoring
from evaluation       import run_full_evaluation, plot_score_overview
from visualization    import build_interactive_map, export_for_esri, export_corridors

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MH_PATH     = BASE_DIR / "AI for Safer Roads 2026 - Dataset" / "ADB_Innovation_Maharashtra.geojson"
DEFAULT_TH_PATH     = BASE_DIR / "AI for Safer Roads 2026 - Dataset" / "ADB_Innovation_Thailand.geojson"
DEFAULT_HELMET_PATH = BASE_DIR / "AI for Safer Roads 2026 - Dataset" / "Archive" / "Road_Safety_Performance_Indicators_(Helmet_Wearing_results)_(adb_dashboard_data_v02).xlsx"
OUTPUT_DIR          = BASE_DIR / "outputs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mh",      default=DEFAULT_MH_PATH)
    p.add_argument("--th",      default=DEFAULT_TH_PATH)
    p.add_argument("--helmet",  default=DEFAULT_HELMET_PATH)
    p.add_argument("--out",     default=OUTPUT_DIR)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--no-map",  action="store_true")
    p.add_argument("--demo",    action="store_true")
    return p.parse_args()


# ─── Demo data generator ──────────────────────────────────────────────────────

def run_demo_mode() -> gpd.GeoDataFrame:
    from shapely.geometry import LineString

    print("\n" + "="*60)
    print("  DEMO MODE — synthetic data (real structure)")
    print("="*60)

    np.random.seed(42)
    N = 600

    def make_segments(cc, n, lat_r, lon_r):
        rcs = np.random.choice(
            ["primary","secondary","tertiary","local","motorway","trunk"],
            size=n, p=[0.20, 0.30, 0.25, 0.12, 0.06, 0.07]
        )
        lus = np.random.choice(["urban","rural"], size=n, p=[0.55, 0.45])
        limit_map = {"motorway":110,"trunk":100,"primary":80,
                     "secondary":60,"tertiary":50,"local":40}
        posted   = np.array([limit_map[r] for r in rcs], float)
        posted  += np.random.randint(-10, 20, n)
        posted   = posted.clip(20, 120)
        median   = posted * np.random.uniform(0.80, 1.15, n)
        f85      = median * np.random.uniform(1.05, 1.30, n)
        pct_over = np.clip(
            np.random.beta(2,5,n)*100 + (f85-posted).clip(0)*1.5, 0, 100
        )
        lats  = np.random.uniform(*lat_r, n)
        lons  = np.random.uniform(*lon_r, n)
        geoms = [
            LineString([(lons[i], lats[i]), (lons[i]+0.01, lats[i]+0.005)])
            for i in range(n)
        ]
        return gpd.GeoDataFrame({
            "segment_id":        [f"{cc}_{i:05d}" for i in range(n)],
            "country":           ["India (Maharashtra)" if cc=="MH" else "Thailand"] * n,
            "country_code":      [cc] * n,
            "road_name":         [f"Road {i}" for i in range(n)],
            "road_class":        rcs,
            "road_class_norm":   rcs,
            "land_use":          lus,
            "land_use_raw":      lus,
            "speed_limit":       posted,
            "speed_limit_floor": posted,
            "median_speed":      median.round(1),
            "speed_85th":        f85.round(1),
            "pct_over_limit":    pct_over.round(1),
            "n_over_limit":      (pct_over * 10).astype(int).astype(float),
            "sample_size":       np.random.randint(3, 150, n).astype(float),
            "sample_size_total": np.random.randint(10, 500, n).astype(float),
            "weighted_sample":   np.random.uniform(50, 500, n).round(1),
            "ranked_percentile": np.random.uniform(0, 100, n).round(2),
            "percentile_band":   np.random.choice(
                                     ["0-25%","25-50%","50-75%","75-100%"], n),
            "analysis_status":   ["included"] * n,
            "has_speed_data":    [True] * n,
            "scoreable":         [True] * n,
            "image_url":         ["https://www.mapillary.com"] * n,
            "urban_pct":         np.where(
                                     lus=="urban",
                                     np.random.uniform(60,100,n),
                                     np.random.uniform(0,40,n)),
            "geometry":          geoms,
        }, crs="EPSG:4326")

    mh = make_segments("MH", N, (18, 21), (73, 80))
    th = make_segments("TH", N, (13, 18), (99, 104))
    combined = gpd.GeoDataFrame(
        pd.concat([mh, th], ignore_index=True), crs="EPSG:4326"
    )
    return combined


# ─── Policy summary printer ───────────────────────────────────────────────────

def print_policy_summary(gdf: gpd.GeoDataFrame, corridors: gpd.GeoDataFrame) -> None:
    mask = gdf["scoreable"] & gdf["sss"].notna()
    df   = gdf[mask]

    print("\n" + "="*60)
    print("  POLICY SUMMARY")
    print("="*60)

    # Nilsson
    if "nilsson_fatal_ratio" in df.columns:
        gt2 = (df["nilsson_fatal_ratio"] > 2).sum()
        gt4 = (df["nilsson_fatal_ratio"] > 4).sum()
        mx  = df["nilsson_fatal_ratio"].max()
        print(f"\n  Fatal Crash Risk (Nilsson Power Model):")
        print(f"    >2× baseline risk:  {gt2:,} segments")
        print(f"    >4× baseline risk:  {gt4:,} segments")
        print(f"    Max risk ratio:     {mx:.1f}×")

    # Credibility
    if "credibility_class" in df.columns:
        print(f"\n  Speed Limit Credibility:")
        for cat in ["Credible","Low Credibility","Non-Credible","Under-Speed"]:
            n = (df["credibility_class"] == cat).sum()
            if n:
                print(f"    {cat:<22} {n:>6,}  ({100*n/len(df):.1f}%)")

    # Change effort
    if "change_effort" in df.columns:
        print(f"\n  Speed Limit Changes Needed:")
        for cat in ["No change needed","Minor (<=10 km/h)",
                    "Moderate (11-20 km/h)","Major (>20 km/h)"]:
            n = (df["change_effort"] == cat).sum()
            if n:
                print(f"    {cat:<28} {n:>6,}  ({100*n/len(df):.1f}%)")

    # Lives saved
    if "est_lives_saved" in df.columns:
        total = df["est_lives_saved"].sum()
        lower = df["lives_saved_lower"].sum()
        upper = df["lives_saved_upper"].sum()
        print(f"\n  Estimated Annual Lives Saved (if all limits corrected):")
        print(f"    Central:  {total:.1f}   Range: {lower:.1f} – {upper:.1f}")
        print(f"    (Order-of-magnitude proxy; assumptions documented)")

    # Corridors
    if corridors is not None and len(corridors):
        saved_col = "est_lives_saved" if "est_lives_saved" in corridors.columns else None
        total_corr_saved = corridors[saved_col].sum() if saved_col else 0
        print(f"\n  High-Risk Corridors:")
        print(f"    Corridors detected:  {len(corridors)}")
        print(f"    Segments covered:    {corridors['n_segments'].sum():,}")
        if saved_col:
            print(f"    Lives saved (corr.): {total_corr_saved:.1f}/yr (central)")

        print(f"\n  Top 5 Priority Corridors:")
        show = [c for c in ["priority_rank","country_code","n_segments",
                             "corridor_length_km","sss",
                             "nilsson_fatal_ratio","est_lives_saved"]
                if c in corridors.columns]
        print(corridors[show].head(5).round(2).to_string(index=False))

    print()


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out = str(Path(args.out) / f"run_{timestamp}")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print(f"\nOutput folder: {args.out}")

    print("\n" + "="*60)
    print("  ADB AI FOR SAFER ROADS — SPEED SAFETY SCORE PIPELINE")
    print("="*60)

    # ── Step 1: Load ──────────────────────────────────────────────────────
    if args.demo:
        combined = run_demo_mode()
    else:
        missing = [p for p in [args.mh, args.th] if not Path(p).exists()]
        if missing:
            print(f"\n  Missing files: {missing}")
            print("  Run with --demo to test with synthetic data")
            sys.exit(1)

        print("\n[1/6] Loading datasets...")
        mh = load_maharashtra(args.mh)
        th = load_thailand(args.th)
        if Path(args.helmet).exists():
            helmet_df = load_helmet_data(args.helmet)
            print(f"  Helmet data: {len(helmet_df)} records loaded")

        print("\n[2/6] Merging datasets...")
        combined = merge_datasets(mh, th)
        combined = get_analysis_subset(combined)

    # ── Step 2: Safe System limits ────────────────────────────────────────
    step_label = "[2/6]" if args.demo else "[3/6]"
    print(f"\n{step_label} Computing Safe System reference limits...")
    combined = add_safe_system_limits(combined)

    # ── Step 3: Base SSS ──────────────────────────────────────────────────
    step_label = "[3/6]" if args.demo else "[4/6]"
    print(f"\n{step_label} Computing Speed Safety Scores (base)...")
    combined = compute_speed_safety_score(combined)

    # Quick SSS preview
    mask = combined["scoreable"] & combined["sss"].notna()
    if mask.any():
        print(f"\n  Top 5 highest-risk segments:")
        cols = [c for c in ["segment_id","country_code","road_class_norm",
                             "land_use","speed_limit","ss_limit",
                             "speed_85th","sss","sss_band"]
                if c in combined.columns]
        print(combined[mask].nlargest(5,"sss")[cols].to_string(index=False))

    # ── Step 4: Advanced scoring ──────────────────────────────────────────
    step_label = "[4/6]" if args.demo else "[5/6]"
    print(f"\n{step_label} Running advanced scoring modules...")
    combined, corridors = run_advanced_scoring(combined)

    # Print human-readable policy summary
    print_policy_summary(combined, corridors)

    # ── Step 5: Evaluation ────────────────────────────────────────────────
    step_label = "[5/6]" if args.demo else "[6/6 part A]"
    if not args.no_eval:
        print(f"\n{step_label} Running evaluation & sensitivity analysis...")
        run_full_evaluation(combined, output_dir=args.out)
        plot_score_overview(combined, output_path=f"{args.out}/score_overview.png")
    else:
        print(f"\n{step_label} Evaluation skipped (--no-eval)")

    # ── Step 6: Outputs ───────────────────────────────────────────────────
    step_label = "[6/6]" if args.demo else "[6/6 part B]"
    print(f"\n{step_label} Generating outputs...")

    if not args.no_map:
        build_interactive_map(
            combined,
            corridors=corridors if (corridors is not None and len(corridors)) else None,
            output_path=f"{args.out}/speed_safety_map.html",
            max_segments=3000,
        )

    export_for_esri(combined, output_dir=args.out)

    if corridors is not None and len(corridors):
        export_corridors(corridors, output_dir=args.out)

    # ── Final file listing ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs in: {args.out}/")
    print()
    output_files = list(Path(args.out).glob("*"))
    output_files.sort()
    for f in output_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size_kb:>8.1f} KB")
    print("="*60 + "\n")

    return combined, corridors


if __name__ == "__main__":
    combined, corridors = main()

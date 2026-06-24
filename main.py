"""
main.py — End-to-end Speed Safety Score pipeline.

Usage:
    python main.py                    # run with real data files
    python main.py --demo             # run with synthetic data (no files needed)
    python main.py --no-eval          # skip sensitivity analysis (faster)
    python main.py --no-map           # skip map generation
"""

import argparse
import io
import logging
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Fix Windows CP1252 terminal crash on Unicode box-drawing chars in print statements
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

from logger import configure as configure_logging, get_logger
log = get_logger(__name__)

from preprocessing    import load_maharashtra, load_thailand, load_helmet_data, \
                             merge_datasets, get_analysis_subset
from scoring          import add_safe_system_limits, compute_speed_safety_score, \
                             compute_alignment_only_score
from geometry_features import compute_geometry_features
from ai_scoring import run_ai_scoring
from advanced_scoring import run_advanced_scoring
from enrichment import enrich_segments
from priority_scoring import run_priority_scoring
from ml_extension     import run_ml_extension
from ghsl_features    import compute_ghsl_settlement
from evaluation       import run_full_evaluation, plot_score_overview
from visualization    import build_interactive_map, export_for_esri, export_corridors, \
                             plot_sss_vs_pct_over_limit, plot_shap_importance
from policy_brief     import export_policy_brief

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_MH_PATH     = DATA_DIR / "ADB_Innovation_Maharashtra.geojson"
DEFAULT_TH_PATH     = DATA_DIR / "ADB_Innovation_Thailand.geojson"
DEFAULT_HELMET_PATH = DATA_DIR / "Archive" / "Road_Safety_Performance_Indicators_(Helmet_Wearing_results)_(adb_dashboard_data_v02).xlsx"
OUTPUT_DIR          = BASE_DIR / "outputs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mh",      default=DEFAULT_MH_PATH)
    p.add_argument("--th",      default=DEFAULT_TH_PATH)
    p.add_argument("--helmet",  default=DEFAULT_HELMET_PATH)
    p.add_argument("--out",     default=OUTPUT_DIR)
    p.add_argument("--no-eval",          action="store_true")
    p.add_argument("--no-map",           action="store_true")
    p.add_argument("--no-ml",            action="store_true")
    p.add_argument("--no-viirs",         action="store_true")
    p.add_argument("--mapillary-token",  default=None,
                   help="Mapillary API token (or set MAPILLARY_TOKEN env var)")
    p.add_argument("--demo",             action="store_true")
    p.add_argument("--verbose", "-v",    action="store_true",
                   help="Show DEBUG-level logs")
    p.add_argument("--quiet",  "-q",     action="store_true",
                   help="Suppress INFO logs (WARNING and above only)")
    return p.parse_args()


# ─── Demo data generator ──────────────────────────────────────────────────────

def run_demo_mode() -> gpd.GeoDataFrame:
    from shapely.geometry import LineString

    log.info("\n" + "="*60)
    log.info("  DEMO MODE — synthetic data (real structure)")
    log.info("="*60)

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
            "alignment_scoreable": [True] * n,
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

    log.info("\n" + "="*60)
    log.info("  POLICY SUMMARY")
    log.info("="*60)

    # Network coverage caveat — stated up front, not left for a reviewer to
    # discover and ask about. The gap is a property of the source ADB data
    # (most segments lack a sufficient GPS speed sample to compute F85/median
    # — see AnalysisStatus/ForAnalysis in preprocessing.py), not a choice
    # made by this methodology, but it should be visible either way.
    total_segments = len(gdf)
    n_scored = mask.sum()
    n_tier1  = gdf["alignment_scoreable"].sum() if "alignment_scoreable" in gdf.columns else 0
    log.info(f"\n  Network Coverage:")
    log.info(f"    Tier 2 (full SSS, behaviourally confirmed): {n_scored:,} / "
             f"{total_segments:,} segments ({100*n_scored/total_segments:.1f}%)")
    log.info(f"    Tier 1 (limit-vs-Safe-System-standard only): {n_tier1:,} / "
             f"{total_segments:,} segments ({100*n_tier1/total_segments:.1f}%)")
    log.info(f"    Unscored:   {total_segments - n_tier1:,} segments lack even a posted "
             f"limit — these are excluded, not scored as 'safe'.")

    # Nilsson
    if "nilsson_fatal_ratio" in df.columns:
        gt2 = (df["nilsson_fatal_ratio"] > 2).sum()
        gt4 = (df["nilsson_fatal_ratio"] > 4).sum()
        mx  = df["nilsson_fatal_ratio"].max()
        log.info(f"\n  Fatal Crash Risk (Nilsson Power Model):")
        log.info(f"    >2× baseline risk:  {gt2:,} segments")
        log.info(f"    >4× baseline risk:  {gt4:,} segments")
        log.info(f"    Max risk ratio:     {mx:.1f}×")

    # Credibility
    if "credibility_class" in df.columns:
        log.info(f"\n  Speed Limit Credibility:")
        for cat in ["Credible","Low Credibility","Non-Credible","Under-Speed"]:
            n = (df["credibility_class"] == cat).sum()
            if n:
                log.info(f"    {cat:<22} {n:>6,}  ({100*n/len(df):.1f}%)")

    # Change effort
    if "change_effort" in df.columns:
        log.info(f"\n  Speed Limit Changes Needed:")
        for cat in ["No change needed","Minor (<=10 km/h)",
                    "Moderate (11-20 km/h)","Major (>20 km/h)"]:
            n = (df["change_effort"] == cat).sum()
            if n:
                log.info(f"    {cat:<28} {n:>6,}  ({100*n/len(df):.1f}%)")

    # Lives saved
    if "est_lives_saved" in df.columns:
        total = df["est_lives_saved"].sum()
        lower = df["lives_saved_lower"].sum()
        upper = df["lives_saved_upper"].sum()
        log.info(f"\n  Estimated Annual Lives Saved (if all limits corrected):")
        log.info(f"    Central:  {total:.1f}   Range: {lower:.1f} – {upper:.1f}")
        log.info(f"    ⚠ ILLUSTRATIVE, NOT VALIDATED — depends on an unverified")
        log.info(f"    GPS-sample-to-vehicle-km conversion (config.VKM_PER_WEIGHTED_SAMPLE).")
        log.info(f"    Use for RELATIVE comparison across segments, not as a public figure.")

    # Priority Index (Exposure × Likelihood × Severity) — alongside SSS
    if "priority_index" in df.columns:
        log.info(f"\n  Priority Index (Exposure × Likelihood × Severity) — SECONDARY")
        log.info(f"  'where to act first' layer. The Tier 1/2 scores above are the")
        log.info(f"  primary answer to 'is this speed limit appropriate.'")
        for cat in ["Critical", "High Risk", "Moderate", "Acceptable"]:
            n = (df["priority_band"] == cat).sum()
            if n:
                log.info(f"    {cat:<22} {n:>6,}  ({100*n/len(df):.1f}%)")
        log.info(f"    (Provisional bands — see config.PRIORITY_BANDS docstring "
                 f"on recalibrating against real data)")

    # Uncovered risk: high-SSS roads that traffic-volume tools would miss
    if "ranked_percentile" in df.columns and df["ranked_percentile"].notna().any():
        rp_cutoff = df["ranked_percentile"].quantile(0.25)
        n_uncovered = int(
            ((df["sss"] >= 40) & (df["ranked_percentile"] <= rp_cutoff)).sum()
        )
        log.info(f"\n  Roads missed by traffic-volume prioritisation:")
        log.info(f"    SSS >= 40 AND bottom 25% by traffic volume: {n_uncovered:,} segments")
        log.info(f"    These are flagged by this model but would be de-prioritised")
        log.info(f"    by approaches that rank roads only by traffic count.")

    # Intervention zones (attribute groups, not spatial corridors — see
    # advanced_scoring.detect_corridors docstring)
    if corridors is not None and len(corridors):
        saved_col = "est_lives_saved" if "est_lives_saved" in corridors.columns else None
        total_corr_saved = corridors[saved_col].sum() if saved_col else 0
        log.info(f"\n  High-Risk Intervention Zones:")
        log.info(f"    Zones detected:      {len(corridors)}")
        log.info(f"    Segments covered:    {corridors['n_segments'].sum():,}")
        if saved_col:
            log.info(f"    Lives saved (illustrative): {total_corr_saved:.1f}/yr (central)")

        log.info(f"\n  Top 5 Priority Intervention Zones:")
        show = [c for c in ["priority_rank","country_code","n_segments",
                             "corridor_label","sss",
                             "nilsson_fatal_ratio","est_lives_saved"]
                if c in corridors.columns]
        log.info(corridors[show].head(5).round(2).to_string(index=False))

    log.info("")



# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out = str(Path(args.out) / f"run_{timestamp}")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    log.info(f"\nOutput folder: {args.out}")

    log.info("\n" + "="*60)
    log.info("  ADB AI FOR SAFER ROADS — SPEED SAFETY SCORE PIPELINE")
    log.info("="*60)

    # ── Step 1: Load ──────────────────────────────────────────────────────
    if args.demo:
        combined = run_demo_mode()
    else:
        missing = [p for p in [args.mh, args.th] if not Path(p).exists()]
        if missing:
            log.error(f"\n  Missing files: {missing}")
            log.error("  Run with --demo to test with synthetic data")
            sys.exit(1)

        log.info("\n[1/6] Loading datasets...")
        mh = load_maharashtra(args.mh)
        th = load_thailand(args.th)
        if Path(args.helmet).exists():
            helmet_df = load_helmet_data(args.helmet)
            log.info(f"  Helmet data: {len(helmet_df)} records loaded")

        log.info("\n[2/6] Merging datasets...")
        combined = merge_datasets(mh, th)
        combined = get_analysis_subset(combined)

    # ── Step 2: Road geometry features ───────────────────────────────────
    step_label = "[2/6]" if args.demo else "[3/6]"
    log.info(f"\n{step_label} Extracting road geometry features...")
    combined = compute_geometry_features(combined)

    # ── GHSL settlement classification ────────────────────────────────────
    # Runs before scoring so get_safe_system_limit() and score_vru_context_risk()
    # can use the 7-level settlement class instead of the binary land_use field.
    # Gracefully skips if enrichment_data/ghsl/GHS_SMOD_E2025.tif is not present.
    log.info("\n[GHSL] Sampling settlement classification...")
    combined = compute_ghsl_settlement(combined, base_dir=str(BASE_DIR))

    # ── Step 3: Safe System limits ────────────────────────────────────────
    step_label = "[3/6]" if args.demo else "[4/6]"
    log.info(f"\n{step_label} Computing Safe System reference limits (with geometry adjustment)...")
    combined = add_safe_system_limits(combined)

    # ── Step 4: Base SSS ──────────────────────────────────────────────────
    step_label = "[4/6]" if args.demo else "[5/6]"
    log.info(f"\n{step_label} Computing Speed Safety Scores (base)...")
    combined = compute_speed_safety_score(combined)

    # Tier 1 — alignment-only score (posted limit vs Safe System standard,
    # no behavioural/GPS data required). Covers alignment_scoreable, a
    # superset of the full-SSS `scoreable` mask. See preprocessing.py and
    # scoring.compute_alignment_only_score docstrings.
    combined = compute_alignment_only_score(combined)
    n_t1 = combined["alignment_scoreable"].sum()
    n_t2 = combined["scoreable"].sum()
    log.info(f"\n  Tier 1 (alignment-only, no behavioural data needed): "
             f"{n_t1:,} / {len(combined):,} segments ({100*n_t1/len(combined):.1f}%)")
    log.info(f"  Tier 2 (full SSS, behaviourally confirmed):           "
             f"{n_t2:,} / {len(combined):,} segments ({100*n_t2/len(combined):.1f}%)")

    # Quick SSS preview
    mask = combined["scoreable"] & combined["sss"].notna()
    if mask.any():
        log.info(f"\n  Top 5 highest-risk segments:")
        cols = [c for c in ["segment_id","country_code","road_class_norm",
                             "land_use","speed_limit","ss_limit",
                             "speed_85th","sss","sss_band"]
                if c in combined.columns]
        log.info(combined[mask].nlargest(5,"sss")[cols].to_string(index=False))

    # ── Step 4: Advanced scoring ──────────────────────────────────────────
    step_label = "[4/6]" if args.demo else "[5/6]"
    log.info(f"\n{step_label} Running advanced scoring modules...")
    combined, corridors = run_advanced_scoring(combined)

# AI anomaly detection (EXPERIMENTAL — not surfaced in map/popup/policy
    # summary; kept for later phases, see ai_scoring.py module docstring)
    print("\n[AI] Running Isolation Forest anomaly detection (experimental)...")
    combined = run_ai_scoring(combined, output_dir=args.out)

    # Exposure enrichment
    print("\n[Enrichment] Building exposure score...")
    combined = enrich_segments(combined, data_dir="enrichment_data")

    # ── Satellite enrichment: VIIRS nighttime lights ──────────────────────────
    if not args.no_viirs:
        from viirs_features import compute_ntl_scores, apply_ntl_to_scoring
        print("\n[Satellite] VIIRS nighttime lights enrichment...")
        combined = compute_ntl_scores(combined, base_dir=str(BASE_DIR))
        combined = apply_ntl_to_scoring(combined)

    # ── Mapillary infrastructure features ─────────────────────────────────────
    mapillary_token = (
        args.mapillary_token
        or os.environ.get("MAPILLARY_TOKEN", "")
    )
    if mapillary_token:
        from mapillary_features import enrich_with_mapillary, apply_mapillary_to_scoring
        # Target API queries at Critical + High Risk segments only — these are the
        # segments where infrastructure visibility evidence matters most.  Results
        # are still written to all segments sharing the same grid cell.
        priority_mask = None
        if "sss_band" in combined.columns:
            priority_mask = combined["sss_band"].isin(["Critical", "High Risk"])
            n_priority = int(priority_mask.sum())
            log.info(f"\n[Mapillary] Targeting {n_priority:,} Critical + High Risk segments")
        else:
            log.info("\n[Mapillary] Querying road infrastructure features (all segments)...")
        combined = enrich_with_mapillary(
            combined,
            token=mapillary_token,
            cache_dir=str(BASE_DIR / "enrichment_data" / "mapillary_cache"),
            segment_mask=priority_mask,
        )
        combined = apply_mapillary_to_scoring(combined)
    else:
        log.info("\n[Mapillary] Skipped — set MAPILLARY_TOKEN env var or --mapillary-token flag")

    # Priority Index (Exposure × Likelihood × Severity) — runs alongside SSS,
    # does not replace it. See priority_scoring.py module docstring.
    combined = run_priority_scoring(combined)

    # CV enrichment (feature columns only — no score changes yet).
    # Run on ALL segments so unscored roads also get mapillary_ped_crossing,
    # mapillary_street_lamp etc. as ML input features.
    # No segment_mask: full network. cv_grid_cache.json keeps repeat runs instant.
    if mapillary_token:
        from mapillary_features import enrich_with_mapillary_cv, apply_mapillary_cv_to_scoring
        print("\n[Mapillary CV] Extracting computer vision detections (full network)...")
        combined = enrich_with_mapillary_cv(
            combined,
            token=mapillary_token,
            cache_dir=str(BASE_DIR / "enrichment_data" / "mapillary_cache"),
            segment_mask=None,
        )

    # ML extension — runs AFTER CV feature enrichment (so CV columns are available
    # as model inputs) but BEFORE CV scoring amplification (so training labels are
    # the pre-amplification SSS — cleaner signal, R² ~0.877 vs ~0.785).
    if not args.no_ml and not args.demo:
        combined = run_ml_extension(combined, output_dir=args.out)

    # CV scoring amplification — applied LAST so ML trains on pre-amplification
    # labels. Modifies sub-scores for scored segments with ped crossings / lamps.
    if mapillary_token:
        combined = apply_mapillary_cv_to_scoring(combined)

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
            max_segments=1000,
            max_amenity_markers=2000,
            data_dir="enrichment_data",
        )

    export_for_esri(combined, output_dir=args.out)

    if corridors is not None and len(corridors):
        export_corridors(corridors, output_dir=args.out)

    plot_sss_vs_pct_over_limit(combined, output_dir=args.out)
    plot_shap_importance(combined, output_dir=args.out)

    export_policy_brief(combined, corridors, output_dir=args.out)

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

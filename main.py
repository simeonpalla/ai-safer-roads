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
from evaluation       import run_full_evaluation, plot_score_overview
from visualization    import build_interactive_map, export_for_esri, export_corridors

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

    print("\n" + "="*60)
    print("  POLICY SUMMARY")
    print("="*60)

    # Network coverage caveat — stated up front, not left for a reviewer to
    # discover and ask about. The gap is a property of the source ADB data
    # (most segments lack a sufficient GPS speed sample to compute F85/median
    # — see AnalysisStatus/ForAnalysis in preprocessing.py), not a choice
    # made by this methodology, but it should be visible either way.
    total_segments = len(gdf)
    n_scored = mask.sum()
    n_tier1  = gdf["alignment_scoreable"].sum() if "alignment_scoreable" in gdf.columns else 0
    print(f"\n  Network Coverage:")
    print(f"    Tier 2 (full SSS, behaviourally confirmed): {n_scored:,} / "
          f"{total_segments:,} segments ({100*n_scored/total_segments:.1f}%)")
    print(f"    Tier 1 (limit-vs-Safe-System-standard only): {n_tier1:,} / "
          f"{total_segments:,} segments ({100*n_tier1/total_segments:.1f}%)")
    print(f"    Unscored:   {total_segments - n_tier1:,} segments lack even a posted "
          f"limit — these are excluded, not scored as 'safe'.")

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
        print(f"    ⚠ ILLUSTRATIVE, NOT VALIDATED — depends on an unverified")
        print(f"    GPS-sample-to-vehicle-km conversion (config.VKM_PER_WEIGHTED_SAMPLE).")
        print(f"    Use for RELATIVE comparison across segments, not as a public figure.")

    # Priority Index (Exposure × Likelihood × Severity) — alongside SSS
    if "priority_index" in df.columns:
        print(f"\n  Priority Index (Exposure × Likelihood × Severity) — SECONDARY")
        print(f"  'where to act first' layer. The Tier 1/2 scores above are the")
        print(f"  primary answer to 'is this speed limit appropriate.'")
        for cat in ["Critical", "High Risk", "Moderate", "Acceptable"]:
            n = (df["priority_band"] == cat).sum()
            if n:
                print(f"    {cat:<22} {n:>6,}  ({100*n/len(df):.1f}%)")
        print(f"    (Provisional bands — see config.PRIORITY_BANDS docstring "
              f"on recalibrating against real data)")

    # Uncovered risk: high-SSS roads that traffic-volume tools would miss
    if "ranked_percentile" in df.columns and df["ranked_percentile"].notna().any():
        rp_cutoff = df["ranked_percentile"].quantile(0.25)
        n_uncovered = int(
            ((df["sss"] >= 40) & (df["ranked_percentile"] <= rp_cutoff)).sum()
        )
        print(f"\n  Roads missed by traffic-volume prioritisation:")
        print(f"    SSS >= 40 AND bottom 25% by traffic volume: {n_uncovered:,} segments")
        print(f"    These are flagged by this model but would be de-prioritised")
        print(f"    by approaches that rank roads only by traffic count.")

    # Intervention zones (attribute groups, not spatial corridors — see
    # advanced_scoring.detect_corridors docstring)
    if corridors is not None and len(corridors):
        saved_col = "est_lives_saved" if "est_lives_saved" in corridors.columns else None
        total_corr_saved = corridors[saved_col].sum() if saved_col else 0
        print(f"\n  High-Risk Intervention Zones:")
        print(f"    Zones detected:      {len(corridors)}")
        print(f"    Segments covered:    {corridors['n_segments'].sum():,}")
        if saved_col:
            print(f"    Lives saved (illustrative): {total_corr_saved:.1f}/yr (central)")

        print(f"\n  Top 5 Priority Intervention Zones:")
        show = [c for c in ["priority_rank","country_code","n_segments",
                             "corridor_label","sss",
                             "nilsson_fatal_ratio","est_lives_saved"]
                if c in corridors.columns]
        print(corridors[show].head(5).round(2).to_string(index=False))

    print()


# ─── Policy brief export ─────────────────────────────────────────────────────

def export_policy_brief(
    gdf: gpd.GeoDataFrame,
    corridors: gpd.GeoDataFrame,
    output_dir: str,
    top_n: int = 20,
) -> None:
    """
    Export a decision-ready Excel workbook: Top-N Priority Interventions.

    Sheet 1 — Top Priority Segments: one row per high-priority road segment
      with the columns a transport ministry needs to brief engineers:
      Road Name | Province | Road Class | Current Limit | Recommended Limit
      | Priority Score | Why Dangerous | Sinuosity | Geometry Risk
      | Nighttime Exposure | Mapillary Coverage

    Sheet 2 — Intervention Zones: aggregated corridor-level view
      (if corridor data available)

    Sheet 3 — Methodology Note: brief description of each column for reviewers
    """
    import openpyxl  # noqa — just checking it's available before we build df

    try:
        out_path = Path(output_dir) / "Top_Priority_Interventions.xlsx"

        # ── Sheet 1: Top segments ─────────────────────────────────────────────
        mask = (gdf.get("scoreable", pd.Series(False, index=gdf.index)) |
                gdf.get("alignment_scoreable", pd.Series(False, index=gdf.index)))
        df = gdf[mask].copy()

        # Sort by priority: prefer priority_index then sss
        sort_col = "priority_index" if "priority_index" in df.columns else "sss"
        if sort_col not in df.columns:
            sort_col = None

        if sort_col:
            df = df.nlargest(min(top_n * 5, len(df)), sort_col)

        # Build readable columns
        rows = []
        for _, r in df.head(top_n).iterrows():
            speed_limit    = r.get("speed_limit", np.nan)
            rec_limit      = r.get("recommended_limit", r.get("ss_limit", np.nan))
            sss            = r.get("sss", np.nan)
            speed_85th     = r.get("speed_85th", np.nan)
            pct_over       = r.get("pct_over_limit", np.nan)
            sinuosity      = r.get("sinuosity", np.nan)
            ntl_score      = r.get("ntl_exposure_score", np.nan)
            mapillary      = r.get("mapillary_covered", False)
            priority_band  = r.get("priority_band", r.get("sss_band", ""))
            change_effort  = r.get("change_effort", "")
            nilsson        = r.get("nilsson_fatal_ratio", np.nan)
            credibility    = r.get("credibility_class", "")
            osm_lit        = str(r.get("osm_lit", "") or "")
            osm_surface    = str(r.get("osm_surface", "") or "")
            blindspot      = bool(r.get("mapillary_blindspot", False))
            land_use       = str(r.get("land_use", "") or "")
            road_class     = str(r.get("road_class_norm", r.get("road_class", "")) or "")

            # Recommended intervention actions — engineering specificity beyond speed limit change
            actions = []
            if pd.notna(speed_limit) and pd.notna(rec_limit) and speed_limit - rec_limit > 20:
                actions.append(f"Reduce speed limit to {rec_limit:.0f} km/h (major revision + enforcement)")
            elif pd.notna(speed_limit) and pd.notna(rec_limit) and speed_limit > rec_limit:
                actions.append(f"Reduce speed limit to {rec_limit:.0f} km/h")
            if pd.notna(sinuosity) and sinuosity >= 1.5:
                actions.append("Install curve warning chevrons and advance warning signs")
            elif pd.notna(sinuosity) and sinuosity >= 1.2:
                actions.append("Install curve advisory speed signs")
            if blindspot:
                actions.append("Deploy speed camera or automated enforcement (unmonitored segment)")
            if osm_lit == "no" and land_use in ("urban", "interurban"):
                actions.append("Install street lighting (confirmed unlit road in populated area)")
            elif pd.notna(ntl_score) and ntl_score > 60 and osm_lit not in ("yes",):
                actions.append("Assess street lighting — high nighttime pedestrian activity detected via VIIRS")
            if osm_surface in ("unpaved", "gravel", "dirt", "compacted", "ground"):
                actions.append("Resurface to sealed asphalt (unpaved surface — loss-of-control risk)")
            if pd.notna(nilsson) and nilsson > 4 and road_class in ("trunk", "primary"):
                actions.append("Install median barrier / physical separation")
            if credibility == "Non-Credible":
                actions.append("Redesign limit scheme — posted limit widely ignored by drivers")
            if road_class == "residential" and pd.notna(speed_limit) and speed_limit > 30:
                actions.append("Implement traffic calming (residential road)")
            if not actions:
                actions.append("Monitor and schedule audit")
            intervention = "; ".join(actions)

            # Human-readable "Why Dangerous" field
            reasons = []
            if pd.notna(sss) and sss >= 50:
                reasons.append(f"SSS {sss:.0f}/100")
            if pd.notna(speed_85th) and pd.notna(speed_limit) and speed_85th > speed_limit + 10:
                reasons.append(f"85th-pct speed {speed_85th:.0f} > limit {speed_limit:.0f}")
            if pd.notna(pct_over) and pct_over > 40:
                reasons.append(f"{pct_over:.0f}% exceed limit")
            if pd.notna(nilsson) and nilsson > 2:
                reasons.append(f"{nilsson:.1f}x crash risk (Nilsson)")
            if pd.notna(sinuosity) and sinuosity >= 1.20:
                reasons.append(f"Sinuous road (SI={sinuosity:.2f})")
            if pd.notna(ntl_score) and ntl_score > 60:
                reasons.append(f"High nighttime exposure (NTL={ntl_score:.0f})")
            if credibility == "Non-Credible":
                reasons.append("Speed limit non-credible")
            why = "; ".join(reasons) if reasons else priority_band

            rows.append({
                "Rank":                r.name if pd.notna(r.name) else "",
                "Road Name":           r.get("road_name", r.get("segment_id", "")),
                "Province/State":      r.get("province", r.get("district",
                                          "Maharashtra" if r.get("country_code","") == "MH"
                                          else "Thailand")),
                "Country":             r.get("country_code", ""),
                "Road Class":          r.get("road_class_norm", r.get("road_class", "")),
                "Posted Limit (km/h)": speed_limit,
                "Recommended Limit":   rec_limit,
                "Change Needed (km/h)": (speed_limit - rec_limit
                                         if pd.notna(speed_limit) and pd.notna(rec_limit) else np.nan),
                "Speed Safety Score":  round(sss, 1) if pd.notna(sss) else "",
                "Priority Band":       priority_band,
                "Why Dangerous":       why,
                "85th pct Speed":      round(speed_85th, 1) if pd.notna(speed_85th) else "",
                "% Over Limit":        round(pct_over, 1) if pd.notna(pct_over) else "",
                "Sinuosity Index":     round(sinuosity, 3) if pd.notna(sinuosity) else "",
                "NTL Exposure (0-100)": round(ntl_score, 1) if pd.notna(ntl_score) else "N/A",
                "Mapillary Covered":   "Yes" if mapillary else "No",
                "Intervention Actions": intervention,
                "Change Effort":       change_effort,
                "Credibility":         credibility,
            })

        seg_df = pd.DataFrame(rows).reset_index(drop=True)
        seg_df.insert(0, "Priority Rank", range(1, len(seg_df) + 1))

        # ── Sheet 2: Corridors ────────────────────────────────────────────────
        corr_df = None
        if corridors is not None and len(corridors) > 0:
            keep = [c for c in ["priority_rank", "corridor_label", "country_code",
                                 "n_segments", "sss", "nilsson_fatal_ratio",
                                 "est_lives_saved", "change_effort"]
                    if c in corridors.columns]
            corr_df = corridors[keep].head(top_n).copy()

        # ── Sheet 3: Methodology note ─────────────────────────────────────────
        method_rows = [
            ("Speed Safety Score (SSS)", "0–100. Composite of speed gap, limit credibility, and VRU risk. Higher = more unsafe."),
            ("Priority Band",            "Critical / High Risk / Moderate / Acceptable based on multi-factor priority index."),
            ("Recommended Limit",        "Safe System speed limit for this road class and land-use context, geometry-adjusted for sinuous roads."),
            ("Change Needed",            "Posted limit minus recommended limit. Positive = limit should be reduced."),
            ("Sinuosity Index",          "Path length / crow-flies distance. 1.0 = straight. ≥1.20 → recommended limit reduced per AASHTO Green Book."),
            ("NTL Exposure",             "Normalized VIIRS nighttime light (0–100). Proxy for informal market activity and nighttime pedestrian density."),
            ("Mapillary Covered",        "Whether Mapillary street imagery CV features were available for this segment."),
            ("Intervention Actions",     "Specific engineering and enforcement actions recommended based on score drivers: speed limit revision, physical separation, lighting, surface quality, curve treatments, and enforcement camera deployment. Derived from SSS sub-scores, sinuosity, Mapillary blindspot flag, NTL score, and OSM infrastructure tags."),
            ("Nilsson Fatal Ratio",      "Estimated crash risk ratio relative to Safe System speed (Nilsson Power Model, WHO endorsed)."),
            ("Credibility",              "Credible = drivers naturally obey; Non-Credible = 85th-pct speed far exceeds posted limit."),
            ("Source",                   "ADB Innovation Challenge dataset. GPS speed data from ADB-provided Maharashtra and Thailand GeoJSON."),
            ("Note",                     "All figures are illustrative estimates based on available ADB sample data. Validate against official crash records before policy action."),
        ]
        method_df = pd.DataFrame(method_rows, columns=["Column / Term", "Explanation"])

        # ── Write workbook ────────────────────────────────────────────────────
        with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
            seg_df.to_excel(writer, sheet_name="Top Priority Segments", index=False)
            if corr_df is not None:
                corr_df.to_excel(writer, sheet_name="Intervention Zones", index=False)
            method_df.to_excel(writer, sheet_name="Methodology Note", index=False)

            # Auto-fit column widths
            for sheet_name, df_s in [("Top Priority Segments", seg_df),
                                       ("Methodology Note", method_df)]:
                ws = writer.sheets[sheet_name]
                for col_idx, col in enumerate(df_s.columns, start=1):
                    max_w = max(len(str(col)), df_s[col].astype(str).str.len().max())
                    ws.column_dimensions[
                        openpyxl.utils.get_column_letter(col_idx)
                    ].width = min(max_w + 2, 60)

        print(f"\n  Policy brief exported: {out_path.name}")
        print(f"  Top {len(seg_df)} priority interventions across "
              f"{seg_df['Country'].nunique()} countries")

    except ImportError:
        print("  Policy brief skipped — pip install openpyxl to enable")
    except Exception as e:
        print(f"  Policy brief export failed — {e}")


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

    # ── Step 2: Road geometry features ───────────────────────────────────
    step_label = "[2/6]" if args.demo else "[3/6]"
    print(f"\n{step_label} Extracting road geometry features...")
    combined = compute_geometry_features(combined)

    # ── Step 3: Safe System limits ────────────────────────────────────────
    step_label = "[3/6]" if args.demo else "[4/6]"
    print(f"\n{step_label} Computing Safe System reference limits (with geometry adjustment)...")
    combined = add_safe_system_limits(combined)

    # ── Step 4: Base SSS ──────────────────────────────────────────────────
    step_label = "[4/6]" if args.demo else "[5/6]"
    print(f"\n{step_label} Computing Speed Safety Scores (base)...")
    combined = compute_speed_safety_score(combined)

    # Tier 1 — alignment-only score (posted limit vs Safe System standard,
    # no behavioural/GPS data required). Covers alignment_scoreable, a
    # superset of the full-SSS `scoreable` mask. See preprocessing.py and
    # scoring.compute_alignment_only_score docstrings.
    combined = compute_alignment_only_score(combined)
    n_t1 = combined["alignment_scoreable"].sum()
    n_t2 = combined["scoreable"].sum()
    print(f"\n  Tier 1 (alignment-only, no behavioural data needed): "
          f"{n_t1:,} / {len(combined):,} segments ({100*n_t1/len(combined):.1f}%)")
    print(f"  Tier 2 (full SSS, behaviourally confirmed):           "
          f"{n_t2:,} / {len(combined):,} segments ({100*n_t2/len(combined):.1f}%)")

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
        print("\n[Mapillary] Querying road infrastructure features...")
        combined = enrich_with_mapillary(combined, token=mapillary_token,
                                         cache_dir=str(BASE_DIR / "enrichment_data" / "mapillary_cache"))
        combined = apply_mapillary_to_scoring(combined)
    else:
        print("\n[Mapillary] Skipped — set MAPILLARY_TOKEN env var or --mapillary-token flag")

    # Priority Index (Exposure × Likelihood × Severity) — runs alongside SSS,
    # does not replace it. See priority_scoring.py module docstring.
    combined = run_priority_scoring(combined)

    # ML coverage extension — predicts SSS for unscored segments
    if not args.no_ml and not args.demo:
        combined = run_ml_extension(combined, output_dir=args.out)

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

"""
ai_scoring.py v2 — Clean AI layer: Isolation Forest anomaly detection only.

STATUS (June 2026 methodology review): EXPERIMENTAL — not used in the
primary map, popup, or policy summary. "Anomalous in feature space" is a
real, defensible claim; "hidden danger" is a stronger claim this module
can't yet back up (anomalous ≠ dangerous, and there's no crash/outcome
data to check that link against). Demoted per external review and the
project's own decision to get the core Safe-System methodology validated
before re-introducing AI — see the project's Phase 1→2→3 plan. The code
and its standalone CSV output (ai_anomaly_segments.csv) are kept for that
later phase, not deleted.

DESIGN RATIONALE (per reviewer feedback):
  - REMOVED: XGBoost on ranked_percentile (would just reproduce ADB's own metric)
  - REMOVED: KMeans safe speed estimation (behaviour ≠ safety)
  - KEPT:    Isolation Forest — genuinely finds statistically unusual roads
  - ADDED:   SHAP explanations per segment (why is this road flagged?)

The Isolation Forest answers ONE clear question:
  "Which roads are anomalous compared to the rest of the network?"
That is NOT the same question as "which roads are dangerous" — anomalous-
in-feature-space and dangerous are different claims, and conflating them
is exactly the overreach this status review is correcting.

Example of what IF flags, framed honestly:
  Road with posted=60, F85=62 (2 over), compliance=85% → SSS=Moderate
  BUT this road has: 90% heavy vehicles, night-time speed spikes,
  very low sample size, high urban_pct
  → Isolation Forest: statistically unusual vs. the rest of the network.
    Worth a look, NOT a validated "hidden danger" finding.
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore")

ROAD_CLASS_ORDER = {
    "local": 1, "residential": 2, "tertiary": 3,
    "secondary": 4, "primary": 5, "trunk": 6, "motorway": 7, "unknown": 3,
}

FEATURE_COLS = [
    "speed_limit",
    "speed_85th",
    "median_speed",
    "pct_over_limit",
    "sample_size",
    "speed_gap_abs",
    "speed_gap_pct",
    "median_gap_abs",
    "speed_volatility",
    "limit_to_85_ratio",
    "road_class_encoded",
    "land_use_encoded",
    "ss_limit",
    "ss_gap",
    "ss_gap_pct",
    "compliance_severity",   # sqrt transform of pct_over_limit (non-linear)
]


def _build_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Engineer all features from raw segment data."""
    df = gdf.copy()

    df["speed_gap_abs"]      = df["speed_85th"] - df["speed_limit"]
    df["speed_gap_pct"]      = df["speed_gap_abs"] / df["speed_limit"].replace(0, np.nan)
    df["median_gap_abs"]     = df["median_speed"] - df["speed_limit"]
    df["speed_volatility"]   = df["speed_85th"] - df["median_speed"]
    df["limit_to_85_ratio"]  = df["speed_limit"] / df["speed_85th"].replace(0, np.nan)
    df["road_class_encoded"] = df["road_class_norm"].map(ROAD_CLASS_ORDER).fillna(3)
    df["land_use_encoded"]   = (df["land_use"] == "urban").astype(float)
    df["compliance_severity"]= np.sqrt(df["pct_over_limit"].clip(0, 100) / 100)

    if "ss_limit" not in df.columns:
        df["ss_limit"] = np.nan
    df["ss_gap"]     = df["speed_limit"] - df["ss_limit"]
    df["ss_gap_pct"] = df["ss_gap"] / df["ss_limit"].replace(0, np.nan)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        med = X[col].median()
        X[col] = X[col].fillna(med if pd.notna(med) else 0)
    return X


def run_isolation_forest(
    gdf: gpd.GeoDataFrame,
    contamination: float = 0.15,
) -> tuple:
    """
    Isolation Forest anomaly detection.

    Trains on ALL scoreable segments across both countries simultaneously
    so anomalies are relative to the FULL network, not per-country.
    This means a road in MH that behaves oddly compared to all MH+TH roads
    gets flagged — cross-country peer comparison.

    Returns:
        anomaly_score   pd.Series 0–100 (higher = more anomalous)
        anomaly_flag    pd.Series bool  (True = top contamination% anomalous)
        feature_scores  pd.DataFrame    (per-feature contribution, for SHAP-like explanation)
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    mask = gdf["scoreable"] & gdf["speed_85th"].notna() & gdf["speed_limit"].notna()
    anomaly_score = pd.Series(0.0, index=gdf.index)
    anomaly_flag  = pd.Series(False, index=gdf.index)

    if mask.sum() < 20:
        print("  [IF] Too few segments")
        return anomaly_score, anomaly_flag, pd.DataFrame()

    X = _build_features(gdf[mask])
    print(f"  [IF] Training on {len(X):,} segments, {len(X.columns)} features...")

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    scores = iso.decision_function(X_scaled)  # negative = more anomalous
    flags  = iso.predict(X_scaled) == -1       # True = anomaly

    # Normalise to 0–100 (100 = most anomalous)
    s_norm = (-scores - (-scores).min()) / ((-scores).max() - (-scores).min() + 1e-9)
    anomaly_score[mask] = (s_norm * 100).clip(0, 100)
    anomaly_flag[mask]  = flags

    n_flagged = flags.sum()
    print(f"  [IF] {n_flagged:,} anomalous segments ({100*n_flagged/len(X):.1f}%)")
    print(f"  [IF] Score range: {anomaly_score[mask].min():.1f} – {anomaly_score[mask].max():.1f}")

    # Per-feature anomaly contribution (mean absolute deviation from centre)
    # Used for explaining WHY a segment is anomalous (replaces SHAP)
    X_df = pd.DataFrame(X_scaled, columns=X.columns, index=gdf[mask].index)
    feature_scores = X_df.abs()  # distance from median (0) per feature per segment

    return anomaly_score, anomaly_flag, feature_scores


def explain_anomaly(
    row: pd.Series,
    feature_scores: pd.DataFrame,
    top_n: int = 3,
) -> str:
    """
    Return plain-English explanation of why a segment is anomalous.
    Used in map popups so a transport official understands the AI flag.
    """
    if row.name not in feature_scores.index:
        return ""

    scores = feature_scores.loc[row.name].sort_values(ascending=False)
    top    = scores.head(top_n)

    explanations = {
        "speed_gap_pct":      "large gap between F85 speed and posted limit",
        "compliance_severity":"high proportion of vehicles exceeding limit",
        "speed_volatility":   "high spread between median and 85th pct speed",
        "ss_gap_pct":         "posted limit far above Safe System standard",
        "limit_to_85_ratio":  "drivers operating far above posted limit",
        "land_use_encoded":   "urban road with atypical speed profile",
        "road_class_encoded": "road class inconsistent with observed speeds",
        "median_gap_abs":     "median speed significantly above posted limit",
        "sample_size":        "unusual traffic volume for this road type",
        "speed_gap_abs":      "absolute speed excess above limit is high",
    }

    reasons = [explanations.get(f, f.replace("_", " ")) for f in top.index]
    return "AI flags: " + "; ".join(reasons)


def run_ai_scoring(
    gdf: gpd.GeoDataFrame,
    output_dir: str = ".",
) -> tuple:
    """
    Main entry point. Adds columns:
        anomaly_score    — IF outlier score 0–100
        anomaly_flag     — True if segment is in top 15% most anomalous
        anomaly_reason   — Plain-English explanation for map popup
        ai_risk_tier     — "AI-Flagged Anomaly" / "Normal" based on flag
    """
    print("\n" + "=" * 60)
    print("  AI LAYER — ISOLATION FOREST ANOMALY DETECTION (EXPERIMENTAL)")
    print("=" * 60)
    print("  Status: not used in the map/popup/policy summary — see module")
    print("  docstring. Purpose: find roads STATISTICALLY UNUSUAL compared")
    print("  to similar roads in this network. Anomalous is NOT the same")
    print("  claim as dangerous — no outcome data exists yet to check that.")

    gdf = gdf.copy()

    anomaly_score, anomaly_flag, feat_scores = run_isolation_forest(gdf)
    gdf["anomaly_score"]  = anomaly_score
    gdf["anomaly_flag"]   = anomaly_flag

    # Explain each flagged segment
    print("  [IF] Generating explanations for flagged segments...")
    gdf["anomaly_reason"] = gdf.apply(
        lambda r: explain_anomaly(r, feat_scores) if r.get("anomaly_flag", False) else "",
        axis=1,
    )

    # Tier label
    gdf["ai_risk_tier"] = np.where(
        gdf["anomaly_flag"],
        "AI-Flagged Anomaly",
        "Normal",
    )

    # Summary
    mask    = gdf["scoreable"] & gdf["speed_85th"].notna()
    flagged = gdf.loc[mask & gdf["anomaly_flag"]]

    print(f"\n  Anomaly breakdown by country:")
    for cc in gdf["country_code"].unique():
        sub = gdf.loc[mask & (gdf["country_code"] == cc)]
        n_flag = sub["anomaly_flag"].sum()
        print(f"    {cc}: {n_flag:,} / {len(sub):,} flagged ({100*n_flag/len(sub):.1f}%)")

    print(f"\n  Anomalies by band (SSS-based):")
    if "sss_band" in flagged.columns:
        for band, n in flagged["sss_band"].value_counts().items():
            pct = 100 * n / len(flagged)
            print(f"    {band:<12} {n:>5,}  ({pct:.1f}%)")
        moderate_flagged = flagged[flagged["sss_band"].isin(["Acceptable", "Moderate"])]
        print(f"\n  Note: {len(moderate_flagged):,} segments flagged as statistically")
        print(f"  unusual are rated Acceptable/Moderate by SSS — worth a look,")
        print(f"  not a validated finding. EXPERIMENTAL, not surfaced elsewhere.")

    # Save anomaly summary CSV
    out = Path(output_dir)
    anomaly_cols = ["segment_id", "country_code", "road_class_norm", "land_use",
                    "speed_limit", "ss_limit", "speed_85th", "pct_over_limit",
                    "sss", "sss_band", "anomaly_score", "anomaly_flag", "anomaly_reason"]
    anomaly_cols = [c for c in anomaly_cols if c in gdf.columns]
    flagged_df   = gdf.loc[mask & gdf["anomaly_flag"], anomaly_cols]
    flagged_df.to_csv(out / "ai_anomaly_segments.csv", index=False)
    print(f"\n  Saved: {out / 'ai_anomaly_segments.csv'}")

    print("=" * 60)
    return gdf

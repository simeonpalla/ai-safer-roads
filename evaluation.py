"""
evaluation.py — Validate methodology and run sensitivity analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from config import WEIGHTS, SENSITIVITY_DELTA, SCORE_BANDS, BAND_COLORS
from scoring import compute_speed_safety_score, add_safe_system_limits


def validate_against_adb_baseline(gdf: gpd.GeoDataFrame) -> dict:
    mask = (
        gdf["scoreable"] &
        gdf["sss"].notna() &
        gdf["ranked_percentile"].notna()
    )
    df = gdf[mask][["sss", "ranked_percentile", "country_code"]].copy()

    results = {"n_segments": len(df)}

    if len(df) < 3:
        print("\n── Validation vs ADB Baseline ──")
        print(f"  Too few segments ({len(df)}) for correlation — skipping")
        results.update({"spearman_rho": np.nan, "p_value": np.nan,
                        "interpretation": "Insufficient data"})
        return results

    rho, pval = stats.spearmanr(df["sss"], df["ranked_percentile"])
    results.update({
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(pval), 6),
        "interpretation": (
            "Strong agreement with ADB baseline" if abs(rho) > 0.7 else
            "Moderate agreement — SSS adds new information" if abs(rho) > 0.4 else
            "Low correlation — SSS captures different risk dimensions"
        )
    })

    for cc in df["country_code"].unique():
        sub = df[df["country_code"] == cc]
        if len(sub) >= 3:
            r, p = stats.spearmanr(sub["sss"], sub["ranked_percentile"])
            results[f"spearman_rho_{cc}"] = round(float(r), 4)

    print("\n── Validation vs ADB Baseline ──")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def top_segment_overlap(gdf: gpd.GeoDataFrame, top_pct: float = 0.20) -> dict:
    mask = (
        gdf["scoreable"] &
        gdf["sss"].notna() &
        gdf["ranked_percentile"].notna()
    )
    df = gdf[mask].copy()

    results = {"top_pct": top_pct, "n_total_scored": len(df)}

    if len(df) < 10:
        print(f"\n── Top-{int(top_pct*100)}% Overlap ──")
        print(f"  Too few segments ({len(df)}) — skipping")
        results.update({"overlap_count": 0, "jaccard_similarity": np.nan})
        return results

    n_top = max(1, int(len(df) * top_pct))
    our_top = set(df.nlargest(n_top, "sss").index)
    adb_top = set(df.nlargest(n_top, "ranked_percentile").index)
    union   = our_top | adb_top
    overlap = len(our_top & adb_top)
    jaccard = overlap / len(union) if union else 0.0

    results.update({
        "n_in_top": n_top,
        "overlap_count": overlap,
        "overlap_pct": round(overlap / n_top * 100, 1),
        "jaccard_similarity": round(jaccard, 4),
    })
    print(f"\n── Top-{int(top_pct*100)}% Overlap with ADB ──")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def sensitivity_analysis(
    gdf: gpd.GeoDataFrame,
    delta: float = SENSITIVITY_DELTA,
    top_pct: float = 0.20,
) -> pd.DataFrame:
    print(f"\n── Sensitivity Analysis (±{delta*100:.0f}% weight perturbation) ──")

    mask = gdf["scoreable"] & gdf["sss"].notna()
    if mask.sum() < 10:
        print("  Too few scored segments — skipping sensitivity analysis")
        return pd.DataFrame()

    base_weights = WEIGHTS.copy()
    base_scores  = gdf.loc[mask, "sss"].copy()
    n_top        = max(1, int(len(base_scores) * top_pct))

    results = []
    scoreable_keys = [k for k in base_weights if k != "confidence_weight"]

    for perturb_key in scoreable_keys:
        for direction, sign in [("+", 1), ("-", -1)]:
            new_w = base_weights.copy()
            new_w[perturb_key] = max(0.01, new_w[perturb_key] + sign * delta)

            total = sum(v for k, v in new_w.items() if k != "confidence_weight")
            for k in scoreable_keys:
                new_w[k] = new_w[k] / total

            # Suppress per-run print output
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                gdf_temp = compute_speed_safety_score(gdf.copy(), weights=new_w)
            finally:
                sys.stdout = old_stdout

            new_scores = gdf_temp.loc[mask, "sss"]

            rho, _ = stats.spearmanr(base_scores, new_scores)
            base_top = set(base_scores.nlargest(n_top).index)
            new_top  = set(new_scores.nlargest(n_top).index)
            pct_changed = 100 * len(base_top.symmetric_difference(new_top)) / n_top

            results.append({
                "perturbed_weight":      perturb_key,
                "direction":             direction,
                "new_value":             round(new_w[perturb_key], 3),
                "spearman_rho_vs_base":  round(float(rho), 4),
                f"pct_top{int(top_pct*100)}_changed": round(pct_changed, 1),
            })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print(df_res.to_string(index=False))
        print(f"\nMean rank stability (ρ): {df_res['spearman_rho_vs_base'].mean():.4f}")
        print("(ρ > 0.95 = robust methodology)")
    return df_res


def cross_country_consistency(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    mask = gdf["scoreable"] & gdf["sss"].notna()
    if mask.sum() == 0:
        print("\n── Cross-Country Consistency: no data ──")
        return pd.DataFrame()

    groups = gdf[mask].groupby(["country_code", "road_class_norm", "land_use"])["sss"]
    summary = groups.agg(["mean", "median", "std", "count"]).round(2)
    summary.columns = ["mean_sss", "median_sss", "std_sss", "n_segments"]
    summary = summary[summary["n_segments"] >= 5]

    print("\n── Cross-Country Score Consistency ──")
    print(summary.to_string())
    return summary


def score_diagnostics(gdf: gpd.GeoDataFrame) -> None:
    mask = gdf["scoreable"] & gdf["sss"].notna()
    scores = gdf.loc[mask, "sss"]

    print("\n── Score Diagnostics ──")
    if len(scores) == 0:
        print("  No scored segments found!")
        return

    print(f"  N scored:  {len(scores):,}")
    print(f"  Mean:      {scores.mean():.1f}")
    print(f"  Std:       {scores.std():.1f}  (want 15–30 for good discrimination)")
    print(f"  Skewness:  {scores.skew():.2f}  (±1 is fine)")
    from config import SCORE_BANDS
    for band, (lo, hi) in SCORE_BANDS.items():
        pct = ((scores >= lo) & (scores < hi)).mean() * 100
        print(f"  % {band:<12} {pct:5.1f}%")


def run_full_evaluation(gdf: gpd.GeoDataFrame, output_dir: str = ".") -> dict:
    print("\n" + "="*60)
    print("  SPEED SAFETY SCORE — EVALUATION REPORT")
    print("="*60)

    score_diagnostics(gdf)

    baseline = validate_against_adb_baseline(gdf)
    overlap  = top_segment_overlap(gdf, top_pct=0.20)
    sens_df  = sensitivity_analysis(gdf)
    cc_df    = cross_country_consistency(gdf)

    if not sens_df.empty:
        sens_df.to_csv(f"{output_dir}/sensitivity_analysis.csv", index=False)
    cc_df.to_csv(f"{output_dir}/cross_country_consistency.csv")

    print("\n" + "="*60)
    print("  Evaluation complete. Files saved to:", output_dir)
    print("="*60)

    return {
        "baseline_validation": baseline,
        "top20_overlap":       overlap,
        "sensitivity":         sens_df,
        "cross_country":       cc_df,
    }


def plot_score_overview(gdf: gpd.GeoDataFrame, output_path: str = "score_overview.png"):
    mask = gdf["scoreable"] & gdf["sss"].notna()
    df   = gdf[mask].copy()

    if len(df) == 0:
        print("No scored data to plot.")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Speed Safety Score — Diagnostic Overview", fontsize=16, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    for cc, color in zip(["MH", "TH"], ["#e74c3c", "#3498db"]):
        sub = df[df["country_code"] == cc]["sss"]
        if len(sub):
            ax1.hist(sub, bins=30, alpha=0.6, label=cc, color=color, edgecolor="white")
    ax1.axvline(80, color="red",    ls="--", alpha=0.7)
    ax1.axvline(60, color="orange", ls="--", alpha=0.7)
    ax1.set_xlabel("SSS"); ax1.set_ylabel("Count")
    ax1.set_title("SSS Distribution by Country"); ax1.legend()

    # 2. Band Pie
    ax2 = fig.add_subplot(gs[0, 1])
    band_counts = df["sss_band"].value_counts()
    if len(band_counts):
        colors = [BAND_COLORS.get(b, "#aaa") for b in band_counts.index]
        ax2.pie(band_counts.values, labels=band_counts.index, colors=colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax2.set_title("Score Band Distribution")

    # 3. Sub-score correlations
    ax3 = fig.add_subplot(gs[0, 2])
    sub_cols = [c for c in df.columns if c.startswith("sub_score_")]
    if sub_cols:
        corr = df[sub_cols + ["sss"]].corr()["sss"].drop("sss")
        corr.index = [c.replace("sub_score_", "") for c in corr.index]
        ax3.barh(corr.index, corr.values, color="#2ecc71", edgecolor="white")
        ax3.set_xlabel("Pearson r with SSS")
    ax3.set_title("Sub-Score Contribution")

    # 4. SSS vs ADB Baseline scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sub_rank = df[df["ranked_percentile"].notna()]
    for cc, color in zip(["MH", "TH"], ["#e74c3c", "#3498db"]):
        s = sub_rank[sub_rank["country_code"] == cc]
        if len(s):
            ax4.scatter(s["ranked_percentile"], s["sss"], alpha=0.3, s=5, c=color, label=cc)
    ax4.set_xlabel("ADB RankedPercentile"); ax4.set_ylabel("SSS")
    ax4.set_title("SSS vs ADB Baseline"); ax4.legend(markerscale=3)

    # 5. Speed gap vs SSS
    ax5 = fig.add_subplot(gs[1, 1])
    gap = (df["speed_85th"] - df["speed_limit"]).clip(-30, 60)
    has_vru = "sub_score_vru_risk" in df.columns
    c_vals = df["sub_score_vru_risk"] if has_vru else "steelblue"
    sc = ax5.scatter(gap, df["sss"], c=c_vals,
                     cmap="RdYlGn_r" if has_vru else None,
                     alpha=0.3, s=5)
    if has_vru:
        plt.colorbar(sc, ax=ax5, label="VRU Risk")
    ax5.axvline(0, color="grey", lw=0.5)
    ax5.set_xlabel("85th pct − posted limit (km/h)"); ax5.set_ylabel("SSS")
    ax5.set_title("Speed Gap vs SSS")

    # 6. Box by road class
    ax6 = fig.add_subplot(gs[1, 2])
    order = ["local","residential","tertiary","secondary","primary","trunk","motorway"]
    order = [o for o in order if o in df.get("road_class_norm", pd.Series()).values]
    if order and "road_class_norm" in df.columns:
        plot_df = df[df["road_class_norm"].isin(order)]
        data = [plot_df[plot_df["road_class_norm"] == o]["sss"].dropna().values for o in order]
        data = [d for d in data if len(d)]
        if data:
            ax6.boxplot(data, labels=order[:len(data)], patch_artist=True, showfliers=False)
            ax6.set_xticklabels(order[:len(data)], rotation=30, ha="right", fontsize=8)
    ax6.set_ylabel("SSS"); ax6.set_title("SSS by Road Class")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nDiagnostic plot saved: {output_path}")
    plt.close()

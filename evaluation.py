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

from config import SENSITIVITY_DELTA, SCORE_BANDS, BAND_COLORS
from scoring import compute_speed_safety_score, add_safe_system_limits, WEIGHTS
import priority_scoring


def compare_to_traffic_ranking(gdf: gpd.GeoDataFrame) -> dict:
    """
    Compare SSS to ADB's RankedPercentile (traffic volume ranking).

    LOW correlation is the EXPECTED and DESIRED outcome here.
    SSS answers "is this speed limit appropriate for this road?"
    RankedPercentile answers "how much traffic uses this road?"
    These are different questions. A road can be dangerously mis-posted
    regardless of how much traffic it carries — that is exactly the point
    of a speed-limit-appropriateness methodology.

    A high rho would suggest SSS is just re-ranking by traffic volume,
    which would mean it adds nothing beyond what ADB already has.
    A low rho confirms SSS is measuring something different — limit
    appropriateness — and will surface high-risk roads that volume-based
    prioritisation misses (see uncovered_risk_analysis).
    """
    mask = (
        gdf["scoreable"] &
        gdf["sss"].notna() &
        gdf["ranked_percentile"].notna()
    )
    df = gdf[mask][["sss", "ranked_percentile", "country_code"]].copy()

    results = {"n_segments": len(df)}

    if len(df) < 3:
        print("\n── SSS vs Traffic Volume Ranking ──")
        print(f"  Too few segments ({len(df)}) for correlation — skipping")
        results.update({"spearman_rho": np.nan, "p_value": np.nan,
                        "interpretation": "Insufficient data"})
        return results

    rho, pval = stats.spearmanr(df["sss"], df["ranked_percentile"])
    results.update({
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(pval), 6),
        "interpretation": (
            "Expected: SSS and traffic volume are different signals — "
            "SSS surfaces limit-appropriateness risk that volume rankings miss."
            if abs(rho) < 0.3 else
            "Moderate overlap — SSS and traffic volume partially agree, "
            "but SSS still adds new information."
            if abs(rho) < 0.6 else
            "High overlap — SSS may be partially proxying traffic volume; "
            "review whether volume is inadvertently driving scores."
        )
    })

    for cc in df["country_code"].unique():
        sub = df[df["country_code"] == cc]
        if len(sub) >= 3:
            r, p = stats.spearmanr(sub["sss"], sub["ranked_percentile"])
            results[f"spearman_rho_{cc}"] = round(float(r), 4)

    print("\n── SSS vs Traffic Volume Ranking (RankedPercentile) ──")
    print(f"  Note: low rho is EXPECTED — SSS measures limit appropriateness,")
    print(f"  RankedPercentile measures traffic volume. Different questions.")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def top_segment_overlap(gdf: gpd.GeoDataFrame, top_pct: float = 0.20) -> dict:
    """
    Compare which segments each method puts in the top X%.

    LOW overlap is the DESIRED outcome: it means SSS is surfacing
    high-risk roads that a traffic-volume tool (RankedPercentile) would
    not prioritise. These are exactly the roads that a speed-limit
    appropriateness methodology is supposed to find.

    High overlap would indicate SSS is selecting mostly the same roads
    as traffic volume — meaning it adds little over existing tools.
    """
    mask = (
        gdf["scoreable"] &
        gdf["sss"].notna() &
        gdf["ranked_percentile"].notna()
    )
    df = gdf[mask].copy()

    results = {"top_pct": top_pct, "n_total_scored": len(df)}

    if len(df) < 10:
        print(f"\n── Top-{int(top_pct*100)}% Coverage Comparison ──")
        print(f"  Too few segments ({len(df)}) — skipping")
        results.update({"overlap_count": 0, "jaccard_similarity": np.nan})
        return results

    n_top = max(1, int(len(df) * top_pct))
    our_top = set(df.nlargest(n_top, "sss").index)
    adb_top = set(df.nlargest(n_top, "ranked_percentile").index)
    union   = our_top | adb_top
    overlap = len(our_top & adb_top)
    jaccard = overlap / len(union) if union else 0.0
    unique_to_sss = n_top - overlap

    results.update({
        "n_in_top": n_top,
        "overlap_count": overlap,
        "overlap_pct": round(overlap / n_top * 100, 1),
        "jaccard_similarity": round(jaccard, 4),
        "unique_to_sss": unique_to_sss,
    })
    print(f"\n── Top-{int(top_pct*100)}% Coverage: SSS vs Traffic Volume ──")
    print(f"  SSS uniquely flags {unique_to_sss:,} high-risk segments "
          f"that traffic-volume ranking would miss.")
    print(f"  Low overlap ({100-results['overlap_pct']:.0f}% non-overlapping) = "
          f"SSS is adding new information, not re-ranking by volume.")
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


def export_manual_review_sample(
    gdf: gpd.GeoDataFrame,
    output_dir: str = ".",
    n: int = 20,
    score_col: str = "sss",
) -> pd.DataFrame:
    """
    Export the n highest- and n lowest-scored segments for manual review.

    WHY THIS EXISTS: validate_against_adb_baseline() compares this score to
    ADB's own RankedPercentile column — that's a comparison between two
    scores, not validation against an outcome (crashes/injuries/fatalities).
    No such outcome data exists in this dataset, so it can't be built from
    code alone. This export is the cheap, honest substitute: pull the
    highest- and lowest-scored segments alongside their street imagery link
    so a road engineer can sanity-check "would I agree this is
    Critical/Acceptable looking at the actual road" — not rigorous
    validation, but a real, defensible check that's currently missing
    entirely.
    """
    mask = gdf["scoreable"] & gdf[score_col].notna()
    df = gdf.loc[mask].copy()
    if len(df) == 0:
        print(f"\n[Manual Review Export] No scored segments available — skipping")
        return pd.DataFrame()

    cols = [c for c in [
        "segment_id", "country_code", "road_class_norm", "land_use",
        score_col, f"{score_col}_band", "speed_limit", "ss_limit", "speed_85th",
        "sub_score_limit_alignment", "sub_score_limit_credibility", "sub_score_vru_risk",
        "sss_recommendation", "image_url",
    ] if c in df.columns]

    top    = df.nlargest(n, score_col)[cols].copy()
    top["review_group"] = f"TOP {n} (highest {score_col.upper()})"
    bottom = df.nsmallest(n, score_col)[cols].copy()
    bottom["review_group"] = f"BOTTOM {n} (lowest {score_col.upper()})"

    review = pd.concat([top, bottom], ignore_index=True)
    out_path = f"{output_dir}/manual_review_sample.csv"
    review.to_csv(out_path, index=False)

    n_with_image = review["image_url"].notna().sum() if "image_url" in review.columns else 0
    print(f"\n[Manual Review Export] {len(review)} segments "
          f"({n} highest + {n} lowest {score_col.upper()}) → {out_path}")
    print(f"  {n_with_image}/{len(review)} have a street imagery link — "
          f"open each and ask: would a road engineer agree with this score?")
    return review


def uncovered_risk_analysis(
    gdf: gpd.GeoDataFrame,
    sss_threshold: float = 40.0,
    volume_percentile: float = 0.25,
) -> dict:
    """
    Find segments flagged as high-risk by SSS that traffic-volume tools
    would de-prioritise: SSS >= sss_threshold AND ranked_percentile in
    the bottom volume_percentile of the network.

    These are the roads a volume-based approach would leave unaddressed.
    They are the core argument for why a speed-limit-appropriateness
    methodology adds value over simply acting on high-traffic corridors.
    """
    mask = gdf["scoreable"] & gdf["sss"].notna()
    df = gdf[mask].copy()

    results = {"sss_threshold": sss_threshold, "n_scored": len(df)}

    if "ranked_percentile" not in df.columns or df["ranked_percentile"].isna().all():
        print("\n── Uncovered Risk Analysis ──")
        print("  ranked_percentile not available — skipping")
        results["n_uncovered"] = 0
        return results

    rp_cutoff = df["ranked_percentile"].quantile(volume_percentile)
    uncovered = df[
        (df["sss"] >= sss_threshold) &
        (df["ranked_percentile"] <= rp_cutoff)
    ]

    pct_of_scored = 100 * len(uncovered) / len(df) if len(df) else 0

    results.update({
        "n_uncovered": len(uncovered),
        "pct_of_scored": round(pct_of_scored, 1),
        "volume_percentile_cutoff": round(float(rp_cutoff), 1),
    })

    print(f"\n── Uncovered Risk Analysis ──")
    print(f"  Roads with SSS >= {sss_threshold} AND in bottom "
          f"{int(volume_percentile*100)}% by traffic volume:")
    print(f"  {len(uncovered):,} segments ({pct_of_scored:.1f}% of scored network)")
    print(f"  These roads would be MISSED by traffic-volume prioritisation.")

    if len(uncovered) > 0:
        show_cols = [c for c in [
            "segment_id", "country_code", "road_class_norm", "land_use",
            "sss", "sss_band", "speed_limit", "ss_limit", "ranked_percentile",
        ] if c in uncovered.columns]
        top5 = uncovered.nlargest(5, "sss")[show_cols]
        print(f"\n  Top 5 uncovered high-risk segments:")
        print(top5.to_string(index=False))

    return results


def run_full_evaluation(gdf: gpd.GeoDataFrame, output_dir: str = ".") -> dict:
    print("\n" + "="*60)
    print("  SPEED SAFETY SCORE — EVALUATION REPORT")
    print("="*60)

    score_diagnostics(gdf)

    baseline   = compare_to_traffic_ranking(gdf)
    overlap    = top_segment_overlap(gdf, top_pct=0.20)
    uncovered  = uncovered_risk_analysis(gdf)
    sens_df    = sensitivity_analysis(gdf)
    cc_df      = cross_country_consistency(gdf)
    review_df  = export_manual_review_sample(gdf, output_dir=output_dir, n=20, score_col="sss")

    if not sens_df.empty:
        sens_df.to_csv(f"{output_dir}/sensitivity_analysis.csv", index=False)
    cc_df.to_csv(f"{output_dir}/cross_country_consistency.csv")

    results = {
        "traffic_ranking_comparison": baseline,
        "top20_overlap":              overlap,
        "uncovered_risk":             uncovered,
        "sensitivity":                sens_df,
        "cross_country":              cc_df,
        "manual_review_sample":       review_df,
    }

    # Priority Index evaluation — runs only if priority_scoring.py has already
    # added the column (see main.py). Kept separate from the SSS evaluation
    # above so SSS results are unaffected either way.
    if "priority_index" in gdf.columns:
        print("\n" + "="*60)
        print("  PRIORITY INDEX — EVALUATION")
        print("="*60)
        sss_vs_priority = priority_scoring.compare_to_sss(gdf)
        priority_sens_df = priority_scoring.priority_sensitivity_analysis(gdf)
        if not priority_sens_df.empty:
            priority_sens_df.to_csv(f"{output_dir}/priority_index_sensitivity_analysis.csv", index=False)
        results["sss_vs_priority_index"]       = sss_vs_priority
        results["priority_index_sensitivity"]  = priority_sens_df

    print("\n" + "="*60)
    print("  Evaluation complete. Files saved to:", output_dir)
    print("="*60)

    return results


def plot_score_overview(gdf: gpd.GeoDataFrame, output_path: str = "score_overview.png"):
    mask = gdf["scoreable"] & gdf["sss"].notna()
    df   = gdf[mask].copy()

    if len(df) == 0:
        print("No scored data to plot.")
        return

    fig = plt.figure(figsize=(22, 12))
    fig.suptitle("Speed Safety Score — Diagnostic Overview", fontsize=16, fontweight="bold")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

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

    # 7. SSS vs Priority Index — only if priority_scoring.py has run.
    # This is the panel most directly useful for "decide after seeing it":
    # tight diagonal clustering = the two methods agree; scatter/fan-out =
    # Priority Index is surfacing a meaningfully different set of roads.
    if "priority_index" in df.columns:
        ax7 = fig.add_subplot(gs[0, 3])
        sub_pi = df[df["priority_index"].notna()]
        for cc, color in zip(["MH", "TH"], ["#e74c3c", "#3498db"]):
            s = sub_pi[sub_pi["country_code"] == cc]
            if len(s):
                ax7.scatter(s["sss"], s["priority_index"], alpha=0.3, s=5, c=color, label=cc)
        ax7.set_xlabel("SSS (legacy)"); ax7.set_ylabel("Priority Index (new)")
        ax7.set_title("SSS vs Priority Index"); ax7.legend(markerscale=3)

    # 4. SSS vs ADB Baseline scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sub_rank = df[df["ranked_percentile"].notna()]
    for cc, color in zip(["MH", "TH"], ["#e74c3c", "#3498db"]):
        s = sub_rank[sub_rank["country_code"] == cc]
        if len(s):
            ax4.scatter(s["ranked_percentile"], s["sss"], alpha=0.3, s=5, c=color, label=cc)
    ax4.set_xlabel("RankedPercentile (traffic volume)"); ax4.set_ylabel("SSS")
    ax4.set_title("SSS vs Traffic Volume Rank"); ax4.legend(markerscale=3)

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
            ax6.boxplot(data, tick_labels=order[:len(data)], patch_artist=True, showfliers=False)
            ax6.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax6.set_ylabel("SSS"); ax6.set_title("SSS by Road Class")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nDiagnostic plot saved: {output_path}")
    plt.close()

"""
priority_scoring.py — Priority Index: Exposure × Likelihood × Severity.

WHY THIS EXISTS:
  External review of the SSS methodology (scoring.py) recommended replacing
  the additive "Alignment + Gap + VRU" architecture with the risk-equation
  structure used by WHO Safe System, iRAP, EuroRAP, Austroads and Vision
  Zero practitioners:

      Risk = Exposure × Likelihood × Severity

  This module computes that as a geometric mean:

      Priority Index = (Exposure × Likelihood × Severity) ^ (1/3)

  A road cannot score high on Priority Index if any one dimension is near
  zero — e.g. heavy speeding on an empty rural lane with no traffic and no
  population nearby should NOT outrank a moderately-speeding road through a
  school zone. The additive SSS formula could mask exactly that kind of
  imbalance. This is intentional, not a bug to "fix" with a high floor.

  SSS is NOT removed — scoring.py is untouched. Both numbers are reported
  side by side (see compare_to_sss() below) so you can compare rankings on
  real data before deciding whether to keep, blend, or retire either one.

LAYER SOURCES:
  Exposure   — enrichment.py's exposure_score (traffic volume + population +
               intersections + schools + hospitals). Run enrich_segments()
               first.
  Likelihood — operating speed gap (scoring.py sub-score) + credibility
               (advanced_scoring.py — used INSTEAD of raw pct_over_limit,
               which scoring.py documents as unreliable) + speed
               variability (new here: F85 − median spread).
  Severity   — Safe System gap (scoring.py's alignment sub-score, demoted
               from "the score" to one severity input) + Nilsson risk ratio
               (advanced_scoring.py) + road class crash-severity proxy +
               helmet SPI (lower helmet use = worse outcome at any speed).

PIPELINE ORDER (see main.py):
  add_safe_system_limits() → compute_speed_safety_score() → run_advanced_scoring()
  → run_ai_scoring() → enrich_segments() → run_priority_scoring()   ← this module
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

from config import (
    LIKELIHOOD_WEIGHTS, SEVERITY_WEIGHTS,
    ROAD_CLASS_SEVERITY_MAP, INFRA_SEVERITY_ADJUSTMENTS, HELMET_SPI,
    PRIORITY_INDEX_FLOOR, PRIORITY_BANDS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared helper
# ═══════════════════════════════════════════════════════════════════════════

def _weighted_combine(components: dict, weights: dict):
    """
    Weighted sum over whichever components are actually available (not
    None), renormalising weights to sum to 1 over the available subset.
    Returns (combined_series, dict_of_weights_actually_used).
    """
    available = {k: v for k, v in components.items() if v is not None}
    used_w    = {k: weights.get(k, 0) for k in available}
    total_w   = sum(used_w.values())
    if total_w <= 0 or not available:
        idx = next(iter(components.values()), pd.Series(dtype=float)).index \
              if any(v is not None for v in components.values()) else pd.Index([])
        return pd.Series(0.0, index=idx), {}
    combined = sum(available[k].fillna(0) * (used_w[k] / total_w) for k in available)
    return combined, used_w


# ═══════════════════════════════════════════════════════════════════════════
# Likelihood layer
# ═══════════════════════════════════════════════════════════════════════════

def score_credibility_likelihood(credibility_gap: pd.Series) -> pd.Series:
    """
    Convert the credibility gap (F85 − posted limit, km/h; see
    advanced_scoring.credibility()) into a 0–100 likelihood sub-score.

    Used INSTEAD of raw pct_over_limit, which scoring.py documents as
    unreliable for ~30% of segments (F85 far above posted limit but
    pct_over < 5%, which is physically impossible if pct_over really means
    "% of vehicles speeding"). Same underlying evidence — how far drivers
    exceed the posted limit — via the AASHTO/TRL gap field instead.

    gap ≤ 0 (at or under the limit) → 0 (no excess-speed likelihood signal)
    gap ≥ 20 km/h (TRL "Non-Credible" territory) → 100
    """
    gap = credibility_gap.fillna(0)
    return (gap.clip(lower=0) / 20).clip(0, 1) * 100


def score_speed_variability(speed_85th: pd.Series, median_speed) -> pd.Series:
    """
    F85 minus median speed = spread of the speed distribution. A wide
    spread means some drivers are going much faster than the typical
    driver — erratic, unpredictable speed behaviour that is itself a
    crash-likelihood signal, independent of how far over the posted limit
    anyone is.

    0 km/h spread → 0. 25+ km/h spread (a very wide distribution) → 100.
    Missing median_speed (optional field — see preprocessing.py) is treated
    as no-evidence-of-volatility (0), not penalised.
    """
    if median_speed is None:
        return pd.Series(0.0, index=speed_85th.index)
    spread = (speed_85th - median_speed).fillna(0).clip(lower=0)
    return (spread / 25).clip(0, 1) * 100


def compute_likelihood_score(gdf: gpd.GeoDataFrame, weights: dict = None) -> gpd.GeoDataFrame:
    """Likelihood layer. HONESTY NOTE: all three inputs are behaviour-
    derived (see config.LIKELIHOOD_WEIGHTS docstring) — this measures
    whether observed driving corroborates that the posted limit needs
    review, not "likelihood of a crash" in the classic iRAP sense, since
    no non-behavioural likelihood signal (violations/conflicts/near-misses)
    exists in this dataset."""
    if weights is None:
        weights = LIKELIHOOD_WEIGHTS

    gdf  = gdf.copy()
    mask = gdf["scoreable"]

    if "sub_score_limit_credibility" in gdf.columns:
        speed_gap_score = gdf["sub_score_limit_credibility"]
    else:
        from scoring import score_limit_credibility_gap
        speed_gap_score = pd.Series(np.nan, index=gdf.index)
        speed_gap_score.loc[mask] = score_limit_credibility_gap(
            gdf.loc[mask, "speed_85th"], gdf.loc[mask, "speed_limit"]
        )

    cred_available = "credibility_gap" in gdf.columns
    cred_score = score_credibility_likelihood(gdf["credibility_gap"]) if cred_available else None

    median_speed = gdf["median_speed"] if "median_speed" in gdf.columns else None
    var_score = score_speed_variability(gdf["speed_85th"], median_speed)

    components = {
        "limit_credibility_gap": speed_gap_score,
        "credibility":            cred_score,
        "speed_variability":      var_score,
    }
    combined, used_weights = _weighted_combine(components, weights)
    if "credibility" not in used_weights:
        print("  [Likelihood] credibility_gap not found — run "
              "advanced_scoring.credibility() before priority_scoring. "
              "Weight redistributed across remaining components.")

    gdf.loc[mask, "sub_likelihood_speed_gap"]   = speed_gap_score[mask]
    gdf.loc[mask, "sub_likelihood_credibility"] = cred_score[mask] if cred_score is not None else np.nan
    gdf.loc[mask, "sub_likelihood_variability"] = var_score[mask]
    gdf.loc[mask, "likelihood_score"]           = combined[mask].clip(0, 100)
    return gdf


# ═══════════════════════════════════════════════════════════════════════════
# Severity layer
# ═══════════════════════════════════════════════════════════════════════════

def score_nilsson_severity(nilsson_fatal_ratio: pd.Series) -> pd.Series:
    """
    WHO Power Model fatal-risk ratio (advanced_scoring.nilsson()) rescaled
    to 0–100. ratio = 1 (at Safe System baseline) → 0. ratio ≥ 5 (5x
    baseline fatal risk — already "Critical" per advanced_scoring's own
    labelling) → 100. Missing ratio → 0 (treated as not-elevated, not
    penalised).
    """
    r = nilsson_fatal_ratio.fillna(1.0)
    return ((r - 1) / 4).clip(0, 1) * 100


def score_road_class_severity(road_class_norm: pd.Series) -> pd.Series:
    """Crash-energy / divided-vs-undivided proxy — see ROAD_CLASS_SEVERITY_MAP
    in config.py for the rationale. This is the FALLBACK PRIOR used by
    score_infrastructure_severity() when no OSM way match is available."""
    fallback = ROAD_CLASS_SEVERITY_MAP.get("unknown", 50)
    return road_class_norm.map(ROAD_CLASS_SEVERITY_MAP).fillna(fallback)


def score_infrastructure_severity(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Severity contribution from road infrastructure.

    REPLACES assumption-only road-class severity with OBSERVED OSM facts
    where available (reviewer feedback: "Severity = Safe System + Helmet +
    Road Class [assumptions]" should become "Severity = Safe System +
    Median presence + Road width + ... using OSM-derived features").

    Starts from the ROAD_CLASS_SEVERITY_MAP prior (score_road_class_severity)
    — still needed as a fallback for segments with no OSM match — then
    applies point adjustments (config.INFRA_SEVERITY_ADJUSTMENTS) from real
    OSM way tags attached by enrichment.match_road_infrastructure():
      oneway=yes          → reduces severity (no head-on collision exposure)
      junction=roundabout → reduces severity (lower-energy crash types)
      lit=no / lit=yes     → increases / decreases severity
      unpaved surface      → increases severity (loss-of-control risk)
      lanes >= 4            → increases severity (wider road, higher design speed)

    Falls back to the prior alone (no adjustment) for any segment with no
    matched OSM way — exactly the same graceful-degradation pattern used
    everywhere else in this pipeline.
    """
    prior = score_road_class_severity(
        gdf.get("road_class_norm", pd.Series("unknown", index=gdf.index))
    )

    osm_cols = ["osm_oneway", "osm_junction", "osm_lit", "osm_surface", "osm_lanes"]
    if not any(c in gdf.columns for c in osm_cols):
        print("  [Severity] No OSM road infrastructure columns found — using "
              "the ROAD_CLASS_SEVERITY_MAP assumption for all segments. Run "
              "extract_osm_data.py, then enrichment.match_road_infrastructure(), "
              "for observed infrastructure data.")
        return prior

    adj     = pd.Series(0.0, index=gdf.index)
    matched = pd.Series(False, index=gdf.index)

    if "osm_oneway" in gdf.columns:
        m = gdf["osm_oneway"].notna() & (gdf["osm_oneway"] != "")
        adj += np.where(gdf["osm_oneway"] == "yes", INFRA_SEVERITY_ADJUSTMENTS["oneway_yes"], 0.0)
        matched |= m
    if "osm_junction" in gdf.columns:
        m = gdf["osm_junction"].notna() & (gdf["osm_junction"] != "")
        adj += np.where(gdf["osm_junction"] == "roundabout", INFRA_SEVERITY_ADJUSTMENTS["roundabout"], 0.0)
        matched |= m
    if "osm_lit" in gdf.columns:
        m = gdf["osm_lit"].notna() & (gdf["osm_lit"] != "")
        adj += np.where(gdf["osm_lit"] == "no",  INFRA_SEVERITY_ADJUSTMENTS["unlit"], 0.0)
        adj += np.where(gdf["osm_lit"] == "yes", INFRA_SEVERITY_ADJUSTMENTS["lit"], 0.0)
        matched |= m
    if "osm_surface" in gdf.columns:
        m = gdf["osm_surface"].notna() & (gdf["osm_surface"] != "")
        unpaved = gdf["osm_surface"].isin(["unpaved", "gravel", "dirt", "ground", "sand", "earth"])
        adj += np.where(unpaved, INFRA_SEVERITY_ADJUSTMENTS["unpaved"], 0.0)
        matched |= m
    if "osm_lanes" in gdf.columns:
        m = gdf["osm_lanes"].notna()
        adj += np.where(gdf["osm_lanes"].fillna(0) >= 4, INFRA_SEVERITY_ADJUSTMENTS["many_lanes"], 0.0)
        matched |= m

    n_matched = int(matched.sum())
    if getattr(score_infrastructure_severity, "_printed", False) is False:
        print(f"  [Severity] {n_matched:,} / {len(gdf):,} segments "
              f"({100*n_matched/max(len(gdf),1):.1f}%) have observed OSM "
              f"infrastructure data; remaining segments use the "
              f"ROAD_CLASS_SEVERITY_MAP assumption alone.")
        score_infrastructure_severity._printed = True

    return (prior + adj).clip(0, 100)


def score_helmet_severity(country_code: pd.Series, land_use: pd.Series) -> pd.Series:
    """
    Lower helmet-wearing rate (HELMET_SPI) means a crash at any given speed
    has a worse outcome. Same HELMET_SPI table scoring.py uses for VRU risk,
    but framed here as a severity OUTCOME modifier (this crash will be
    worse) rather than an exposure/who's-at-risk signal.
    """
    def _spi(cc, lu):
        return HELMET_SPI.get((cc, lu), HELMET_SPI.get((cc, "unknown"), 0.75))
    spi = pd.Series(
        [_spi(c, l) for c, l in zip(country_code, land_use)],
        index=country_code.index,
    )
    return (1.0 - spi) * 100


def compute_severity_score(gdf: gpd.GeoDataFrame, weights: dict = None) -> gpd.GeoDataFrame:
    """Severity = how bad is the crash if one occurs. Safe System gap is one
    input here, not the whole score — see SEVERITY_WEIGHTS in config.py."""
    if weights is None:
        weights = SEVERITY_WEIGHTS

    gdf  = gdf.copy()
    mask = gdf["scoreable"]

    # Safe System gap reuses scoring.py's alignment sub-score directly —
    # same number, reframed here as a severity input instead of the score itself.
    if "sub_score_limit_alignment" in gdf.columns:
        ss_score = gdf["sub_score_limit_alignment"]
    else:
        from scoring import score_speed_limit_alignment
        ss_score = pd.Series(np.nan, index=gdf.index)
        ss_score.loc[mask] = score_speed_limit_alignment(
            gdf.loc[mask, "speed_limit"], gdf.loc[mask, "ss_limit"]
        )

    nilsson_available = "nilsson_fatal_ratio" in gdf.columns
    nilsson_score = score_nilsson_severity(gdf["nilsson_fatal_ratio"]) if nilsson_available else None

    infra_score = score_infrastructure_severity(gdf)
    helmet_score = score_helmet_severity(
        gdf.get("country_code", pd.Series("unknown", index=gdf.index)),
        gdf.get("land_use",     pd.Series("unknown", index=gdf.index)),
    )

    components = {
        "safe_system_gap":         ss_score,
        "nilsson_risk":             nilsson_score,
        "infrastructure_severity": infra_score,
        "helmet_severity":          helmet_score,
    }
    combined, used_weights = _weighted_combine(components, weights)
    if "nilsson_risk" not in used_weights:
        print("  [Severity] nilsson_fatal_ratio not found — run "
              "advanced_scoring.nilsson() before priority_scoring. "
              "Weight redistributed across remaining components.")

    gdf.loc[mask, "sub_severity_safe_system"]     = ss_score[mask]
    gdf.loc[mask, "sub_severity_nilsson"]          = nilsson_score[mask] if nilsson_score is not None else np.nan
    gdf.loc[mask, "sub_severity_infrastructure"]   = infra_score[mask]
    gdf.loc[mask, "sub_severity_helmet"]           = helmet_score[mask]
    gdf.loc[mask, "severity_score"]                = combined[mask].clip(0, 100)
    return gdf


# ═══════════════════════════════════════════════════════════════════════════
# Combine layers — geometric mean
# ═══════════════════════════════════════════════════════════════════════════

def compute_priority_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Priority Index = (Exposure × Likelihood × Severity) ^ (1/3).
    See PRIORITY_INDEX_FLOOR in config.py for why a *small* floor is applied
    before the geometric mean (numerical safety only, not a policy choice).
    """
    gdf  = gdf.copy()
    mask = gdf["scoreable"]

    required = ["exposure_score", "likelihood_score", "severity_score"]
    missing  = [c for c in required if c not in gdf.columns]
    if missing:
        print(f"  [Priority Index] Missing required columns: {missing} — "
              f"skipping. Run enrich_segments(), compute_likelihood_score() "
              f"and compute_severity_score() first.")
        return gdf

    exposure   = gdf["exposure_score"].clip(lower=PRIORITY_INDEX_FLOOR)
    likelihood = gdf["likelihood_score"].clip(lower=PRIORITY_INDEX_FLOOR)
    severity   = gdf["severity_score"].clip(lower=PRIORITY_INDEX_FLOOR)

    priority = (exposure * likelihood * severity) ** (1 / 3)
    gdf.loc[mask, "priority_index"] = priority[mask].clip(0, 100)
    gdf.loc[mask, "priority_band"]  = gdf.loc[mask, "priority_index"].apply(_classify_priority_band)
    return gdf


def _classify_priority_band(score: float) -> str:
    if pd.isna(score):
        return "No Data"
    for band, (lo, hi) in PRIORITY_BANDS.items():
        if lo <= score < hi:
            return band
    crit_lo = PRIORITY_BANDS["Critical"][0]
    return "Critical" if score >= crit_lo else "Acceptable"


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def _check_exposure_degeneracy(gdf: gpd.GeoDataFrame, mask: pd.Series) -> None:
    if "exposure_score" not in gdf.columns:
        return
    exp = gdf.loc[mask, "exposure_score"]
    if len(exp) == 0:
        return
    near_zero_pct = (exp < 2).mean() * 100
    if near_zero_pct > 90:
        print(f"\n  ⚠ WARNING: {near_zero_pct:.0f}% of segments have an "
              f"Exposure Score < 2. Priority Index will be compressed toward "
              f"zero for almost everything. This usually means the optional "
              f"enrichment files (WorldPop tif, schools/hospitals/intersections "
              f"GeoJSON in enrichment_data/) aren't present AND weighted_sample "
              f"is missing or has very low variance. Check the [Exposure] "
              f"lines printed above before trusting Priority Index rankings.")


def compare_to_sss(gdf: gpd.GeoDataFrame) -> dict:
    """
    Quick rank comparison between the legacy SSS and the new Priority Index,
    so you can decide whether to keep, blend, or retire either one.

    Spearman rho close to 1 → the two methodologies broadly agree (Priority
    Index is mostly re-deriving the same ranking with better theoretical
    grounding). Low rho / low top-20% overlap → they're surfacing genuinely
    different roads — worth understanding WHY before picking one (e.g. is
    Priority Index catching high-exposure-but-moderate-speeding corridors
    that SSS's additive formula buried?).
    """
    if "priority_index" not in gdf.columns:
        return {}
    mask = gdf["scoreable"] & gdf["sss"].notna() & gdf["priority_index"].notna()
    df = gdf.loc[mask, ["sss", "priority_index"]]
    if len(df) < 10:
        print("  Too few segments to compare SSS vs Priority Index")
        return {}

    rho, _ = stats.spearmanr(df["sss"], df["priority_index"])
    n_top   = max(1, int(len(df) * 0.20))
    sss_top = set(df["sss"].nlargest(n_top).index)
    pi_top  = set(df["priority_index"].nlargest(n_top).index)
    overlap = len(sss_top & pi_top) / n_top * 100

    interp = ("Similar rankings — the two methods largely agree" if rho > 0.7 else
              "Moderate agreement — Priority Index adds some new information" if rho > 0.4 else
              "Materially different rankings — Priority Index surfaces different roads")

    print(f"\n  ── SSS vs Priority Index ──")
    print(f"  Spearman rho:     {rho:.3f}")
    print(f"  Top-20% overlap:  {overlap:.1f}%")
    print(f"  {interp}")
    return {"spearman_rho": round(float(rho), 4), "top20_overlap_pct": round(overlap, 1)}


def suggest_priority_bands(scores: pd.Series) -> dict:
    """
    Percentile-based band suggestion from a real run — the same approach
    used to recalibrate SCORE_BANDS for SSS in config.py v2.1. Paste the
    printed dict into config.PRIORITY_BANDS once you have real (non-demo)
    data.
    """
    s = scores.dropna()
    if len(s) < 20:
        print("  Too few scored segments to suggest bands")
        return {}
    p90, p70, p40 = s.quantile([0.90, 0.70, 0.40])
    bands = {
        "Critical":   (round(float(p90), 1), 100),
        "High Risk":  (round(float(p70), 1), round(float(p90), 1)),
        "Moderate":   (round(float(p40), 1), round(float(p70), 1)),
        "Acceptable": (0, round(float(p40), 1)),
    }
    print("\n  Suggested PRIORITY_BANDS (paste into config.py):")
    print("  PRIORITY_BANDS = {")
    for b, (lo, hi) in bands.items():
        print(f"      \"{b}\": ({lo}, {hi}),")
    print("  }")
    return bands


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════

def priority_sensitivity_analysis(
    gdf: gpd.GeoDataFrame,
    delta: float = 0.10,
    top_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Same idea as evaluation.sensitivity_analysis(), applied to the new
    Likelihood and Severity weight sets: perturb one weight ±delta at a
    time, renormalise, recompute, and measure Spearman rank stability of
    priority_index. (Exposure weights aren't perturbed here since
    enrichment.py recomputes exposure_score from raw spatial joins each
    time, which is expensive — Likelihood/Severity recompute cheaply from
    columns already on gdf.)
    """
    mask = gdf["scoreable"] & gdf.get("priority_index", pd.Series(dtype=float)).notna()
    if mask.sum() < 10:
        print("  Too few scored segments — skipping Priority Index sensitivity analysis")
        return pd.DataFrame()

    base_scores = gdf.loc[mask, "priority_index"].copy()
    n_top = max(1, int(len(base_scores) * top_pct))
    results = []

    weight_sets = {
        "likelihood": (LIKELIHOOD_WEIGHTS, compute_likelihood_score),
        "severity":   (SEVERITY_WEIGHTS,   compute_severity_score),
    }

    for layer_name, (base_w, compute_fn) in weight_sets.items():
        for perturb_key in base_w:
            for direction, sign in [("+", 1), ("-", -1)]:
                new_w = base_w.copy()
                new_w[perturb_key] = max(0.01, new_w[perturb_key] + sign * delta)
                total = sum(new_w.values())
                new_w = {k: v / total for k, v in new_w.items()}

                gdf_temp = compute_fn(gdf.copy(), weights=new_w)
                gdf_temp = compute_priority_index(gdf_temp)
                new_scores = gdf_temp.loc[mask, "priority_index"]

                rho, _ = stats.spearmanr(base_scores, new_scores)
                base_top = set(base_scores.nlargest(n_top).index)
                new_top  = set(new_scores.nlargest(n_top).index)
                pct_changed = 100 * len(base_top.symmetric_difference(new_top)) / n_top

                results.append({
                    "layer":                layer_name,
                    "perturbed_weight":      perturb_key,
                    "direction":             direction,
                    "new_value":             round(new_w[perturb_key], 3),
                    "spearman_rho_vs_base":  round(float(rho), 4),
                    f"pct_top{int(top_pct*100)}_changed": round(pct_changed, 1),
                })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print(f"\n── Priority Index Sensitivity Analysis (±{delta*100:.0f}% weight perturbation) ──")
        print(df_res.to_string(index=False))
        print(f"\nMean rank stability (ρ): {df_res['spearman_rho_vs_base'].mean():.4f}")
        print("(ρ > 0.95 = robust methodology)")
    return df_res


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_priority_scoring(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("\n" + "=" * 60)
    print("  PRIORITY INDEX — Exposure × Likelihood × Severity")
    print("=" * 60)

    gdf  = gdf.copy()
    mask = gdf["scoreable"]

    print("\n[1/3] Likelihood layer (speed gap + credibility + variability)...")
    gdf = compute_likelihood_score(gdf)

    print("\n[2/3] Severity layer (Safe System gap + Nilsson + road class + helmet)...")
    gdf = compute_severity_score(gdf)

    print("\n[3/3] Combining layers — geometric mean...")
    gdf = compute_priority_index(gdf)

    if "priority_index" in gdf.columns:
        scored = gdf.loc[mask & gdf["priority_index"].notna(), "priority_index"]
        if len(scored):
            print(f"\n  Priority Index distribution:")
            print(scored.describe().round(1))
            print(f"\n  Band distribution:")
            print(gdf.loc[mask, "priority_band"].value_counts())
            _check_exposure_degeneracy(gdf, mask)
            compare_to_sss(gdf)
        else:
            print("\n  No segments scored — check the [Priority Index] message above.")

    print("=" * 60)
    return gdf

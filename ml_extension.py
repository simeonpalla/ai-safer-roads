"""
ml_extension.py — XGBoost SSS predictor for unscored road segments.

PURPOSE: The ~55,000 unscored segments (lacking GPS behavioural data) still
have road attributes (class, land use, speed limit, Safe System limit,
exposure). A model trained on the 14,711 Tier-2-scored segments can predict
approximate SSS for these roads, extending coverage from 21% to ~100% of
the network.

HONEST CAVEATS:
  - Predictions replace field measurement, not complement it. Use for
    triage/prioritisation only — not for enforcement or policy claims.
  - The model is trained and tested on the SAME two countries. Generalisation
    to other countries is unknown.
  - SHAP values explain the model's prediction, not the road's real risk.
  - `ml_confidence` (std across 5 CV fold models) measures PREDICTION
    CONSISTENCY, not accuracy. Low confidence = models disagree; that
    doesn't tell you which model is right.
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

from sklearn.model_selection import KFold
import xgboost as xgb
# shap imported inside try/except block to handle XGBoost 2.x version mismatch gracefully

warnings.filterwarnings("ignore")

from logger import get_logger
from config import SCORE_BANDS
from scoring import get_safe_system_limit, classify_band

log = get_logger(__name__)

# Features split into two groups by availability:
#
# PRIMARY — always present for all 70k segments (scored + unscored).
# These are what the model uses when predicting unscored roads.
# Accuracy of predictions on unscored rows depends entirely on these.
PRIMARY_FEATURES = [
    "ss_limit",           # Safe System speed ceiling for this road class × land use
    "speed_limit",        # Posted limit (imputed for unscored if missing)
    "sinuosity",          # Road curvature — available from geometry for all segments
    "urban_pct",          # Urban percentage from ADB dataset
    "exposure_score",     # Composite exposure (traffic + schools + hospitals)
    "pop_density_500m",   # WorldPop density if available
    "dist_to_school_m",   # Distance to nearest school
    "dist_to_hospital_m", # Distance to nearest hospital
    "ghsl_settlement_code",
    "ntl_exposure_score",
    "infra_visibility_score",
    "dist_to_nearest_vru_attractor_m",
]

# BEHAVIOURAL — only present on Tier 2 GPS-measured rows (training data).
# NaN on all 45k unscored rows. XGBoost learns to route through PRIMARY_FEATURES
# when these are missing. Including them improves training fit and teaches the
# model the relationship between road attributes and speed behaviour.
# IMPORTANT: R² of 0.993 when these are present is DATA LEAKAGE (model is
# essentially reconstructing the SSS formula). True generalisation R² is
# measured by masking these out — see the held-out leakage test below.
BEHAVIOURAL_FEATURES = [
    "speed_85th",      # F85 — direct input to credibility sub-score (0.45 weight)
    "median_speed",    # Median speed — dual-signal credibility
    "credibility_gap", # F85 − posted limit (km/h) — already computed by advanced_scoring
    "pct_over_limit",  # Compliance proxy
]

NUMERIC_FEATURES = PRIMARY_FEATURES + BEHAVIOURAL_FEATURES
CAT_FEATURES = ["road_class_norm", "land_use", "ghsl_settlement_class"]

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=5,
    reg_alpha=0.1,          # L1 regularisation — helps when many features are NaN
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def _ensure_ss_limit(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute ss_limit for segments where it wasn't set by the scoring step."""
    if "ss_limit" not in gdf.columns:
        gdf["ss_limit"] = np.nan
    missing = gdf["ss_limit"].isna()
    if missing.any():
        gdf.loc[missing, "ss_limit"] = gdf[missing].apply(
            lambda r: get_safe_system_limit(
                r.get("road_class_norm", "unknown"),
                r.get("land_use", "unknown"),
            ),
            axis=1,
        )
    return gdf


def _build_feature_matrix(
    df: pd.DataFrame,
    cat_dummy_cols: list = None,
) -> tuple:
    """
    One-hot-encode categoricals + numeric stack.
    If cat_dummy_cols is provided, align to that column set (train→test consistency).
    Returns (X, cat_dummy_cols).
    """
    num = df.reindex(columns=NUMERIC_FEATURES).copy().astype(float)
    cat = pd.get_dummies(
        df.reindex(columns=CAT_FEATURES).fillna("unknown"),
        prefix=CAT_FEATURES,
        dtype=float,
    )
    if cat_dummy_cols is not None:
        cat = cat.reindex(columns=cat_dummy_cols, fill_value=0.0)
    X = pd.concat([num, cat], axis=1)
    return X, list(cat.columns)


def run_ml_extension(
    gdf: gpd.GeoDataFrame,
    output_dir: str = ".",
) -> gpd.GeoDataFrame:
    """
    Train XGBRegressor on Tier-2-scored segments; predict SSS for unscored
    segments that have at least a posted speed limit.
    Adds columns: ml_predicted_sss, ml_predicted_band, ml_shap_top_feature,
                  ml_confidence.
    """
    gdf = _ensure_ss_limit(gdf.copy())

    train_mask = gdf["scoreable"] & gdf["sss"].notna()
    # Include all unscored segments that have at least a usable road class
    # (from which ss_limit can be computed). Segments with no speed_limit
    # still get predictions — XGBoost handles NaN natively, using ss_limit
    # and road attribute features as the primary signal for those rows.
    pred_mask  = (
        ~gdf["scoreable"]
        & gdf["road_class_norm"].notna()
        & (gdf["road_class_norm"] != "unknown")
    )

    n_train = int(train_mask.sum())
    n_pred  = int(pred_mask.sum())

    log.info(f"\n{'='*60}")
    log.info("  ML COVERAGE EXTENSION -- XGBoost SSS Predictor")
    log.info(f"{'='*60}")
    log.info(f"  Training (Tier 2 scored):   {n_train:,}")
    log.info(f"  Prediction (unscored):      {n_pred:,}")

    if n_train < 50:
        log.warning("  Too few training samples -- skipping ML extension")
        return gdf

    # Impute speed_limit for unscored segments using the median posted/ss ratio
    # observed in training data for that road class × land use combination.
    # Rationale: without a posted limit, the most defensible assumption is
    # "typical for this road type in this dataset" — better than letting
    # XGBoost's NaN default direction push all predictions to one extreme.
    if "speed_limit" in NUMERIC_FEATURES:
        train_df = gdf.loc[train_mask, ["road_class_norm", "land_use", "ss_limit", "speed_limit"]].copy()
        train_df["ratio"] = train_df["speed_limit"] / train_df["ss_limit"].replace(0, np.nan)
        typical_ratio = train_df.groupby(["road_class_norm", "land_use"])["ratio"].median()
        overall_ratio = float(train_df["ratio"].median())

        missing_limit = pred_mask & (gdf["speed_limit"].isna() | (gdf["speed_limit"] == 0))
        if missing_limit.any():
            def _impute(row):
                r = typical_ratio.get((row.get("road_class_norm"), row.get("land_use")), overall_ratio)
                return float(row.get("ss_limit", 50)) * r
            gdf.loc[missing_limit, "speed_limit"] = gdf[missing_limit].apply(_impute, axis=1)
            gdf.loc[missing_limit, "_speed_limit_imputed"] = True
            n_imputed = int(missing_limit.sum())
            log.info(f"  Imputed speed_limit for {n_imputed:,} unscored segments "
                     f"(median posted/ss ratio by road class)")

    X_train, cat_cols = _build_feature_matrix(gdf[train_mask])
    y_train = gdf.loc[train_mask, "sss"].values.astype(float)

    # 5-fold CV — OOF predictions for RMSE + confidence baseline
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(n_train, np.nan)
    fold_models = []

    log.info(f"\n  5-fold cross-validation...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof[val_idx] = m.predict(X_train.iloc[val_idx])
        fold_models.append(m)
        rmse_fold = float(np.sqrt(np.mean((oof[val_idx] - y_train[val_idx]) ** 2)))
        log.info(f"    Fold {fold}: RMSE = {rmse_fold:.2f}")

    cv_rmse = float(np.sqrt(np.nanmean((oof - y_train) ** 2)))
    r2_with_gps = float(1 - np.nansum((oof - y_train)**2) / np.nansum((y_train - y_train.mean())**2))
    log.info(f"  CV RMSE (overall): {cv_rmse:.2f}   R² (with GPS features): {r2_with_gps:.3f}")

    # ── LEAKAGE-AWARE GENERALISATION TEST ────────────────────────────────────
    # R² of ~0.99 when speed_85th/median/credibility_gap are present is DATA
    # LEAKAGE — the model is essentially reconstructing the SSS formula from its
    # own inputs. The meaningful R² for predictions on UNSCORED roads (which
    # have no GPS data) is measured by masking out behavioural features and
    # re-running a quick CV on primary features only.
    log.info(f"\n  Leakage-aware generalisation test (primary features only)...")
    log.info(f"  (This simulates performance on unscored roads with no GPS data)")
    primary_cols = [c for c in X_train.columns
                    if not any(b in c for b in ["speed_85th", "median_speed",
                                                 "credibility_gap", "pct_over_limit"])]
    X_primary = X_train[primary_cols]
    oof_primary = np.full(n_train, np.nan)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
    for _, (tr_idx, val_idx) in enumerate(kf2.split(X_primary)):
        m2 = xgb.XGBRegressor(**XGB_PARAMS)
        m2.fit(X_primary.iloc[tr_idx], y_train[tr_idx])
        oof_primary[val_idx] = m2.predict(X_primary.iloc[val_idx])
    rmse_primary = float(np.sqrt(np.nanmean((oof_primary - y_train)**2)))
    r2_primary = float(1 - np.nansum((oof_primary - y_train)**2)
                       / np.nansum((y_train - y_train.mean())**2))
    log.info(f"  Primary-only RMSE: {rmse_primary:.2f}   R² (generalisation): {r2_primary:.3f}")
    log.info(f"  Gap (leakage effect): R² {r2_with_gps:.3f} → {r2_primary:.3f}")
    log.info(f"  NOTE: predictions on unscored roads use primary features only.")
    log.info(f"  R²={r2_primary:.3f} is the honest estimate of prediction quality on unscored network.")

    if r2_primary < 0.60:
        log.warning(
            f"  ⚠ R²={r2_primary:.3f} on primary features is low — predictions on "
            f"unscored roads will be approximate. Consider collecting more GPS data "
            f"or adding OSM infrastructure features (extract_osm_data.py)."
        )

    # OOF scatter plot — predicted vs actual SSS for all training segments
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_train, oof, alpha=0.25, s=10, color="#4f9fd4", linewidths=0)
        mn, mx = 0, 100
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual SSS (Tier 2 measured)", fontsize=11)
        ax.set_ylabel("Predicted SSS (OOF cross-validation)", fontsize=11)
        ax.set_title(
            f"XGBoost SSS Predictor — Validation\n"
            f"RMSE={cv_rmse:.2f}  R²(GPS)={r2_with_gps:.3f}  R²(generalisation)={r2_primary:.3f}",
            fontsize=11,
        )
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        ax.legend(fontsize=10)
        scatter_path = Path(output_dir) / "ml_validation_scatter.png"
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)
        log.info(f"  Saved: {scatter_path.name}")
    except Exception as e:
        log.warning(f"  (Scatter plot skipped: {e})")

    # ── TWO SEPARATE MODELS ───────────────────────────────────────────────────
    # 1. Full-feature model (GPS features included): used for CV R²/RMSE
    #    diagnostics and OOF scatter plot only. NOT used for predictions on
    #    unscored roads — behavioural features are NaN there.
    # 2. Primary-feature model: trained without GPS-behavioural features.
    #    This is what actually predicts SSS for the 45k unscored segments.
    #    R²=0.735 on primary features is the honest prediction performance.
    #
    # Why not just use the full model with NaN? XGBoost learns its NaN routing
    # direction from training data. When 100% of training rows have speed_85th,
    # it learns to route the NaN branch toward the base_score (~40), causing
    # all unscored predictions to collapse to ~40 or to extremes depending on
    # split direction. Training on primary features avoids this entirely.

    # Full-feature final model (diagnostics only)
    final_model_full = xgb.XGBRegressor(**XGB_PARAMS)
    final_model_full.fit(X_train, y_train)

    # Primary-feature final model (used for actual predictions on unscored roads)
    primary_cols_pred = [c for c in X_train.columns
                         if not any(b in c for b in ["speed_85th", "median_speed",
                                                      "credibility_gap", "pct_over_limit"])]
    X_train_primary = X_train[primary_cols_pred]
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train_primary, y_train)
    log.info(f"  Primary-feature prediction model trained ({len(primary_cols_pred)} features).")
    log.info(f"  Predictions on unscored roads use this model (R²={r2_primary:.3f}).")

    if n_pred == 0:
        log.warning("  No unscored segments with a posted limit -- nothing to predict")
        return gdf

    X_pred_full, _ = _build_feature_matrix(gdf[pred_mask], cat_dummy_cols=cat_cols)
    # Use only primary columns for prediction — matches what the model was trained on
    X_pred = X_pred_full.reindex(columns=primary_cols_pred, fill_value=0.0)

    # Confidence from fold models also rebuilt on primary features
    fold_models_primary = []
    kf3 = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, _ in kf3.split(X_train_primary):
        mp = xgb.XGBRegressor(**XGB_PARAMS)
        mp.fit(X_train_primary.iloc[tr_idx], y_train[tr_idx])
        fold_models_primary.append(mp)

    fold_preds = np.column_stack([mp.predict(X_pred) for mp in fold_models_primary])
    ml_pred    = np.clip(final_model.predict(X_pred), 0.0, 100.0)
    ml_conf    = fold_preds.std(axis=1)

    # SHAP on the primary-feature model (the one used for actual predictions)
    try:
        import shap as shap_lib
        log.info(f"  Computing SHAP values ({n_pred:,} segments)...")
        explainer   = shap_lib.TreeExplainer(final_model)
        explanation = explainer(X_pred)
        sv          = explanation.values
        feat_names  = list(X_pred.columns)
        top_idx     = np.abs(sv).argmax(axis=1)
        ml_shap_top = [feat_names[i] for i in top_idx]
    except ValueError as e:
        log.warning(
            f"  SHAP failed — XGBoost/SHAP version mismatch: {e}\n"
            f"  Fix: pip install --upgrade shap  (needs ≥ 0.45.0)\n"
            f"  ml_shap_top_feature set to 'unavailable'. All other ML outputs unaffected."
        )
        ml_shap_top = ["unavailable"] * n_pred
    except Exception as e:
        log.warning(f"  SHAP failed: {e}. Skipping.")
        ml_shap_top = ["unavailable"] * n_pred

    # Write back to gdf
    gdf.loc[pred_mask, "ml_predicted_sss"]    = ml_pred
    gdf.loc[pred_mask, "ml_predicted_band"]   = [classify_band(s) for s in ml_pred]
    gdf.loc[pred_mask, "ml_shap_top_feature"] = ml_shap_top
    gdf.loc[pred_mask, "ml_confidence"]       = ml_conf

    pred_bands = pd.Series([classify_band(s) for s in ml_pred])
    log.info(f"\n  ML-predicted band distribution ({n_pred:,} unscored segments):")
    log.info(f"  (Based on primary road attributes — no GPS data. R²(generalisation)={r2_primary:.3f})")
    for band in ["Critical", "High Risk", "Moderate", "Acceptable"]:
        c = (pred_bands == band).sum()
        log.info(f"    {band:12s}  {c:5,}  ({100*c/n_pred:.1f}%)")

    # Export
    export_cols = [c for c in [
        "segment_id", "country_code", "road_class_norm", "land_use",
        "speed_limit", "ss_limit",
        "ml_predicted_sss", "ml_predicted_band",
        "ml_shap_top_feature", "ml_confidence",
        "geometry",
    ] if c in gdf.columns]

    ml_gdf   = gdf.loc[pred_mask, export_cols].copy()
    out_gpkg = Path(output_dir) / "ml_coverage_extension.gpkg"
    out_csv  = Path(output_dir) / "ml_coverage_extension.csv"
    ml_gdf.to_file(out_gpkg, driver="GPKG")
    ml_gdf.drop(columns="geometry", errors="ignore").to_csv(out_csv, index=False)

    log.info(f"\n  Saved: {out_gpkg.name}  ({len(ml_gdf):,} segments)")
    log.info(f"  Saved: {out_csv.name}")
    log.info(f"{'='*60}")

    return gdf

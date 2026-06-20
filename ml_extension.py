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
import shap

warnings.filterwarnings("ignore")

from config import SCORE_BANDS
from scoring import get_safe_system_limit

NUMERIC_FEATURES = [
    "ss_limit",
    "speed_limit",
    "urban_pct",
    "exposure_score",
    "pop_density_500m",
    "dist_to_school_m",
    "dist_to_hospital_m",
]
CAT_FEATURES = ["road_class_norm", "land_use"]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def _classify_band(score: float) -> str:
    if pd.isna(score):
        return "No Data"
    for name, (lo, hi) in SCORE_BANDS.items():
        if lo <= score < hi:
            return name
    return "Critical" if score >= max(lo for lo, _ in SCORE_BANDS.values()) else "Acceptable"


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

    print(f"\n{'='*60}")
    print("  ML COVERAGE EXTENSION -- XGBoost SSS Predictor")
    print(f"{'='*60}")
    print(f"  Training (Tier 2 scored):   {n_train:,}")
    print(f"  Prediction (unscored):      {n_pred:,}")

    if n_train < 50:
        print("  Too few training samples -- skipping ML extension")
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
            print(f"  Imputed speed_limit for {n_imputed:,} unscored segments "
                  f"(median posted/ss ratio by road class)")

    X_train, cat_cols = _build_feature_matrix(gdf[train_mask])
    y_train = gdf.loc[train_mask, "sss"].values.astype(float)

    # 5-fold CV — OOF predictions for RMSE + confidence baseline
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(n_train, np.nan)
    fold_models = []

    print(f"\n  5-fold cross-validation...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof[val_idx] = m.predict(X_train.iloc[val_idx])
        fold_models.append(m)
        rmse_fold = float(np.sqrt(np.mean((oof[val_idx] - y_train[val_idx]) ** 2)))
        print(f"    Fold {fold}: RMSE = {rmse_fold:.2f}")

    cv_rmse = float(np.sqrt(np.nanmean((oof - y_train) ** 2)))
    print(f"  CV RMSE (overall): {cv_rmse:.2f}  (target < 15)")

    # Final model trained on all labelled data
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train, y_train)

    if n_pred == 0:
        print("  No unscored segments with a posted limit -- nothing to predict")
        return gdf

    X_pred, _ = _build_feature_matrix(gdf[pred_mask], cat_dummy_cols=cat_cols)

    # Predict with each fold model → confidence = std across fold predictions
    fold_preds  = np.column_stack([m.predict(X_pred) for m in fold_models])
    ml_pred     = np.clip(final_model.predict(X_pred), 0.0, 100.0)
    ml_conf     = fold_preds.std(axis=1)

    # SHAP — TreeExplainer, top-driving feature per segment
    print(f"  Computing SHAP values ({n_pred:,} segments)...")
    explainer   = shap.TreeExplainer(final_model)
    explanation = explainer(X_pred)
    sv          = explanation.values          # (n_pred, n_features)
    feat_names  = list(X_pred.columns)
    top_idx     = np.abs(sv).argmax(axis=1)
    ml_shap_top = [feat_names[i] for i in top_idx]

    # Write back to gdf
    gdf.loc[pred_mask, "ml_predicted_sss"]   = ml_pred
    gdf.loc[pred_mask, "ml_predicted_band"]  = [_classify_band(s) for s in ml_pred]
    gdf.loc[pred_mask, "ml_shap_top_feature"]= ml_shap_top
    gdf.loc[pred_mask, "ml_confidence"]      = ml_conf

    pred_bands = pd.Series([_classify_band(s) for s in ml_pred])
    print(f"\n  ML-predicted band distribution ({n_pred:,} unscored segments):")
    for band in ["Critical", "High Risk", "Moderate", "Acceptable"]:
        c = (pred_bands == band).sum()
        print(f"    {band:12s}  {c:5,}  ({100*c/n_pred:.1f}%)")

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

    print(f"\n  Saved: {out_gpkg.name}  ({len(ml_gdf):,} segments)")
    print(f"  Saved: {out_csv.name}")
    print(f"{'='*60}")

    return gdf

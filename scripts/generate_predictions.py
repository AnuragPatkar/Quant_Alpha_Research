"""
Standalone Production Inference Pipeline
========================================
Loads production-grade ML models and generates out-of-sample alpha signals.

Purpose
-------
This module orchestrates the generation of predictive alpha signals for a given set
of trading days. It applies the exact same preprocessing methodologies (Winsorization
and Sector Neutralization) utilized during model training to strictly prevent
look-ahead bias or distribution drift.

Role in Quantitative Workflow
-----------------------------
Executes daily (or on a defined schedule) as the core signal generation layer.
Consumes updated market data and outputs scored alpha signals to be natively
ingested by the portfolio optimization and order generation components.

Dependencies
------------
- **NumPy / Pandas**: Time-series manipulation and feature alignment.
- **Joblib**: Deserialization of production machine learning artifacts.
"""

from __future__ import annotations
import os
import sys
import joblib
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ingests shared primitives; executes train_models module-level code (CPU throttle, Numba warmup).
from scripts.train_models import (
    load_and_build_full_dataset,
    add_macro_features,
    build_ensemble_alpha,
    weighted_symmetric_mae,
    _warmup_numba,
)

# Monkey patch: Injects the custom objective function into the __main__ namespace to 
# guarantee deterministic resolution during Joblib unpickling of LightGBM/XGBoost artifacts.
import sys as _sys
_sys.modules["__main__"].weighted_symmetric_mae = weighted_symmetric_mae  # type: ignore
from config.settings import config
from quant_alpha.utils.preprocessing import WinsorisationScaler, SectorNeutralScaler
from quant_alpha.utils import setup_logging
from quant_alpha.utils import save_parquet, time_execution

setup_logging()
logger = logging.getLogger("Quant_Alpha")

# Guards against insufficient historical data for stable distribution parameter estimation
MIN_SCALER_DAYS = 252

# Unifies deserialization artifact nomenclature for standardized logging
_MODEL_NAME_MAP = {
    "lightgbm": "LightGBM",
    "xgboost":  "XGBoost",
    "catboost": "CatBoost",
}

@time_execution
def load_production_models() -> dict:
    """
    Deserializes the latest promoted production models from the local registry.

    Args:
        None

    Returns:
        dict: A dictionary mapping model names to their loaded artifact payloads,
            which include the model object and expected feature names.
    """
    models_dir = config.MODELS_DIR / "production"
    if not models_dir.exists():
        logger.error(f"Production models directory not found: {models_dir}")
        return {}

    model_files = list(models_dir.glob("*_latest.pkl"))
    if not model_files:
        logger.error(f"No *_latest.pkl files found in {models_dir}")
        return {}

    loaded = {}
    for path in model_files:
        try:
            stem       = path.stem.replace("_latest", "").lower()
            model_name = _MODEL_NAME_MAP.get(stem, stem.capitalize())
            payload    = joblib.load(path)

            if "model" not in payload or "feature_names" not in payload:
                logger.warning(f"Skipping {path.name}: missing 'model' or 'feature_names'.")
                continue

            loaded[model_name] = payload
            logger.info(
                f"✅ Loaded {model_name} "
                f"(trained to {payload.get('trained_to', 'unknown')}, "
                f"{len(payload['feature_names'])} features)"
            )
        except Exception as exc:
            logger.error(f"❌ Failed to load {path}: {exc}")

    return loaded

@time_execution
def preprocess_inference_data(
    data: pd.DataFrame,
    scaler_fit_start: pd.Timestamp,
    scaler_fit_end: pd.Timestamp,
    inference_start: pd.Timestamp,
    inference_end: pd.Timestamp,
    model_feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, "WinsorisationScaler"], dict[str, "SectorNeutralScaler"]]:
    """
    Applies point-in-time cross-sectional transformations to inference data.

    Fits scaling artifacts strictly on a historical lookback window to prevent 
    information leakage, then transforms the target inference window.

    Args:
        data (pd.DataFrame): The raw combined feature dataset.
        scaler_fit_start (pd.Timestamp): Start boundary for historical scaler fitting.
        scaler_fit_end (pd.Timestamp): End boundary for historical scaler fitting.
        inference_start (pd.Timestamp): Start boundary for the signal generation window.
        inference_end (pd.Timestamp): End boundary for the signal generation window.
        model_feature_sets (dict[str, list[str]]): Feature definitions mapped per model.

    Returns:
        tuple:
            - pd.DataFrame: The normalized and clipped feature subset.
            - dict[str, WinsorisationScaler]: Per-model winsorisation artifacts.
            - dict[str, SectorNeutralScaler]: Per-model sector neutralization artifacts.
    
    Raises:
        ValueError: If the required scaler fit window or inference window is completely empty.
    """
    logger.info(
        f"Scaler fit window : {scaler_fit_start.date()} → {scaler_fit_end.date()}"
    )
    logger.info(
        f"Inference window  : {inference_start.date()} → {inference_end.date()}"
    )

    # Slice only the required temporal bounds to bypass processing full 1M+ row dataset
    window_mask  = (data["date"] >= scaler_fit_start) & (data["date"] <= inference_end)
    window_data  = data[window_mask].copy().reset_index(drop=True)

    fit_mask     = window_data["date"] <= scaler_fit_end
    infer_mask   = window_data["date"] >= inference_start

    fit_df   = window_data[fit_mask]
    infer_df = window_data[infer_mask]

    # Regime-shift guard: Validates that the inference feature distributions do not 
    # differ significantly from the scaler fit window, flagging structural breaks.
    if not fit_df.empty and not infer_df.empty:
        meta_excl = {
            "ticker", "date", "target", "raw_ret_5d", "pnl_return",
            "open", "high", "low", "close", "volume",
            "sector", "industry", "index", "level_0",
        }
        check_cols = [
            c for c in fit_df.select_dtypes(include=[np.number]).columns
            if c not in meta_excl
        ][:10]
        if check_cols:
            fit_med   = fit_df[check_cols].median()
            infer_med = infer_df[check_cols].median()
            fit_std   = fit_df[check_cols].std().replace(0, 1e-8)
            shift     = ((infer_med - fit_med) / fit_std).abs().mean()
            if shift > 1.0:
                logger.warning(
                    f"[PREPROC] Regime shift detected: median feature shift = {shift:.2f}σ "
                    f"(fit window vs inference). Winsorisation bounds may be stale. "
                    f"Consider retraining or widening scaler_fit_window."
                )
            else:
                logger.info(f"[PREPROC] Distribution check: {shift:.3f}σ shift (OK < 1.0σ)")

    if fit_df.empty:
        raise ValueError(
            f"Scaler fit window is empty "
            f"({scaler_fit_start.date()} → {scaler_fit_end.date()}). "
            "Check that data covers this period."
        )
    if len(fit_df["date"].unique()) < MIN_SCALER_DAYS:
        logger.warning(
            f"Scaler fit window has only {fit_df['date'].nunique()} trading days "
            f"(minimum recommended: {MIN_SCALER_DAYS}). "
            "Statistics may be unstable."
        )
    if infer_df.empty:
        raise ValueError(
            f"Inference window is empty "
            f"({inference_start.date()} → {inference_end.date()})."
        )

    meta_exclude = {
        "ticker", "date", "target", "raw_ret_5d", "pnl_return",
        "open", "high", "low", "close", "volume",
        "sector", "industry", "index", "level_0",
        "next_open", "future_open",
    }

    wins_scalers: dict = {}
    norm_scalers: dict = {}

    # Dynamically preprocess each model uniquely based on its persisted feature dependencies
    # to avoid null propagation from models observing distinct metric combinations.
    processed = window_data.copy()

    for model_name, features in model_feature_sets.items():
        numeric = [
            f for f in features
            if f in window_data.columns
            and f not in meta_exclude
            and window_data[f].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        if not numeric:
            logger.warning(f"[{model_name}] No numeric features to scale.")
            continue

        # Extrapolates Winsorization bounds calibrated strictly from historical fit slice
        wins = WinsorisationScaler(clip_pct=0.01).fit(fit_df, numeric)
        wins_scalers[model_name] = wins

        fit_rows   = processed[fit_mask].copy()
        fit_rows   = wins.transform(fit_rows, numeric)

        # Interpolates trailing inference dates to the most proximal boundary configuration
        infer_rows = processed[infer_mask].copy()
        infer_rows = wins.transform(infer_rows, numeric)

        # Cross-sectional neutralization logic mapping
        norm = SectorNeutralScaler().fit(fit_df, numeric)
        norm_scalers[model_name] = norm

        fit_rows = norm.transform(fit_rows, numeric)

        # Bypasses date-lookup limitations via in-place standard scoring for live inference dates
        infer_rows = norm.inference_transform(infer_rows, numeric)

        processed.loc[fit_mask,   numeric] = fit_rows[numeric].values
        processed.loc[infer_mask, numeric] = infer_rows[numeric].values

    # Deterministic Categorical Imputation preventing string type-casting faults
    cat_cols = [
        c for c in processed.columns
        if c not in meta_exclude and processed[c].dtype == object
    ]
    processed[cat_cols] = processed[cat_cols].fillna("Unknown")
    processed[cat_cols] = processed[cat_cols].astype("category")

    logger.info("✅ Preprocessing complete.")
    return processed, wins_scalers, norm_scalers

@time_execution
def _find_last_predicted_date(predictions_dir) -> pd.Timestamp | None:
    """
    Scans the destination directory to resolve the latest generated signal date.

    Args:
        predictions_dir (Path): The file path to the saved prediction parquet files.

    Returns:
        pd.Timestamp | None: The most recent date successfully processed, or None 
            if the directory is empty.
    """
    pred_path = predictions_dir
    if not pred_path.exists():
        return None
    parquets = sorted(pred_path.glob("alpha_signals_*.parquet"))
    if not parquets:
        return None
    last = parquets[-1].stem.replace("alpha_signals_", "")
    try:
        return pd.to_datetime(last)
    except Exception:
        return None


def generate_predictions(
    predict_date: str | None = None,
    predict_days: int | None = None,
    last_day_only: bool = False,
) -> None:
    """
    Orchestrates the complete inference DAG: loads artifacts, preprocesses, predicts, and ensembles.

    Args:
        predict_date (str | None, optional): Specific isolated date to generate signals for (DD-MM-YYYY).
        predict_days (int | None, optional): Specifies the trailing N trading days to process.
        last_day_only (bool, optional): If True, processes strictly the latest date in the dataset.

    Returns:
        None
    """
    _warmup_numba()

    models = load_production_models()
    if not models:
        logger.error("Aborting: no production models found.")
        return

    data = load_and_build_full_dataset(force_rebuild=False)
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    data = add_macro_features(data)

    data_end  = data["date"].max()
    all_dates = sorted(pd.to_datetime(d) for d in data["date"].unique())

    if predict_date:
        inference_end   = pd.to_datetime(predict_date, dayfirst=True)
        inference_start = inference_end
        logger.info(f"[MODE] specific date: {inference_end.date()}")

    elif predict_days is not None:
        inference_end   = data_end
        inference_start = pd.Timestamp(all_dates[max(0, len(all_dates) - predict_days)])
        logger.info(f"[MODE] last {predict_days} trading days: {inference_start.date()} → {inference_end.date()}")

    elif last_day_only:
        inference_end   = data_end
        inference_start = data_end
        logger.info(f"[MODE] last day only: {inference_end.date()}")

    else:
        output_dir = config.PREDICTIONS_DIR
        last_pred  = _find_last_predicted_date(output_dir)
        inference_end = data_end
        if last_pred is not None:
            new_dates_list = [d for d in all_dates if d > last_pred]
            if not new_dates_list:
                logger.info(f"[MODE] Already up to date. Last prediction: {last_pred.date()}")
                return
            inference_start = new_dates_list[0]
            logger.info(
                f"[MODE] catch-up from {last_pred.date()} | "
                f"{len(new_dates_list)} new dates: {inference_start.date()} → {inference_end.date()}"
            )
        else:
            inference_start = all_dates[0]
            logger.info(f"[MODE] First run — predicting full history: {inference_start.date()} → {inference_end.date()}")

    scaler_fit_end   = inference_start - pd.Timedelta(days=1)
    lookback_years   = getattr(config, "INFERENCE_SCALER_LOOKBACK_YEARS", 3)
    scaler_fit_start = scaler_fit_end - pd.Timedelta(days=365 * lookback_years)

    # Validation Bound: Flags inversion logic to prevent look-ahead scaling contamination
    if inference_start <= scaler_fit_end:
        logger.warning(
            "inference_start is not after scaler_fit_end — "
            "winsorisation bounds may be applied from the wrong direction. "
            "Predictions may be less reliable."
        )

    data_min = data["date"].min()
    if scaler_fit_start < data_min:
        logger.warning(
            f"Requested scaler lookback starts at {scaler_fit_start.date()} "
            f"but data begins at {data_min.date()}. "
            "Using full available history for scaler fit."
        )
        scaler_fit_start = data_min

    logger.info(f"Prediction range  : {inference_start.date()} → {inference_end.date()}")

    model_feature_sets = {
        name: payload["feature_names"] for name, payload in models.items()
    }
    processed, wins_scalers, norm_scalers = preprocess_inference_data(
        data,
        scaler_fit_start, scaler_fit_end,
        inference_start,  inference_end,
        model_feature_sets,
    )

    inference_df = processed[
        (processed["date"] >= inference_start) &
        (processed["date"] <= inference_end)
    ].copy().reset_index(drop=True)

    if inference_df.empty:
        logger.error("Inference slice is empty. Aborting.")
        return

    del data, processed
    import gc; gc.collect()

    predictions: dict[str, pd.DataFrame] = {}
    for name, payload in models.items():
        model    = payload["model"]
        features = payload["feature_names"]

        missing = [f for f in features if f not in inference_df.columns]
        if missing:
            logger.warning(
                f"[{name}] {len(missing)} features missing from inference data "
                f"(e.g. {missing[:3]}). Skipping."
            )
            continue

        try:
            logger.info(f"[{name}] Generating predictions on {len(inference_df):,} rows…")
            raw_preds = model.predict(inference_df[features])

            preds_df = inference_df[["date", "ticker"]].copy()
            preds_df["prediction"] = raw_preds
            predictions[name] = preds_df

            logger.info(
                f"[{name}] pred range: "
                f"[{raw_preds.min():.4f}, {raw_preds.max():.4f}]  "
                f"mean={raw_preds.mean():.4f}"
            )
        except Exception as exc:
            logger.error(f"[{name}] Prediction failed: {exc}", exc_info=True)

    if not predictions:
        logger.error("No predictions generated. Aborting.")
        return

    logger.info(f"Ensembling {len(predictions)} models…")
    ensemble_df = build_ensemble_alpha(predictions)

    if ensemble_df.empty:
        logger.error("Ensemble produced empty output. Aborting.")
        return

    if "pnl_return" in inference_df.columns:
        ensemble_df = ensemble_df.merge(
            inference_df[["date", "ticker", "pnl_return"]],
            on=["date", "ticker"], how="left",
        )

    output_dir = config.PREDICTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"alpha_signals_{inference_end.strftime('%Y-%m-%d')}.parquet"
    save_parquet(ensemble_df, output_path)

    # Summary
    n_tickers = ensemble_df["ticker"].nunique()
    n_dates   = ensemble_df["date"].nunique()
    alpha_col = ensemble_df["ensemble_alpha"]
    print(f"\n{'='*55}")
    print("  INFERENCE COMPLETE")
    print(f"{'='*55}")
    print(f"  Dates     : {n_dates}  ({inference_start.date()} → {inference_end.date()})")
    print(f"  Tickers   : {n_tickers}")
    print(f"  Alpha     : mean={alpha_col.mean():.4f}  std={alpha_col.std():.4f}")
    print(f"  Saved to  : {output_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Alpha Signals from Production Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python scripts/generate_predictions.py                   # DEFAULT: all new dates since last run
  python scripts/generate_predictions.py --last-day        # only latest date
  python scripts/generate_predictions.py --days 10         # last 10 trading days
  python scripts/generate_predictions.py --date 15-06-2024 # specific date (DD-MM-YYYY)

Workflow:
  1. python scripts/update_data.py                         # fetch new market data
  2. python scripts/generate_predictions.py                # predict all new dates automatically
        """
    )
    parser.add_argument(
        "--last-day", action="store_true",
        help="Predict only the latest date in the dataset.",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Predict last N trading days.",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Predict a specific date in DD-MM-YYYY format.",
    )
    args = parser.parse_args()
    generate_predictions(
        predict_date=args.date,
        predict_days=args.days,
        last_day_only=args.last_day,
    )
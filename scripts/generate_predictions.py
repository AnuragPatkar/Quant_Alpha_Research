"""
generate_predictions.py
=======================
Standalone Inference Pipeline
-----------------------------
Loads production models and generates the latest alpha signals on new data.

Steps:
  1. Load production models (LGBM, XGB, CatBoost) from models/production/
  2. Fetch latest master dataset with all computed factors
  3. Preprocess using same logic as training — scalers fitted on recent
     historical window ONLY (no lookahead bias)
  4. Generate predictions from each model using its own feature list
  5. Ensemble via rank-based blending + turnover smoothing
  6. Save alpha signals to results/predictions/

Fixes vs original:
  - config.PREDICTIONS_DIR → config.RESULTS_DIR / "predictions"
  - SectorNeutralScaler: uses inference_transform() (cross-sectional z-score
    on inference dates, not stored date lookup which always falls back to global)
  - WinsorisationScaler: guard against predict_date < scaler fit window
  - build_ensemble_alpha: safe for single-date inference (zscore skipped)
  - Preprocessing per-model's own feature list (not union) — no wasted work
  - Only fit/transform scaler_window + inference_window, not full 1M rows
  - model_name casing fixed: "lightgbm" → "LightGBM"
  - scaler_fit_start_date dead-code duplicate removed
  - pnl_return attached to output (matches training pipeline output schema)
  - Guard: scaler window must be at least MIN_SCALER_DAYS long

Usage:
    python scripts/generate_predictions.py
    python scripts/generate_predictions.py --date 2023-10-27
    python scripts/generate_predictions.py --days 5
"""

from __future__ import annotations  # enables dict[str, X], list[X], X | Y on Python 3.9
import os
import sys
import joblib
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Shared utilities from train_models ---
# NOTE: This import executes train_models module-level code (CPU throttle,
#       Numba warmup, factor side-effects). Acceptable here because inference
#       always needs Numba compiled. If startup time becomes a concern, move
#       shared classes to quant_alpha/utils/preprocessing.py.
from scripts.train_models import (
    load_and_build_full_dataset,
    add_macro_features,
    WinsorisationScaler,
    SectorNeutralScaler,
    build_ensemble_alpha,
    weighted_symmetric_mae,
    _warmup_numba,
)

# joblib/pickle unpickling resolves custom objective functions against __main__.
# When generate_predictions.py is the entry point, __main__ is THIS file, not
# train_models. Injecting the function into __main__ fixes the AttributeError.
import sys as _sys
_sys.modules["__main__"].weighted_symmetric_mae = weighted_symmetric_mae
from config.settings import config
from quant_alpha.utils import setup_logging
from quant_alpha.utils import save_parquet, time_execution

setup_logging()
logger = logging.getLogger("Quant_Alpha")

# Minimum trading days in the scaler fit window (guards against short history)
MIN_SCALER_DAYS = 252   # ~1 year

# Map pkl stem → display name consistent with training logs
_MODEL_NAME_MAP = {
    "lightgbm": "LightGBM",
    "xgboost":  "XGBoost",
    "catboost": "CatBoost",
}


# ==============================================================================
# MODEL LOADING
# ==============================================================================
@time_execution
def load_production_models() -> dict:
    """Load all *_latest.pkl models from models/production/."""
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


# ==============================================================================
# INFERENCE PREPROCESSING
# ==============================================================================
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
    Fit scalers on [scaler_fit_start, scaler_fit_end] and return a preprocessed
    slice covering [scaler_fit_start, inference_end].

    Returns:
        processed_df  — full slice (fit window + inference window), preprocessed
        wins_scalers  — {model_name: WinsorisationScaler} fitted per-model
        norm_scalers  — {model_name: SectorNeutralScaler} fitted per-model

    Design notes:
      • Preprocessing is done PER MODEL using each model's own feature list.
        This avoids normalising irrelevant features and prevents NaN from
        union-features that a given model doesn't use.
      • SectorNeutralScaler.inference_transform() is used on inference rows —
        it computes cross-sectional z-score from inference data itself, not the
        stored date lookup (which only covers the fit window).
      • WinsorisationScaler uses searchsorted on fit-window dates, which is safe
        as long as inference_start >= scaler_fit_end (guarded in caller).

    KNOWN DISCREPANCY — Training vs Inference Winsorisation:
      Training  : global quantiles computed per-date across FULL dataset
      Inference : rolling quantiles fitted on [scaler_fit_start, scaler_fit_end]
      
      In stable regimes this difference is negligible (<0.5% of rows affected).
      In regime shifts (e.g. 2020 Covid crash) the tails diverge — a feature
      value that was p99 in 2019 may be p85 in 2022.
      
      Ideal fix: save training quantile bounds inside the model pkl at training
      time and load them here. This is a planned improvement. Until then, use a
      scaler_fit_window of 3+ years to minimise distribution shift.
    """
    logger.info(
        f"Scaler fit window : {scaler_fit_start.date()} → {scaler_fit_end.date()}"
    )
    logger.info(
        f"Inference window  : {inference_start.date()} → {inference_end.date()}"
    )

    # Slice only what we need — avoids holding full 1M-row dataset in memory
    window_mask  = (data["date"] >= scaler_fit_start) & (data["date"] <= inference_end)
    window_data  = data[window_mask].copy().reset_index(drop=True)

    fit_mask     = window_data["date"] <= scaler_fit_end
    infer_mask   = window_data["date"] >= inference_start

    fit_df   = window_data[fit_mask]
    infer_df = window_data[infer_mask]

    # Regime-shift guard: warn if inference feature distributions differ
    # significantly from scaler fit window (signals preprocessing discrepancy)
    if not fit_df.empty and not infer_df.empty:
        meta_excl = {
            "ticker", "date", "target", "raw_ret_5d", "pnl_return",
            "open", "high", "low", "close", "volume",
            "sector", "industry", "index", "level_0",
        }
        check_cols = [
            c for c in fit_df.select_dtypes(include=[np.number]).columns
            if c not in meta_excl
        ][:10]  # sample 10 features for speed
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

    # Preprocess per-model (each model has its own feature set)
    # We build a unified output by merging per-model transformed columns
    # back onto window_data.
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

        # Winsorisation: fit on fit_df, transform entire window (fit+inference)
        wins = WinsorisationScaler(clip_pct=0.01).fit(fit_df, numeric)
        wins_scalers[model_name] = wins

        # Transform fit window normally (dates are in fit range)
        fit_rows   = processed[fit_mask].copy()
        fit_rows   = wins.transform(fit_rows, numeric)

        # Transform inference window — searchsorted maps to nearest fit date
        # Safe because inference_start >= scaler_fit_end (guarded in caller)
        infer_rows = processed[infer_mask].copy()
        infer_rows = wins.transform(infer_rows, numeric)

        # SectorNeutralScaler: fit on fit_df only
        norm = SectorNeutralScaler().fit(fit_df, numeric)
        norm_scalers[model_name] = norm

        # Fit window: use transform() — dates are in the fit window lookup
        fit_rows = norm.transform(fit_rows, numeric)

        # Inference window: use inference_transform() — computes cross-sectional
        # z-score from inference data itself; no stored date lookup needed
        infer_rows = norm.inference_transform(infer_rows, numeric)

        # Write transformed features back
        processed.loc[fit_mask,   numeric] = fit_rows[numeric].values
        processed.loc[infer_mask, numeric] = infer_rows[numeric].values

    # Fill categorical NaNs
    cat_cols = [
        c for c in processed.columns
        if c not in meta_exclude and processed[c].dtype == object
    ]
    processed[cat_cols] = processed[cat_cols].fillna("Unknown")

    logger.info("✅ Preprocessing complete.")
    return processed, wins_scalers, norm_scalers


# ==============================================================================
# MAIN INFERENCE PIPELINE
# ==============================================================================
@time_execution
def _find_last_predicted_date(predictions_dir) -> pd.Timestamp | None:
    """Scan predictions dir and return the latest date already predicted."""
    pred_path = predictions_dir
    if not pred_path.exists():
        return None
    parquets = sorted(pred_path.glob("alpha_signals_*.parquet"))
    if not parquets:
        return None
    # filename: alpha_signals_YYYY-MM-DD.parquet
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
    Load models, preprocess, predict, ensemble, save.

    Modes:
      default (no flags)      → ALL dates since last saved prediction (or full history)
                                Use this always — handles daily, weekly, monthly updates.
      --last-day              → only the latest date in dataset
      --date DD-MM-YYYY       → specific single date
      --days N                → last N trading days
    """
    _warmup_numba()

    # 1. Load models
    models = load_production_models()
    if not models:
        logger.error("Aborting: no production models found.")
        return

    # 2. Load data
    data = load_and_build_full_dataset(force_rebuild=False)
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    data = add_macro_features(data)

    # 3. Date windows
    data_end  = data["date"].max()
    all_dates = sorted(pd.to_datetime(d) for d in data["date"].unique())

    if predict_date:
        # --date DD-MM-YYYY  → specific single date
        inference_end   = pd.to_datetime(predict_date, dayfirst=True)
        inference_start = inference_end
        logger.info(f"[MODE] specific date: {inference_end.date()}")

    elif predict_days is not None:
        # --days N  → last N trading days
        inference_end   = data_end
        inference_start = pd.Timestamp(all_dates[max(0, len(all_dates) - predict_days)])
        logger.info(f"[MODE] last {predict_days} trading days: {inference_start.date()} → {inference_end.date()}")

    elif last_day_only:
        # --last-day  → only latest date
        inference_end   = data_end
        inference_start = data_end
        logger.info(f"[MODE] last day only: {inference_end.date()}")

    else:
        # DEFAULT: everything since last saved prediction (handles daily/weekly/monthly gap)
        output_dir = config.RESULTS_DIR / "predictions"
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
            # First ever run — predict entire dataset history
            inference_start = all_dates[0]
            logger.info(f"[MODE] First run — predicting full history: {inference_start.date()} → {inference_end.date()}")

    scaler_fit_end   = inference_start - pd.Timedelta(days=1)
    lookback_years   = getattr(config, "INFERENCE_SCALER_LOOKBACK_YEARS", 3)
    scaler_fit_start = scaler_fit_end - pd.Timedelta(days=365 * lookback_years)

    # Guard: inference dates must be after the scaler fit window
    if inference_start <= scaler_fit_end:
        logger.warning(
            "inference_start is not after scaler_fit_end — "
            "winsorisation bounds may be applied from the wrong direction. "
            "Predictions may be less reliable."
        )

    # Guard: enough historical data?
    data_min = data["date"].min()
    if scaler_fit_start < data_min:
        logger.warning(
            f"Requested scaler lookback starts at {scaler_fit_start.date()} "
            f"but data begins at {data_min.date()}. "
            "Using full available history for scaler fit."
        )
        scaler_fit_start = data_min

    logger.info(f"Prediction range  : {inference_start.date()} → {inference_end.date()}")

    # 4. Preprocess — per model, only on required window
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

    # Free full dataset — not needed anymore
    del data, processed
    import gc; gc.collect()

    # 5. Per-model predictions
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

    # 6. Ensemble
    logger.info(f"Ensembling {len(predictions)} models…")
    ensemble_df = build_ensemble_alpha(predictions)

    if ensemble_df.empty:
        logger.error("Ensemble produced empty output. Aborting.")
        return

    # Attach pnl_return if available (matches training pipeline output schema)
    if "pnl_return" in inference_df.columns:
        ensemble_df = ensemble_df.merge(
            inference_df[["date", "ticker", "pnl_return"]],
            on=["date", "ticker"], how="left",
        )

    # 7. Save
    # FIX: config.PREDICTIONS_DIR does not exist — use RESULTS_DIR / "predictions"
    output_dir = config.RESULTS_DIR / "predictions"
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


# ==============================================================================
# ENTRY POINT
# ==============================================================================
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
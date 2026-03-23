"""
quant_alpha/preprocessing/__init__.py
======================================
Public API for the preprocessing package.

Exports
-------
WinsorisationScaler         — stateful per-date quantile clipping (fit on train only)
SectorNeutralScaler         — cross-sectional Z-score by (date, sector)
apply_preprocessing_pipeline — convenience wrapper used by trainer.py
enhance_fundamentals_for_registry — PiT merger called by DataManager / run_pipeline

Architecture contract (confirmed decisions — do not change)
-----------------------------------------------------------
1. WinsorisationScaler.fit() is called on train_df ONLY inside each fold.
   It is NEVER fitted on the full dataset before the fold split.
   Rationale: fitting on the full panel leaks future distribution into training.

2. The fitted WinsorisationScaler and SectorNeutralScaler are saved inside
   the model pkl so that inference uses the exact same bounds as training.

3. SectorNeutralScaler.transform() computes stats on the fly (stateless in time)
   so it can safely be applied at inference time on a single prediction batch.

4. clip_pct is read from config.WINSORIZE_QUANTILES — single source of truth.
   Never hardcode 0.01 anywhere else.
"""

from __future__ import annotations

# Core scalers — always available
# NOTE: WinsorisationScaler and SectorNeutralScaler live in
# quant_alpha/utils/preprocessing.py — NOT in this package.
# Import them directly:
#     from quant_alpha.utils.preprocessing import WinsorisationScaler, SectorNeutralScaler
# They are NOT re-exported from here to prevent circular imports and to keep
# the import path that all callers (train_models.py, generate_predictions.py,
# run_hyperopt.py, test_data_flow.py, test_production.py) already use.

# PiT merger — lives in this package (integration.py)
from .integration import enhance_fundamentals_for_registry

# Fundamental per-ticker preprocessor
from .fundamental_preprocessor import (
    preprocess_fundamentals,
    validate_preprocessed_data,
)

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def apply_preprocessing_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    sector_col: str = "sector",
    clip_pct: float | None = None,
):
    """
    Full fold-level preprocessing pipeline.

    Fit on train_df only → transform both train and test.
    This is the single correct call site for per-fold preprocessing.
    trainer.py calls this inside _train_single_model() for every fold.

    Steps
    -----
    1. WinsorisationScaler  — fit on train, clip outliers in both splits
    2. SectorNeutralScaler  — fit schema on train, Z-score both splits

    Parameters
    ----------
    train_df   : DataFrame containing training rows (date, ticker, features, sector)
    test_df    : DataFrame containing test / OOS rows
    features   : list of feature column names to preprocess
    sector_col : column name for sector grouping (default 'sector')
    clip_pct   : winsorization lower quantile (default: from config)

    Returns
    -------
    train_clean  : winsorized + Z-scored training DataFrame
    test_clean   : winsorized + Z-scored test DataFrame  (bounds from train)
    winsoriser   : fitted WinsorisationScaler  (save this in the model pkl)
    sector_scaler: fitted SectorNeutralScaler  (save this in the model pkl)

    Usage in trainer.py
    -------------------
        train_clean, test_clean, winsoriser, sector_scaler = (
            apply_preprocessing_pipeline(train_df, test_df, feature_cols)
        )
        # ... train model on train_clean ...
        # ... evaluate on test_clean ...
        # save: pkl['winsoriser'] = winsoriser
        # save: pkl['sector_scaler'] = sector_scaler
    """
    # Scalers live in quant_alpha/utils/preprocessing.py — import locally to
    # avoid circular import (utils.preprocessing → config → this package)
    from quant_alpha.utils.preprocessing import (  # noqa: PLC0415
        WinsorisationScaler,
        SectorNeutralScaler,
    )

    # Guard: only process columns that actually exist in both DataFrames
    present = [f for f in features
               if f in train_df.columns and f in test_df.columns]
    missing = set(features) - set(present)
    if missing:
        logger.warning(
            f"[Preprocessing] {len(missing)} feature(s) missing from one split "
            f"and will be skipped: {sorted(missing)[:10]}"
        )
    if not present:
        raise ValueError(
            "[Preprocessing] No valid feature columns found in both train and test splits."
        )

    # ---- Step 1: Winsorisation ----
    winsoriser = WinsorisationScaler(clip_pct=clip_pct)
    winsoriser.fit(train_df, present)               # fit on train ONLY
    train_w = winsoriser.transform(train_df, present)
    test_w  = winsoriser.transform(test_df,  present)

    # ---- Step 2: Sector-neutral Z-score ----
    sector_scaler = SectorNeutralScaler(sector_col=sector_col)
    sector_scaler.fit(train_w, present)             # registers schema on train
    train_clean = sector_scaler.transform(train_w,  present)
    test_clean  = sector_scaler.transform(test_w,   present)

    logger.info(
        f"[Preprocessing] Fold preprocessed: "
        f"train={len(train_clean):,} rows, test={len(test_clean):,} rows, "
        f"features={len(present)}"
    )
    return train_clean, test_clean, winsoriser, sector_scaler


def apply_inference_preprocessing(
    inference_df: pd.DataFrame,
    features: list[str],
    winsoriser,
    sector_scaler,
) -> pd.DataFrame:
    """
    Apply saved scaler objects to a live inference batch.

    Called by predictor.py. Uses the exact bounds from the training fold.
    Both scaler objects must be loaded from the model pkl.

    Parameters
    ----------
    inference_df  : DataFrame with the same feature schema as training
    features      : list of feature column names (must match training exactly)
    winsoriser    : WinsorisationScaler loaded from model pkl
    sector_scaler : SectorNeutralScaler loaded from model pkl

    Returns
    -------
    pd.DataFrame — preprocessed inference batch ready for model.predict()
    """
    if winsoriser is None or sector_scaler is None:
        raise ValueError(
            "[Preprocessing] winsoriser and sector_scaler must be loaded from "
            "the model pkl before calling apply_inference_preprocessing(). "
            "Missing scalers → inference receives unscaled features → CRITICAL."
        )

    present = [f for f in features if f in inference_df.columns]
    if not present:
        raise ValueError(
            "[Preprocessing] No valid feature columns found in inference DataFrame."
        )

    df_w = winsoriser.transform(inference_df, present)
    df_z = sector_scaler.inference_transform(df_w, present)
    return df_z


__all__ = [
    "WinsorisationScaler",
    "SectorNeutralScaler",
    "apply_preprocessing_pipeline",
    "apply_inference_preprocessing",
    "enhance_fundamentals_for_registry",
    "preprocess_fundamentals",
    "validate_preprocessed_data",
]
"""
Preprocessing & Feature Normalization Subsystem
===============================================

Provides the public API for cross-sectional feature preprocessing and 
Point-in-Time (PiT) data integration engines.

Purpose
-------
This module exports centralized transformation pipelines that guarantee strict 
out-of-sample data hygiene. It coordinates the application of stateful scalers 
(Winsorization and Sector-Neutral Z-Scoring) during both walk-forward training 
and live temporal inference.

Role in Quantitative Workflow
-----------------------------
Acts as the strict mathematical barrier preventing Look-Ahead Bias. Enforces 
the rigid architectural constraint that distribution percentiles must only be 
fitted on training folds and strictly never on the full temporal panel.
"""

from __future__ import annotations

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
    Orchestrates the full cross-sectional preprocessing pipeline bounded strictly by fold logic.

    Forces extraction scaling logic where statistical parameters are fitted strictly 
    on `train_df` before symmetrically transforming both the training and test sets. 
    This natively mitigates localized out-of-sample data leakage during walk-forward training.

    Args:
        train_df (pd.DataFrame): Distribution tensor bounding training horizons containing 
            strictly formatted (date, ticker, features, sector) matrices.
        test_df (pd.DataFrame): Sequential out-of-sample temporal holdout tensor.
        features (list[str]): String sequence matching target execution vectors.
        sector_col (str): Matrix key bounding exact grouping categorization. Defaults to "sector".
        clip_pct (Optional[float]): Standard percentage constraint dictating outlier distribution boundaries.

    Returns:
        Tuple: A tuple yielding the strictly neutralized training matrix, neutralized test matrix, 
            and both stateful fitted scaling engines (WinsorisationScaler, SectorNeutralScaler).
            
    Raises:
        ValueError: If overlapping features are entirely missing from target splits.
    """
    from quant_alpha.utils.preprocessing import (  # noqa: PLC0415
        WinsorisationScaler,
        SectorNeutralScaler,
    )

    # Resolves internal feature arrays confirming subset alignments across temporal panels
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

    # Step 1: Isolates training parameters fitting exact Winsorisation clipping barriers 
    winsoriser = WinsorisationScaler(clip_pct=clip_pct)
    winsoriser.fit(train_df, present)
    train_w = winsoriser.transform(train_df, present)
    test_w  = winsoriser.transform(test_df,  present)

    # Step 2: Implements continuous group neutralization scaling orthogonal components
    sector_scaler = SectorNeutralScaler(sector_col=sector_col)
    sector_scaler.fit(train_w, present)
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
    Projects historically fitted scaling barriers onto continuous live inference blocks.

    Forces exact alignment boundaries mapped historically during the original training 
    execution to guarantee mathematically equivalent standardized coordinates identically 
    evaluating localized predictions.

    Args:
        inference_df (pd.DataFrame): Incoming continuous tensor requiring normalization.
        features (list[str]): Target column execution keys mapping evaluation scopes.
        winsoriser: The persisted scaling model mitigating data outliers.
        sector_scaler: The persisted engine extracting relative cross-sectional variance.
        
    Returns:
        pd.DataFrame: A mathematically neutralized frame compatible with live model consumption.
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
"""
Data Persistence & Serialization Utilities
==========================================
Robust I/O layer for high-throughput market data and factor storage.

Purpose
-------
This module provides a unified interface for persisting Pandas DataFrames to
Apache Parquet format. It abstracts the complexity of engine selection and
implements a "Fail-Over" strategy to ensure data durability.

Usage
-----
.. code-block:: python

    # Persist factor matrix with fallback protection
    save_parquet(factor_df, "data/factors/momentum_v1.parquet")

    # Hydrate data for model training
    df = load_parquet("data/factors/momentum_v1.parquet")

Importance
----------
-   **Columnar Storage**: Parquet is optimized for OLAP workloads (factor analysis),
    allowing for efficient compression (Snappy) and column pruning.
-   **Schema Preservation**: Unlike CSV, Parquet preserves complex data types
    (e.g., datetime64[ns], float32), preventing type inference errors during ETL.
-   **Resilience**: The dual-engine approach (pyarrow -> fastparquet) mitigates
    runtime binary incompatibilities often found in containerized environments.

Tools & Frameworks
------------------
-   **PyArrow**: Primary high-performance C++ serialization engine.
-   **FastParquet**: Pure Python fallback engine.
-   **Pathlib**: OS-agnostic path manipulation.

NOTES
-----
  Reviewed at Session 10 — no logic bugs found.
  Minor improvement: save_parquet now returns the resolved Path so callers
  can log the exact path written. This is purely additive and non-breaking.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """
    Persists a DataFrame to disk using the Apache Parquet columnar format.

    Implements a redundant serialization strategy: attempts pyarrow first,
    then falls back to fastparquet.

    Args:
        df (pd.DataFrame): The data payload to serialize.
        path (Union[str, Path]): Target filesystem destination.

    Returns:
        Path: The resolved path that was written.

    Raises:
        Exception: If both primary and fallback engines fail.
    """
    path = Path(path)
    # Directory Provisioning: Ensure the target directory structure exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Primary Strategy: PyArrow (High performance C++ bindings)
    try:
        df.to_parquet(path, engine='pyarrow')
        logger.info(f"Saved {len(df):,} rows to {path} (pyarrow)")
        return path
    except Exception as e_pyarrow:
        logger.warning(
            f"pyarrow failed to save {path}: {e_pyarrow}. "
            "Retrying with fastparquet..."
        )

    # Redundancy Strategy: FastParquet (Pure Python implementation)
    try:
        df.to_parquet(path, engine='fastparquet')
        logger.info(f"Saved {len(df):,} rows to {path} (fastparquet)")
        return path
    except Exception as e_fastparquet:
        logger.error(
            f"Both engines failed to save {path}: "
            f"pyarrow: {e_pyarrow} | fastparquet: {e_fastparquet}"
        )
        raise


def load_parquet(
    path: Union[str, Path],
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Hydrates a DataFrame from an on-disk Parquet file.

    Implements the same dual-engine strategy as save_parquet.

    Args:
        path    : Parquet file path.
        columns : Optional list of column names to load (projection push-down).
                  If None, all columns are loaded.

    Returns:
        pd.DataFrame: The loaded data, or an empty DataFrame if the file is
                      missing or corrupt (Defensive Null Object pattern).
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Parquet file not found: {path}")
        return pd.DataFrame()

    kwargs = {}
    if columns is not None:
        kwargs['columns'] = columns

    # Primary Strategy: PyArrow
    try:
        df = pd.read_parquet(path, engine='pyarrow', **kwargs)
        logger.info(f"Loaded {len(df):,} rows from {path} (pyarrow)")
        return df
    except Exception as e_pyarrow:
        logger.warning(
            f"pyarrow failed to read {path}: {e_pyarrow}. "
            "Retrying with fastparquet..."
        )

    # Redundancy Strategy: FastParquet
    try:
        df = pd.read_parquet(path, engine='fastparquet', **kwargs)
        logger.info(f"Loaded {len(df):,} rows from {path} (fastparquet)")
        return df
    except Exception as e_fastparquet:
        logger.error(
            f"Both engines failed to read {path}: "
            f"pyarrow: {e_pyarrow} | fastparquet: {e_fastparquet}"
        )
        return pd.DataFrame()
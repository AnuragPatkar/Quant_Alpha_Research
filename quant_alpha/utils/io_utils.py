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
    (e.g., `datetime64[ns]`, `float32`), preventing type inference errors during ETL.
-   **Resilience**: The dual-engine approach (`pyarrow` $\to$ `fastparquet`) mitigates
    runtime binary incompatibilities often found in containerized environments.

Tools & Frameworks
------------------
-   **PyArrow**: Primary high-performance C++ serialization engine.
-   **FastParquet**: Pure Python fallback engine.
-   **Pathlib**: OS-agnostic path manipulation.
"""
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def save_parquet(df: pd.DataFrame, path: Union[str, Path]):
    """
    Persists a DataFrame to disk using the Apache Parquet columnar format.
    
    Implements a redundant serialization strategy: attempts to write using the
    optimized C++ `pyarrow` engine first, falling back to the pure-Python
    `fastparquet` engine upon binary failure.

    Args:
        df (pd.DataFrame): The data payload to serialize.
        path (Union[str, Path]): Target filesystem destination.
    
    Raises:
        Exception: If both primary and fallback engines fail to persist data.
    """
    try:
        path = Path(path)
        # Directory Provisioning: Ensure the target directory structure exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Primary Strategy: PyArrow (High performance C++ bindings)
        try:
            df.to_parquet(path, engine='pyarrow')
            logger.info(f"Saved data to {path} using pyarrow")
        except Exception as e_pyarrow:
            logger.warning(f"Pyarrow failed to save {path}: {e_pyarrow}. Retrying with fastparquet...")
            
            # Redundancy Strategy: FastParquet (Pure Python implementation)
            try:
                df.to_parquet(path, engine='fastparquet')
                logger.info(f"Saved data to {path} using fastparquet")
            except Exception as e_fastparquet:
                # Critical Failure: Data persistence impossible
                logger.error(f"Both engines failed to save {path}: {e_fastparquet}")
                raise
    except Exception as e:
        logger.error(f"Failed to save parquet to {path}: {e}")
        raise

def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Hydrates a DataFrame from an on-disk Parquet file.

    Returns:
        pd.DataFrame: The loaded data, or an empty DataFrame if the file is 
                      missing or corrupt (Defensive Null Object pattern).
    """
    try:
        path = Path(path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        
        # Primary Strategy: PyArrow
        try:
            df = pd.read_parquet(path, engine='pyarrow')
            logger.info(f"Loaded data from {path} using pyarrow")
            return df
        except Exception as e_pyarrow:
            logger.warning(f"Pyarrow failed to read {path}: {e_pyarrow}. Retrying with fastparquet...")
            
            # Redundancy Strategy: FastParquet
            try:
                df = pd.read_parquet(path, engine='fastparquet')
                logger.info(f"Loaded data from {path} using fastparquet")
                return df
            except Exception as e_fastparquet:
                logger.error(f"Both engines failed to read {path}: {e_fastparquet}")
                return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load parquet from {path}: {e}")
        return pd.DataFrame()
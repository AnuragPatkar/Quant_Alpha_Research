"""
File I/O utilities
"""
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def save_parquet(df: pd.DataFrame, path: Union[str, Path]):
    """
    Save DataFrame to Parquet with error handling.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try default engine (pyarrow)
        try:
            df.to_parquet(path, engine='pyarrow')
            logger.info(f"Saved data to {path} using pyarrow")
        except Exception as e_pyarrow:
            logger.warning(f"Pyarrow failed to save {path}: {e_pyarrow}. Retrying with fastparquet...")
            # Fallback to fastparquet
            try:
                df.to_parquet(path, engine='fastparquet')
                logger.info(f"Saved data to {path} using fastparquet")
            except Exception as e_fastparquet:
                logger.error(f"Both engines failed to save {path}: {e_fastparquet}")
                raise
    except Exception as e:
        logger.error(f"Failed to save parquet to {path}: {e}")
        raise

def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load DataFrame from Parquet with error handling.
    """
    try:
        path = Path(path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        
        # Try with default engine (pyarrow)
        try:
            df = pd.read_parquet(path, engine='pyarrow')
            logger.info(f"Loaded data from {path} using pyarrow")
            return df
        except Exception as e_pyarrow:
            logger.warning(f"Pyarrow failed to read {path}: {e_pyarrow}. Retrying with fastparquet...")
            # Fallback to fastparquet
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
"""
S&P 500 Point-in-Time Membership Mask Generator
===============================================
Creates a longitudinal boolean matrix to enforce point-in-time universe selection.

Purpose
-------
This script processes a chronological ledger of S&P 500 constituents and transforms
it into a structurally aligned boolean matrix. This matrix acts as a temporal mask,
guaranteeing that cross-sectional alpha modeling strictly observes index membership
at any historical timestamp. 

Importance
----------
- **Survivorship Bias Mitigation**: Natively embeds delisted and bankrupt entities 
  within the backtest universe by referencing exact historical index composition.
- **Data Leakage Prevention**: Enforces a strict point-in-time (PiT) constraint,
  preventing forward-looking selection bias from inflating backtest performance.

Role in Quantitative Workflow
-----------------------------
Executes as a foundational preprocessing step. The resulting materialized binary 
matrix (`sp500_membership_mask.pkl`) is consumed natively by the factor calculation 
layer to dynamically filter the active cross-sectional universe prior to Z-scoring.
"""

import pandas as pd
from pathlib import Path
import sys
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging

setup_logging()
logger = logging.getLogger("Quant_Alpha")

def create_membership_mask(input_csv: Path, output_pkl: Path):
    """
    Ingests raw longitudinal constituent data and constructs a dense boolean mask.

    Args:
        input_csv (Path): Absolute path to the raw S&P 500 historical composition file.
        output_pkl (Path): Absolute destination path for the serialized binary mask matrix.

    Returns:
        None: Materializes the matrix directly to the local storage tier.
    """
    logger.info(f"Loading historical S&P 500 components from: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logger.error(f"ERROR: Input file not found at {input_csv}")
        logger.error("This file is typically named 'S&P 500 Historical Components & Changes(...).csv'")
        logger.error("Please place it in the project root directory, similar to 'extract_ticker.py'.")
        return

    logger.info(f"Filtering for dates between {config.BACKTEST_START_DATE} and {config.BACKTEST_END_DATE}...")
    
    # Ensure temporal isolation by strictly bounding constituents to the backtest window
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= config.BACKTEST_START_DATE) & (df['date'] <= config.BACKTEST_END_DATE)
    df_filtered = df.loc[mask].copy()

    logger.info("Cleaning ticker symbols (removing suffixes like '-2015')...")
    
    # Eliminate NaNs to prevent type-casting errors during string splitting
    df_filtered = df_filtered.dropna(subset=['tickers'])

    def clean_ticker_list(ticker_string):
        """
        Strips class-share suffixes from ticker strings to align with canonical schema.

        Args:
            ticker_string (str): A comma-delimited string of daily constituent tickers.

        Returns:
            list: A parsed and cleansed list of base ticker symbols.
        """
        return [t.split('-')[0].strip() for t in str(ticker_string).split(',')]

    df_filtered['cleaned_tickers'] = df_filtered['tickers'].apply(clean_ticker_list)

    logger.info("Building boolean membership matrix...")
    
    # Unnest the arrays to establish a rigid row-per-ticker longitudinal structure
    exploded = df_filtered[['date', 'cleaned_tickers']].explode('cleaned_tickers')
    exploded = exploded.rename(columns={'cleaned_tickers': 'ticker'})
    exploded = exploded[exploded['ticker'] != '']
    exploded['is_member'] = True

    # Transpose the ledger into an O(1) look-up matrix (Date x Ticker) mapping index inclusion
    # Cast entirely to deterministic booleans, implicitly filling non-members as False
    membership_matrix = exploded.pivot_table(index='date', columns='ticker', values='is_member', aggfunc='first')
    membership_matrix = membership_matrix.fillna(False).astype(bool)

    # Persist the optimized binary matrix to disk to bypass compute latency on downstream loads
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    membership_matrix.to_pickle(output_pkl)
    logger.info(f"✅ Successfully saved membership mask to: {output_pkl}")
    logger.info(f"   Matrix shape: {membership_matrix.shape[0]} dates x {membership_matrix.shape[1]} tickers")

def main():
    """
    Primary execution routine for the point-in-time mask generation.
    """
    input_csv = config.PROJECT_ROOT / 'S&P 500 Historical Components & Changes(01-17-2026).csv'
    output_pkl = config.MEMBERSHIP_MASK_PATH
    create_membership_mask(input_csv, output_pkl)

if __name__ == "__main__":
    main()
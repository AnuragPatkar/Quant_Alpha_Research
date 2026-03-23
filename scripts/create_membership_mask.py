"""
S&P 500 Membership Mask Generator
=================================
Creates a boolean matrix indicating historical S&P 500 membership.

Purpose
-------
This script processes a raw CSV file containing historical S&P 500 constituents
and transforms it into a boolean "membership mask". This mask is a DataFrame
where rows are dates and columns are tickers. A cell is `True` if a ticker was
a member of the index on that date, and `False` otherwise.

This mask is crucial for backtesting and research to ensure that analysis is
performed only on the historically accurate universe of stocks, avoiding
survivorship bias.

Usage
-----
.. code-block:: bash

    python scripts/create_membership_mask.py

The script reads the historical components CSV from the project root and saves
the output to `data/processed/sp500_membership_mask.pkl` as defined in the config.
"""

import pandas as pd
from pathlib import Path
import sys
import logging

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging

setup_logging()
logger = logging.getLogger("Quant_Alpha")

def create_membership_mask(input_csv: Path, output_pkl: Path):
    """
    Loads historical S&P 500 component data, creates a boolean membership
    matrix, and saves it as a pickle file.
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
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= config.BACKTEST_START_DATE) & (df['date'] <= config.BACKTEST_END_DATE)
    df_filtered = df.loc[mask].copy()

    logger.info("Cleaning ticker symbols (removing suffixes like '-2015')...")
    df_filtered = df_filtered.dropna(subset=['tickers'])

    def clean_ticker_list(ticker_string):
        return [t.split('-')[0].strip() for t in str(ticker_string).split(',')]

    df_filtered['cleaned_tickers'] = df_filtered['tickers'].apply(clean_ticker_list)

    logger.info("Building boolean membership matrix...")
    exploded = df_filtered[['date', 'cleaned_tickers']].explode('cleaned_tickers')
    exploded = exploded.rename(columns={'cleaned_tickers': 'ticker'})
    exploded = exploded[exploded['ticker'] != '']
    exploded['is_member'] = True

    membership_matrix = exploded.pivot_table(index='date', columns='ticker', values='is_member', aggfunc='first')
    membership_matrix = membership_matrix.fillna(False).astype(bool)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    membership_matrix.to_pickle(output_pkl)
    logger.info(f"✅ Successfully saved membership mask to: {output_pkl}")
    logger.info(f"   Matrix shape: {membership_matrix.shape[0]} dates x {membership_matrix.shape[1]} tickers")

def main():
    """Main function to run the script."""
    input_csv = config.PROJECT_ROOT / 'S&P 500 Historical Components & Changes(01-17-2026).csv'
    output_pkl = config.MEMBERSHIP_MASK_PATH
    create_membership_mask(input_csv, output_pkl)

if __name__ == "__main__":
    main()
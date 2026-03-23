"""
Historical Universe Extraction Module
=====================================
Constructs the point-in-time investment universe to mitigate survivorship bias.

Purpose
-------
This script parses a longitudinal record of S&P 500 constituents and dynamically
filters it against the globally configured backtest date range. It extracts a
deduplicated list of all tickers that were members of the index at any point
during the specified period. This guarantees that downstream data ingestion and
factor modeling include delisted or bankrupt entities, satisfying institutional
look-ahead bias constraints.

Role in Quantitative Workflow
-----------------------------
Executed prior to the data acquisition phase. The resulting `sp500_tickers.txt`
serves as the definitive universe master list utilized by the overarching data
warehouse loaders (e.g., OHLCV and Fundamental acquisition).
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.settings import config

def main():
    """
    Extracts and normalizes historical S&P 500 constituents for a specified timeframe.

    Args:
        None

    Returns:
        None

    Raises:
        FileNotFoundError: If the historical constituents CSV file is missing from
            the project root directory.
    """
    input_csv = config.PROJECT_ROOT / 'S&P 500 Historical Components & Changes(01-17-2026).csv'
    output_txt = config.DATA_DIR / 'sp500_tickers.txt'

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_csv}")
        return

    print(f"Filtering for dates between {config.BACKTEST_START_DATE} and {config.BACKTEST_END_DATE}...")
    
    # Isolate constituents active during the configured backtest window
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= config.BACKTEST_START_DATE) & (df['date'] <= config.BACKTEST_END_DATE)
    df_filtered = df.loc[mask]

    print("Extracting and cleaning tickers...")
    
    # Aggregate a deduplicated superset of all historical tickers
    all_tickers = set()
    
    # Extract tickers, stripping multi-class share suffixes (e.g., 'BRK-B' -> 'BRK') to normalize ingestion
    for ticker_string in df_filtered['tickers'].dropna():
        tickers = [t.split('-')[0].strip() for t in str(ticker_string).split(',')]
        all_tickers.update(tickers)

    master_ticker_list = sorted(list(all_tickers))
    print(f"Found {len(master_ticker_list)} unique tickers to download.")

    print(f"Saving ticker list to: {output_txt}")
    # Persist the immutable universe list for downstream data pipelines
    with open(output_txt, 'w') as f:
        for ticker in master_ticker_list:
            f.write(f"{ticker}\n")
    
    print("Done.")

if __name__ == '__main__':
    main()
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.settings import config

def main():
    """
    Extracts historical S&P 500 tickers from a CSV file for a specified
    date range and saves them to a text file.
    """
    # Use pathlib to construct a path relative to this script's location.
    # This is more robust than a hardcoded absolute path.
    input_csv = config.PROJECT_ROOT / 'S&P 500 Historical Components & Changes(01-17-2026).csv'
    output_txt = config.DATA_DIR / 'sp500_tickers.txt'

    # Ensure output directory exists
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_csv}")
        return

    # 1. Filter using dates from settings.py dynamically
    print(f"Filtering for dates between {config.BACKTEST_START_DATE} and {config.BACKTEST_END_DATE}...")
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= config.BACKTEST_START_DATE) & (df['date'] <= config.BACKTEST_END_DATE)
    df_filtered = df.loc[mask]

    # 2. Extract all unique tickers
    print("Extracting and cleaning tickers...")
    all_tickers = set()
    # Use .dropna() to avoid errors if 'tickers' column has missing values
    for ticker_string in df_filtered['tickers'].dropna():
        # Split by comma and clean suffixes
        tickers = [t.split('-')[0].strip() for t in str(ticker_string).split(',')]
        all_tickers.update(tickers)

    master_ticker_list = sorted(list(all_tickers))
    print(f"Found {len(master_ticker_list)} unique tickers to download.")

    # 3. Save the list to a file
    print(f"Saving ticker list to: {output_txt}")
    with open(output_txt, 'w') as f:
        for ticker in master_ticker_list:
            f.write(f"{ticker}\n")
    
    print("Done.")

if __name__ == '__main__':
    main()
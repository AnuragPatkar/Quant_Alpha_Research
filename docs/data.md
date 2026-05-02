# Data Guide

> **Purpose**: Documentation of all data sources, schemas, validation rules, and how to work with the data pipeline.

---

## 1. Data Sources & Ingestion

### 1.1 Price Data (OHLCV)

**Source**: Yahoo Finance (`yfinance` library)  
**Update Frequency**: Daily (after market close, ~4 PM ET)  
**Universe**: S&P 500 constituents (dynamic, updated quarterly)  
**Storage Location**: `data/raw/sp500_prices/`

| Field | Type | Range | Notes |
|-------|------|-------|-------|
| `Date` | datetime64 | 2015-01-01 onwards | Trading days only (no weekends/holidays) |
| `Open` | float64 | > 0 | Opening price ($ per share) |
| `High` | float64 | ≥ Open | Intraday high |
| `Low` | float64 | ≤ Open | Intraday low |
| `Close` | float64 | > 0 | Closing price (adjusted for splits/dividends) |
| `Volume` | int64 | ≥ 0 | Shares traded (daily) |

**File Format**: CSV (one ticker per file)

```
Date,Open,High,Low,Close,Volume
2024-01-02,182.50,184.25,182.30,183.95,48230500
2024-01-03,184.00,185.50,183.75,184.25,42345670
```

**Data Quality Checks**:
- ✓ No NaN values in OHLCV columns
- ✓ High ≥ max(Open, Close)
- ✓ Low ≤ min(Open, Close)
- ✓ Volume ≥ 0
- ✓ Adjusted Close reflects splits and dividends

**Download Instructions**:

```bash
# Download all historical data
python scripts/download_data.py \
  --universe sp500 \
  --start-date 2015-01-01 \
  --data-types prices

# Update with latest data
python scripts/download_data.py \
  --universe sp500 \
  --data-types prices \
  --incremental
```

---

### 1.2 Fundamental Data (Quarterly)

**Source**: SimFin (free financial data API)  
**Update Frequency**: Quarterly (45-90 days after quarter-end)  
**Storage Location**: `data/raw/fundamentals/`  
**Reporting Lag**: 90 days (enforced by pipeline)

#### Balance Sheet

| Metric | Field Name | Unit | Reporting Lag |
|--------|-----------|------|---------------|
| Total Assets | `Total Assets` | USD | 90 days |
| Total Debt | `Total Debt` | USD | 90 days |
| Stockholders' Equity | `Stockholders Equity` | USD | 90 days |
| Current Assets | `Current Assets` | USD | 90 days |
| Current Liabilities | `Current Liabilities` | USD | 90 days |

**Computed Metrics** (in pipeline):

```python
debt_to_equity = total_debt / stockholders_equity
current_ratio = current_assets / current_liabilities
quick_ratio = (current_assets - inventory) / current_liabilities
```

#### Income Statement

| Metric | Field Name | Unit | Reporting Lag |
|--------|-----------|------|---------------|
| Total Revenue | `Revenue` | USD | 90 days |
| Operating Income | `Operating Income` | USD | 90 days |
| Net Income | `Net Income` | USD | 90 days |
| EBITDA | `EBITDA` | USD | 90 days |

**Computed Metrics**:

```python
gross_margin = gross_profit / revenue
operating_margin = operating_income / revenue
net_margin = net_income / revenue
roe = net_income / stockholders_equity
roa = net_income / total_assets
```

#### Cash Flow Statement

| Metric | Field Name | Unit | Reporting Lag |
|--------|-----------|------|---------------|
| Operating Cash Flow | `Operating Cash Flow` | USD | 90 days |
| Free Cash Flow | `Free Cash Flow` | USD | 90 days |
| Capital Expenditures | `CapEx` | USD | 90 days |
| Cash & Equivalents | `Cash` | USD | 90 days |

**Computed Metrics**:

```python
fcf_yield = free_cash_flow / market_cap
fcf_margin = free_cash_flow / revenue
cash_conversion = operating_cash_flow / net_income
```

#### Info Fields

| Metric | Field Name | Unit | Notes |
|--------|-----------|------|-------|
| EPS | `Earnings Per Share` | USD/share | Current fiscal year estimate |
| Forward EPS | `Forward EPS` | USD/share | Next fiscal year estimate |
| Price | `Last Price` | USD/share | Yahoo Finance daily close |
| Market Cap | `Market Capitalization` | USD | Computed: price × shares outstanding |
| P/E Ratio | `P/E Ratio` | ratio | Price / EPS |
| Forward P/E | `Forward P/E` | ratio | Price / Forward EPS |
| P/B Ratio | `Price-to-Book` | ratio | Price / (equity / shares) |
| P/S Ratio | `Price-to-Sales` | ratio | Market cap / revenue |
| Beta | `Beta` | ratio | 60-month market correlation |
| Book Value | `Book Value Per Share` | USD/share | Equity / shares outstanding |
| Dividend Yield | `Dividend Yield` | % | Annual dividend / price |

**Download Instructions**:

```bash
python scripts/download_data.py \
  --universe sp500 \
  --start-date 2015-01-01 \
  --data-types fundamentals

# Update quarterly
python scripts/download_data.py \
  --universe sp500 \
  --data-types fundamentals \
  --incremental
```

**Data Quality Issues & Fixes**:

From [Raw Columns Validation](../memories/user/quant_alpha_raw_columns_fix.md):
- ✓ Fixed: Column name mapping (space-separated vs. camelCase)
- ✓ Fixed: NaN columns from missing data sources
- ✓ Extraction validates key names with fallbacks
- ✓ Reporting lag enforced (90 days before data available)

---

### 1.3 Earnings Data (Event-Driven)

**Source**: Yahoo Finance / SEC EDGAR  
**Update Frequency**: Real-time (upon announcement)  
**Storage Location**: `data/raw/earnings/`

| Field | Type | Notes |
|-------|------|-------|
| `Date` | datetime64 | Earnings announcement date |
| `Ticker` | object | Stock symbol |
| `EPS` | float64 | Reported EPS for period |
| `EPS_Estimate` | float64 | Consensus estimate (prior announcement) |
| `Revenue` | float64 | Reported revenue (USD millions) |
| `Revenue_Estimate` | float64 | Consensus estimate |
| `Beat_Miss` | float64 | EPS surprise: (actual - estimate) / estimate |

**Computed Metrics** (in feature engineering):

```python
eps_surprise = (actual_eps - consensus_estimate) / abs(consensus_estimate)
revenue_surprise = (actual_revenue - consensus_estimate) / consensus_estimate
earnings_direction = sign(eps_surprise)
```

**Point-in-Time Alignment**:

```python
# Earnings data is event-driven, treated specially:
# - On announcement date: Use actual / surprise
# - Before announcement: Use consensus estimates
# - Implementation: merge_asof with backward-fill for estimates
```

---

### 1.4 Alternative Data (Macro & Market Indicators)

**Sources**: FRED, Yahoo Finance, Alternative data providers  
**Update Frequency**: Daily / Monthly as available

| Indicator | Ticker | Frequency | Provider |
|-----------|--------|-----------|----------|
| VIX | ^VIX | Daily | Yahoo Finance |
| S&P 500 | ^GSPC | Daily | Yahoo Finance |
| 10Y Treasury Yield | ^TNX | Daily | Yahoo Finance |
| USD Index | DXY | Daily | Yahoo Finance |
| Oil Price | CL=F | Daily | Yahoo Finance |
| 3M-10Y Term Spread | T10Y3M | Monthly | FRED |
| Corporate Spread (HY) | BAMLH0A0HYM2 | Monthly | FRED |
| Consumer Sentiment | UMCSENT | Monthly | FRED |

**Storage**: `data/raw/alternative/`

**Time Series Format**:

```
Date,VIX,SPX_Return,Term_Spread,HY_Spread,Sentiment
2024-01-02,12.45,0.0015,-0.35,3.20,67.5
2024-01-03,12.75,-0.0020,-0.40,3.25,67.8
```

---

## 2. Data Warehouse Schema

### 2.1 MultiIndex DataFrame Format

All data in pipeline uses **MultiIndex(date, ticker)** with data-aligned columns:

```python
import pandas as pd

# Example structure
data.index.names = ['date', 'ticker']
data.columns = ['open', 'high', 'low', 'close', 'volume', 'eps', 'pe_ratio', ...]

# Retrieve AAPL prices
aapl = data.xs('AAPL', level='ticker')

# Retrieve all tickers on specific date
latest = data.xs(data.index.get_level_values('date').max(), level='date')

# Retrieve time series for cross-sectional window
window = data.loc[pd.IndexSlice['2024-01-01':'2024-01-31', :], :]
```

### 2.2 Cache Storage (Parquet)

All raw CSVs are cached as **Parquet files** for fast reloads:

```
data/cache/
├── prices_raw.parquet              # OHLCV (500 tickers × 2500 dates)
├── fundamentals_raw.parquet        # Balance sheet, Income, Cash flow
├── earnings_raw.parquet            # Earnings events
├── alternative_raw.parquet         # Macro indicators
└── combined_price_fundamental.parquet  # Merged dataset (point-in-time)
```

**Advantages of Parquet**:
- ✓ 10x smaller than CSV (compression)
- ✓ Columnar format (fast column selection)
- ✓ Type preservation (no re-inferencing)
- ✓ Read in ~100-200ms (vs. 2-5s for CSV)

**Regenerate cache**:

```bash
# Force cache rebuild
python scripts/download_data.py --force-cache-rebuild
```

---

## 3. Validation Rules & Quality Assurance

### 3.1 Data Quality Gates

Applied **immediately after data load**:

```python
# From config/settings.py

DATA_VALIDATION = {
    'MIN_VALID_DATA_POINTS': 252,       # Minimum 1 year of history per ticker
    'MAX_NAN_FILL_LIMIT': 5,            # Max 5 day forward-fill allowed
    'MIN_VOLUME_THRESHOLD': 1_000_000,  # Minimum $1M avg daily volume
    'MIN_PRICE': 1.0,                   # Minimum $ price (filter penny stocks)
    'MAX_PRICE_JUMP': 0.30,             # Max 30% single-day price jump
    'REQUIRED_COLUMNS': ['open', 'high', 'low', 'close', 'volume'],
}
```

### 3.2 Survivorship Bias Correction

**Problem**: Using contemporaneous index constituents creates lookahead bias.

**Solution**: Dynamic membership mask tracking historical S&P 500 constituents.

```python
# Load membership mask
membership_mask = pd.read_pickle('data/processed/sp500_membership_mask.pkl')

# Filter to period's constituents
date_constituents = membership_mask.loc['2024-01-15']  # Get constituents on that date
valid_tickers = date_constituents[date_constituents == True].index

# Only use these tickers for backtest on that date
data_today = data.loc[('2024-01-15', valid_tickers), :]
```

**File Creation**:

```bash
python scripts/create_membership_mask.py \
  --constituents-csv "S&P 500 Historical Components & Changes(01-17-2026).csv" \
  --output data/processed/sp500_membership_mask.pkl
```

### 3.3 Missing Data Handling

**Philosophy**: Forward-fill fundamental data (updated quarterly), but drop price data gaps.

```python
# Price data: Drop any ticker-date with missing OHLCV
prices_clean = prices.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

# Fundamentals: Forward-fill quarterly snapshots to daily
fundamentals_filled = fundamentals.groupby('ticker').fillna(method='ffill', limit=90)
# (Limit 90 days = reporting lag + quarterly period)

# Check gap tolerance
max_gap = fundamentals_filled.groupby('ticker').apply(
    lambda x: x.isnull().sum().max()
)
assert max_gap.max() < 252, "Data gaps exceed tolerance"
```

### 3.4 Consistency Checks

**Run before each pipeline execution**:

```bash
# Validate data integrity
python scripts/diagnose_data.py

# Expected output:
# ✓ Prices loaded: 497 tickers × 2,515 trading days
# ✓ Date range: 2015-01-02 to 2026-01-17 (11+ years)
# ✓ No gaps > 5 trading days
# ✓ OHLC consistency: 100% pass
# ✓ Survivor mask: 497 live constituents
# ✓ Fundamental data: 28 quarterly snapshots (forward-filled)
# ✓ All data quality gates passed
```

---

## 4. Point-in-Time (PiT) Alignment & Look-Ahead Bias Prevention

### 4.1 The Problem

**Violation Example**: Using Q4 2023 earnings data in Jan 2024 feature computation before it's officially available.

```python
# ❌ WRONG: Trains on information not available at decision point
announcements_date = '2024-01-15'
available_earnings = earnings.loc[:announcements_date]  # All announcements up to Jan 15 ✗
```

### 4.2 The Solution: Reporting Lag

**Each data source has a statutory reporting lag**:

| Data Type | Reporting Lag | Example |
|-----------|---------------|---------|
| Q4 earnings | 60-90 days (Feb-Mar) | Used only after original filing date |
| Annual earnings | 60-90 days | |
| Quarterly 10-Q | 45 days | |
| Quarterly 10-K | 90 days | |
| Balance sheet snapshot | 90 days | Available 90 days after quarter-end |

**Implementation**:

```python
from datetime import timedelta

# In fundamental_preprocessor.py
REPORTING_LAG_DAYS = 90

# A Q4 2023 earnings announcement (filed 2024-02-15):
# Can only be used for features computed on 2024-05-15 or later
available_date = filing_date + timedelta(days=REPORTING_LAG_DAYS)

# Merge fundamental snapshots with 90-day lag
prices_raw = load_prices()
fundamentals_raw = load_fundamentals()

# Shift fundamentals forward by reporting lag
fundamentals_lagged = fundamentals_raw.set_index('date').shift(REPORTING_LAG_DAYS)

# Now merge: each date gets fundamentals from 90 days prior
merged = prices_raw.join(fundamentals_lagged, how='left')
# Date 2024-05-15 gets Q1 2024 fundamentals (filed in S/Q < 45 days after quarter-end)
```

### 4.3 Cross-Check: Embargo Periods

**Walk-Forward Training** adds another layer:

```python
# In trainer.py
EMBARGO_DAYS = 21  # 21 trading days

# For test fold starting 2024-06-01:
# Training data: [2023-01-01 : 2024-06-01 - 21 days] = [2023-01-01 : 2024-05-11]
# Test data:     [2024-06-01 : 2024-08-31]
# Models fit on data strictly before test period + embargo buffer
```

**Result**: No test-set leakage + fundamental data fully settled before use.

---

## 5. Working with Data Programmatically

### 5.1 Load Data for Analysis

```python
from quant_alpha.data import DataManager
import pandas as pd

dm = DataManager()

# Load all available data
data = dm.get_data(
    start_date='2023-01-01',
    end_date='2024-12-31',
    include_fundamentals=True,
    include_alternative=True,
    force_reload=False  # Use cache if available
)

print(data.info())
# MultiIndex: ['date', 'ticker']
# Columns: 120+ (prices, fundamentals, macro)
# Memory: ~5.2 GB (500 tickers × 500 days × 120 columns)
```

### 5.2 Filter & Resample

```python
# Get specific ticker
msft = data.xs('MSFT', level='ticker')
print(msft[['close', 'volume']].head())

# Filter date range
recent = data.loc[pd.IndexSlice['2024-06-01':, :], :]

# Resample to weekly
weekly = data.groupby('ticker').resample('W')['close'].last()

# Get cross-section on specific date
snapshot = data.xs('2024-01-15', level='date')
print(f"{len(snapshot)} stocks available on 2024-01-15")
```

### 5.3 Handle Missing Data

```python
# Inspect missing data
print(data.isnull().sum())

# Forward-fill short gaps (up to 5 days)
data_filled = data.groupby('ticker').fillna(method='ffill', limit=5)

# Drop rows with any NaN
data_clean = data.dropna()
print(f"Kept {len(data_clean) / len(data) * 100:.1f}% of data")
```

### 5.4 Compute Features

```python
# Returns (backward-looking)
data['ret_1d'] = data.groupby('ticker')['close'].pct_change(1)
data['ret_5d'] = data.groupby('ticker')['close'].pct_change(5)
data['ret_21d'] = data.groupby('ticker')['close'].pct_change(21)

# Forward returns (for targets)
data['ret_forward_5d'] = data.groupby('ticker')['close'].pct_change(5).shift(-5)

# Log returns
data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

# Rolling statistics
data['vol_21d'] = data.groupby('ticker')['ret_1d'].rolling(21).std()
data['mean_vol_21d'] = data.groupby('ticker')['volume'].rolling(21).mean()
```

---

## 6. FAQ & Troubleshooting

### Q: Why does my backtest exclude some S&P 500 tickers?

**Always check the membership mask**. The current S&P 500 has ~500 stocks, but historical constituents vary. A company delisted in 2020 is excluded from backtests on that date despite being in your CSV.

```python
# Debug: see which tickers were live
from pathlib import Path
mask = pd.read_pickle('data/processed/sp500_membership_mask.pkl')
alive_on_date = mask.loc['2024-01-15'][mask.loc['2024-01-15'] == True]
print(f"Alive on 2024-01-15: {len(alive_on_date)} constituents")
```

### Q: Why does fundamental data have gaps?

Companies file quarterly, not every day. The pipeline forward-fills quarterly snapshots to daily. If a company hasn't reported yet for a quarter, you get the prior quarter's data (clearly marked with forward-fill limit).

```python
# Check forward-fill depth
gaps = data['eps'].groupby('ticker').apply(
    lambda x: (x == x.shift(1)).sum()  # Count unchanged values
)
```

### Q: How fresh is the data?

```bash
# Check last update date
python -c "
import pandas as pd
from quant_alpha.data import DataManager
dm = DataManager()
data = dm.get_data()
last_date = data.index.get_level_values('date').max()
print(f'Last update: {last_date}')
"
```

### Q: Can I use this for intraday trading?

**No**, the data is daily closing prices. For intraday, you'd need minute or tick data (not currently supported).

---

For more details, see:
- [Setup & Installation](setup.md) — Data download instructions
- [Architecture](architecture.md) — Data layer design
- [FAQ](faq.md) — Common data questions

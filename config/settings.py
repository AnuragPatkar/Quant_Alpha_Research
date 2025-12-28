"""
Configuration Settings
"""

from dataclasses import dataclass
from typing import List, Dict
from datetime import date


@dataclass
class DataConfig:
    """Data related settings."""
    universe: str = "sp500"  # Changed to sp500 (more reliable)
    start_date: str = "2019-01-01"
    end_date: str = "2024-01-01"
    min_history_days: int = 252


@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    momentum_windows: tuple = (5, 10, 21, 63, 126, 252)
    volatility_windows: tuple = (10, 21, 63)
    rsi_windows: tuple = (14, 21)
    forward_return_days: int = 21


@dataclass 
class ModelConfig:
    """Model settings."""
    model_type: str = "lightgbm"
    lgb_params: Dict = None
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1
            }


@dataclass
class ValidationConfig:
    """Walk-forward validation settings."""
    train_months: int = 24
    test_months: int = 3
    embargo_days: int = 5
    min_train_samples: int = 500


@dataclass
class BacktestConfig:
    """Backtesting settings."""
    initial_capital: float = 10_000_000
    top_n_stocks: int = 10
    transaction_cost_bps: float = 20


@dataclass
class Settings:
    """Master config."""
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    validation: ValidationConfig = None
    backtest: BacktestConfig = None
    
    def __post_init__(self):
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.validation = ValidationConfig()
        self.backtest = BacktestConfig()


# Global settings
settings = Settings()


# ============================================
# STOCK UNIVERSES
# ============================================

# S&P 500 Top 50 (RELIABLE - USE THIS)
SP500_TOP50 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
    'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'MA', 'HD',
    'CVX', 'MRK', 'ABBV', 'LLY', 'PEP',
    'KO', 'PFE', 'COST', 'TMO', 'AVGO',
    'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
    'DHR', 'NEE', 'VZ', 'ADBE', 'TXN',
    'CRM', 'NKE', 'PM', 'INTC', 'QCOM',
    'UPS', 'HON', 'LOW', 'IBM', 'CAT',
    'GS', 'BA', 'AMD', 'ORCL', 'MS'
]

# Nifty 50 (May have issues)
NIFTY50_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
    'TITAN.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'ONGC.NS',
    'NTPC.NS', 'POWERGRID.NS', 'M&M.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS',
    'TATAMOTORS.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'HINDALCO.NS',
    'DRREDDY.NS', 'CIPLA.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'BPCL.NS',
    'BRITANNIA.NS', 'HEROMOTOCO.NS', 'INDUSINDBK.NS', 'EICHERMOT.NS',
    'TATACONSUM.NS', 'TECHM.NS', 'HCLTECH.NS', 'NESTLEIND.NS', 'HDFCLIFE.NS'
]


def get_universe(name: str) -> List[str]:
    """Get stock list based on universe name."""
    if name == "nifty50":
        return NIFTY50_STOCKS
    elif name == "sp500":
        return SP500_TOP50
    else:
        raise ValueError(f"Unknown universe: {name}")
"""
Configuration Settings
======================
Central configuration for ML Alpha Model.
Data downloaded from Stooq, stored in data/processed/
"""

from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path


# ============================================
# PROJECT ROOT
# ============================================
ROOT = Path(__file__).parent.parent


# ============================================
# DATA CONFIGURATION
# ============================================
@dataclass
class DataConfig:
    """Data pipeline settings."""
    
    # Universe
    universe: str = "sp500"
    
    # Paths
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    processed_dir: Path = field(default_factory=lambda: ROOT / "data" / "processed")
    raw_dir: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    
    # Data file (Stooq data)
    panel_file: str = "panel_sp500.pkl"
    
    # Date range
    start_date: str = "2020-01-01"
    end_date: str = "2025-01-01"
    
    # Requirements
    min_history_days: int = 252
    min_stocks: int = 20
    
    @property
    def panel_path(self) -> Path:
        """Full path to panel data."""
        return self.processed_dir / self.panel_file


# ============================================
# FEATURE CONFIGURATION
# ============================================
@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    
    # Momentum windows
    momentum_windows: tuple = (5, 10, 21, 63, 126)
    
    # Volatility windows
    volatility_windows: tuple = (10, 21, 63)
    
    # Mean reversion
    rsi_windows: tuple = (14, 21)
    ma_windows: tuple = (10, 21, 50)
    
    # Target variable
    forward_return_days: int = 21  # Predict 21-day returns
    
    # Data cleaning
    winsorize_limits: tuple = (0.01, 0.99)


# ============================================
# MODEL CONFIGURATION
# ============================================
@dataclass 
class ModelConfig:
    """ML Model settings."""
    
    model_type: str = "lightgbm"
    
    # LightGBM parameters
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


# ============================================
# VALIDATION CONFIGURATION
# ============================================
@dataclass
class ValidationConfig:
    """Walk-forward validation settings."""
    
    train_months: int = 18      # 18 months training
    test_months: int = 3        # 3 months testing
    embargo_days: int = 5       # Gap between train/test
    purge_window: int = 21      # Same as forward_return_days
    min_train_samples: int = 500


# ============================================
# BACKTEST CONFIGURATION
# ============================================
@dataclass
class BacktestConfig:
    """Backtesting settings."""
    
    initial_capital: float = 10_000_000  # 1 Crore INR / $10M
    
    # Portfolio
    top_n_long: int = 10        # Long top 10 stocks
    top_n_short: int = 0        # No shorting (long only)
    rebalance_frequency: str = "monthly"
    
    # Transaction costs (realistic)
    commission_bps: float = 5   # 0.05%
    slippage_bps: float = 10    # 0.10%
    
    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps
    
    @property
    def total_cost_pct(self) -> float:
        return self.total_cost_bps / 10000


# ============================================
# RISK CONFIGURATION
# ============================================
@dataclass
class RiskConfig:
    """Risk management settings."""
    
    max_position_pct: float = 0.10    # Max 10% per stock
    max_sector_pct: float = 0.30      # Max 30% per sector
    max_drawdown_pct: float = 0.20    # Stop at 20% drawdown
    volatility_target: float = 0.15   # Target 15% annual vol


# ============================================
# MASTER SETTINGS
# ============================================
@dataclass
class Settings:
    """Master configuration - combines all settings."""
    
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    validation: ValidationConfig = None
    backtest: BacktestConfig = None
    risk: RiskConfig = None
    
    # Output directories
    results_dir: Path = field(default_factory=lambda: ROOT / "results")
    plots_dir: Path = field(default_factory=lambda: ROOT / "results" / "plots")
    models_dir: Path = field(default_factory=lambda: ROOT / "results" / "models")
    reports_dir: Path = field(default_factory=lambda: ROOT / "results" / "reports")
    
    def __post_init__(self):
        """Initialize sub-configs."""
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.validation = ValidationConfig()
        self.backtest = BacktestConfig()
        self.risk = RiskConfig()
    
    def create_dirs(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.results_dir, self.plots_dir, 
                         self.models_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """Print current configuration."""
        print("\n" + "="*60)
        print("⚙️  CONFIGURATION")
        print("="*60)
        print(f"   Data Path: {self.data.panel_path}")
        print(f"   Universe: {self.data.universe}")
        print(f"   Date Range: {self.data.start_date} to {self.data.end_date}")
        print(f"   Forward Days: {self.features.forward_return_days}")
        print(f"   Model: {self.model.model_type}")
        print(f"   Train Months: {self.validation.train_months}")
        print(f"   Test Months: {self.validation.test_months}")
        print(f"   Top N Stocks: {self.backtest.top_n_long}")
        print(f"   Transaction Cost: {self.backtest.total_cost_bps} bps")


# ============================================
# GLOBAL SETTINGS INSTANCE
# ============================================
settings = Settings()


# ============================================
# STOCK UNIVERSE
# ============================================
STOCKS = [
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


def get_universe(name: str = "sp500") -> List[str]:
    """Get stock universe."""
    if name == "sp500":
        return STOCKS
    else:
        raise ValueError(f"Unknown universe: {name}")
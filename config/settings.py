"""
Configuration Settings
======================
Central configuration for ML Alpha Model.
Supports S&P 500 universe with 5-year historical data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import logging


# ============================================
# PROJECT ROOT
# ============================================
ROOT = Path(__file__).parent.parent


# ============================================
# LOGGING CONFIGURATION
# ============================================
@dataclass
class LogConfig:
    """Logging settings."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")
    log_file: str = "quant_alpha.log"
    
    def setup_logging(self):
        """Configure logging."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.level),
            format=self.format,
            handlers=[
                logging.FileHandler(self.log_dir / self.log_file),
                logging.StreamHandler()
            ]
        )


# ============================================
# DATA CONFIGURATION
# ============================================
@dataclass
class DataConfig:
    """Data pipeline settings."""
    
    # Universe
    universe: str = "sp500_top50"
    
    # Paths
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    processed_dir: Path = field(default_factory=lambda: ROOT / "data" / "processed")
    raw_dir: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    
    # Data file
    panel_file: str = "panel_sp500.pkl"
    
    # Date range (5 years - includes COVID crash and multiple regimes)
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    
    # Data source
    data_source: str = "yahoo"  # "yahoo" or "stooq"
    
    # Requirements
    min_history_days: int = 252  # ~1 year minimum
    min_stocks: int = 20
    
    # Data quality
    max_missing_pct: float = 0.10  # Max 10% missing data allowed
    
    @property
    def panel_path(self) -> Path:
        """Full path to panel data."""
        return self.processed_dir / self.panel_file
    
    def create_dirs(self):
        """Create data directories."""
        for dir_path in [self.data_dir, self.processed_dir, self.raw_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================
# FEATURE CONFIGURATION
# ============================================
@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    
    # Momentum windows (in trading days)
    momentum_windows: tuple = (5, 10, 21, 63, 126)  # 1w, 2w, 1m, 3m, 6m
    
    # Volatility windows
    volatility_windows: tuple = (10, 21, 63)  # 2w, 1m, 3m
    
    # Mean reversion
    rsi_windows: tuple = (14, 21)
    ma_windows: tuple = (10, 21, 50, 200)  # Added 200-day MA
    
    # Volume indicators
    volume_windows: tuple = (5, 10, 21)
    
    # Target variable
    forward_return_days: int = 21  # Predict 1-month returns
    
    # Data cleaning
    winsorize_limits: tuple = (0.01, 0.99)  # Remove extreme 1% outliers
    
    # Feature selection
    use_feature_selection: bool = True
    max_features: Optional[int] = 30  # Limit to top 30 features
    feature_importance_threshold: float = 0.001  # Min importance
    
    # Cross-sectional normalization
    normalize_cross_section: bool = True  # Rank/standardize within date


# ============================================
# MODEL CONFIGURATION
# ============================================
@dataclass 
class ModelConfig:
    """ML Model settings."""
    
    model_type: str = "lightgbm"  # "lightgbm" or "xgboost"
    random_seed: int = 42  # Global random seed
    
    # LightGBM parameters
    lgb_params: Dict = None
    
    # XGBoost parameters (if using XGBoost)
    xgb_params: Dict = None
    
    # Model tuning
    use_hyperparameter_tuning: bool = False
    cv_folds: int = 3  # For hyperparameter tuning only
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,  # L1 regularization
                "reg_lambda": 1.0,  # L2 regularization
                "min_child_samples": 20,
                "random_state": self.random_seed,
                "verbose": -1,
                "n_jobs": -1
            }
        
        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "reg:squarederror",
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_seed,
                "n_jobs": -1,
                "verbosity": 0
            }


# ============================================
# VALIDATION CONFIGURATION
# ============================================
@dataclass
class ValidationConfig:
    """Walk-forward validation settings."""
    
    # Window sizes (in months)
    train_months: int = 18      # 18 months training (optimal for 5-year data)
    test_months: int = 3        # 3 months testing
    step_months: int = 2        # Roll forward by 2 months
    
    # Purging (prevent data leakage)
    embargo_days: int = 21      # Must match forward_return_days
    purge_window: int = 21      # Same as forward_return_days
    
    # Quality checks
    min_train_samples: int = 500
    min_test_samples: int = 100
    
    # Validation metrics
    primary_metric: str = "ic"  # Information Coefficient (Spearman)
    secondary_metrics: tuple = ("rmse", "mae", "r2")


# ============================================
# BACKTEST CONFIGURATION
# ============================================
@dataclass
class BacktestConfig:
    """Backtesting settings."""
    
    # Capital
    initial_capital: float = 1_000_000  # $1M (more realistic for demo)
    
    # Portfolio construction
    top_n_long: int = 10        # Long top 10 stocks
    top_n_short: int = 0        # Long-only strategy
    weighting_scheme: str = "equal"  # "equal" or "prediction_weighted"
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # "monthly" or "weekly"
    
    # Transaction costs (U.S. market - realistic)
    commission_bps: float = 0.5   # 0.005% (modern discount brokers)
    slippage_bps: float = 5.0     # 0.05% (conservative for large-caps)
    market_impact_bps: float = 2.0  # 0.02% (minimal for S&P 500)
    
    @property
    def total_cost_bps(self) -> float:
        """Total transaction cost in basis points."""
        return self.commission_bps + self.slippage_bps + self.market_impact_bps
    
    @property
    def total_cost_pct(self) -> float:
        """Total transaction cost as percentage."""
        return self.total_cost_bps / 10000
    
    # Performance tracking
    benchmark: str = "SPY"  # S&P 500 ETF
    risk_free_rate: float = 0.04  # 4% annual (approximate)


# ============================================
# RISK CONFIGURATION
# ============================================
@dataclass
class RiskConfig:
    """Risk management settings."""
    
    # Position limits
    max_position_pct: float = 0.15    # Max 15% per stock (10 stocks)
    min_position_pct: float = 0.05    # Min 5% to maintain position
    
    # Sector limits (optional - requires sector data)
    use_sector_limits: bool = False
    max_sector_pct: float = 0.35      # Max 35% per sector
    
    # Stop losses (optional)
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.15       # -15% stop loss
    
    # Volatility targeting (optional)
    use_vol_targeting: bool = False
    volatility_target: float = 0.15   # Target 15% annual vol
    
    # Drawdown control
    max_drawdown_pct: float = 0.20    # Stop trading at -20% DD


# ============================================
# INTERPRETABILITY CONFIGURATION
# ============================================
@dataclass
class InterpretabilityConfig:
    """SHAP and model interpretability settings."""
    
    use_shap: bool = True
    shap_sample_size: int = 500  # Sample for SHAP (full data can be slow)
    
    # Feature importance
    plot_top_n_features: int = 20
    
    # Partial dependence
    use_pdp: bool = False  # Partial Dependence Plots (optional)
    pdp_features: tuple = ("mom_126", "rsi_14", "vol_21")


# ============================================
# MASTER SETTINGS
# ============================================
@dataclass
class Settings:
    """Master configuration - combines all settings."""
    
    # Sub-configurations
    log: LogConfig = None
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    validation: ValidationConfig = None
    backtest: BacktestConfig = None
    risk: RiskConfig = None
    interpretability: InterpretabilityConfig = None
    
    # Output directories
    results_dir: Path = field(default_factory=lambda: ROOT / "results")
    plots_dir: Path = field(default_factory=lambda: ROOT / "results" / "plots")
    models_dir: Path = field(default_factory=lambda: ROOT / "results" / "models")
    reports_dir: Path = field(default_factory=lambda: ROOT / "results" / "reports")
    
    # Execution flags
    verbose: bool = True
    save_predictions: bool = True
    save_models: bool = True
    
    def __post_init__(self):
        """Initialize sub-configs."""
        self.log = LogConfig()
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.validation = ValidationConfig()
        self.backtest = BacktestConfig()
        self.risk = RiskConfig()
        self.interpretability = InterpretabilityConfig()
    
    def create_dirs(self):
        """Create all output directories."""
        dirs = [
            self.results_dir, self.plots_dir, 
            self.models_dir, self.reports_dir,
            self.log.log_dir, self.data.data_dir,
            self.data.processed_dir, self.data.raw_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup(self):
        """Full setup: create dirs and configure logging."""
        self.create_dirs()
        self.log.setup_logging()
    
    def print_config(self):
        """Print current configuration."""
        print("\n" + "="*70)
        print("⚙️  ML ALPHA MODEL - CONFIGURATION")
        print("="*70)
        print(f"\n📊 DATA:")
        print(f"   Universe:        {self.data.universe}")
        print(f"   Date Range:      {self.data.start_date} → {self.data.end_date}")
        print(f"   Data Source:     {self.data.data_source}")
        print(f"   Panel File:      {self.data.panel_path}")
        
        print(f"\n🔧 FEATURES:")
        print(f"   Forward Days:    {self.features.forward_return_days}")
        print(f"   Momentum Windows: {self.features.momentum_windows}")
        print(f"   Max Features:    {self.features.max_features}")
        
        print(f"\n🤖 MODEL:")
        print(f"   Type:            {self.model.model_type}")
        print(f"   N Estimators:    {self.model.lgb_params['n_estimators']}")
        print(f"   Learning Rate:   {self.model.lgb_params['learning_rate']}")
        print(f"   Random Seed:     {self.model.random_seed}")
        
        print(f"\n📈 VALIDATION:")
        print(f"   Train Months:    {self.validation.train_months}")
        print(f"   Test Months:     {self.validation.test_months}")
        print(f"   Embargo Days:    {self.validation.embargo_days}")
        
        print(f"\n💼 BACKTEST:")
        print(f"   Initial Capital: ${self.backtest.initial_capital:,.0f}")
        print(f"   Top N Long:      {self.backtest.top_n_long}")
        print(f"   Rebalance:       {self.backtest.rebalance_frequency}")
        print(f"   Total Cost:      {self.backtest.total_cost_bps:.2f} bps")
        
        print(f"\n🛡️  RISK:")
        print(f"   Max Position:    {self.risk.max_position_pct*100:.0f}%")
        print(f"   Max Drawdown:    {self.risk.max_drawdown_pct*100:.0f}%")
        
        print("="*70 + "\n")
    
    def validate_config(self) -> bool:
        """Validate configuration consistency."""
        errors = []
        
        # Check embargo matches forward returns
        if self.validation.embargo_days != self.features.forward_return_days:
            errors.append(
                f"⚠️  embargo_days ({self.validation.embargo_days}) should match "
                f"forward_return_days ({self.features.forward_return_days})"
            )
        
        # Check position limits
        if self.backtest.top_n_long > 0:
            min_weight = 1.0 / self.backtest.top_n_long
            if self.risk.max_position_pct < min_weight:
                errors.append(
                    f"⚠️  max_position_pct ({self.risk.max_position_pct:.2%}) "
                    f"too low for {self.backtest.top_n_long} positions "
                    f"(need >{min_weight:.2%})"
                )
        
        # Check data path exists
        if not self.data.data_dir.exists():
            errors.append(f"⚠️  Data directory not found: {self.data.data_dir}")
        
        if errors:
            print("\n❌ Configuration Validation Errors:")
            for error in errors:
                print(f"   {error}")
            return False
        
        print("✅ Configuration validated successfully")
        return True


# ============================================
# GLOBAL SETTINGS INSTANCE
# ============================================
settings = Settings()


# ============================================
# STOCK UNIVERSE
# ============================================
# Top 50 S&P 500 stocks by market cap (as of 2020-2024)
# Note: BRK-B ticker corrected to BRK.B for Yahoo Finance
STOCKS_SP500_TOP50 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',  
    'JPM', 'V', 'PG', 'XOM', 'MA',  
    'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'TMO',  
    'WMT', 'MCD', 'CSCO', 'ACN', 'ABT',
    'DHR', 'NEE', 'VZ', 'ADBE', 'TXN',
    'CRM', 'NKE', 'PM', 'INTC', 'QCOM',
    'UPS', 'HON', 'LOW', 'IBM', 'CAT',
    'GS', 'BA', 'AMD', 'ORCL', 'MS'
]

# Alternative: Extend to top 100 or full 500 later
STOCKS_SP500_TOP100 = STOCKS_SP500_TOP50  # Placeholder for future


def get_universe(name: str = "sp500_top50") -> List[str]:
    """
    Get stock universe by name.
    
    Args:
        name: Universe identifier
        
    Returns:
        List of ticker symbols
        
    Raises:
        ValueError: If universe name is unknown
    """
    universes = {
        "sp500_top50": STOCKS_SP500_TOP50,
        "sp500": STOCKS_SP500_TOP50,  # Alias
        "sp500_top100": STOCKS_SP500_TOP100,
    }
    
    if name not in universes:
        raise ValueError(
            f"Unknown universe: {name}. "
            f"Available: {list(universes.keys())}"
        )
    
    return universes[name]


def get_feature_names() -> List[str]:
    """Generate expected feature names based on config."""
    features = []
    
    # Momentum features
    for window in settings.features.momentum_windows:
        features.append(f"mom_{window}")
        features.append(f"mom_{window}_rank")
    
    # Volatility features
    for window in settings.features.volatility_windows:
        features.append(f"vol_{window}")
        features.append(f"vol_{window}_rank")
    
    # Mean reversion
    for window in settings.features.rsi_windows:
        features.append(f"rsi_{window}")
    
    for window in settings.features.ma_windows:
        features.append(f"dist_ma_{window}")
    
    # Volume
    for window in settings.features.volume_windows:
        features.append(f"vol_zscore_{window}")
    
    return features


# ============================================
# UTILITY FUNCTIONS
# ============================================
def print_welcome():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        ML-BASED MULTI-FACTOR ALPHA MODEL                ║
    ║        S&P 500 Quantitative Research                    ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


if __name__ == "__main__":
    # Test configuration
    print_welcome()
    settings.print_config()
    settings.validate_config()
    
    print(f"\n📋 Stock Universe ({len(get_universe())} stocks):")
    print(f"   {', '.join(get_universe()[:10])}...")
    
    print(f"\n📊 Expected Features ({len(get_feature_names())} total):")
    print(f"   {', '.join(get_feature_names()[:10])}...")
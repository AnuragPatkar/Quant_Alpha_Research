"""
Configuration Settings
======================
Central config file for the ML Alpha Model.
Handles all settings for data, features, models, backtesting, etc.

Author: [Your Name]
Last Updated: 2024

IMPORTANT WARNINGS:
- See SURVIVORSHIP_BIAS_WARNING for universe limitations
- This config is for research/demo purposes
- Do NOT use backtest results for live trading decisions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import logging
import warnings


# ============================================
# PROJECT ROOT
# ============================================
ROOT = Path(__file__).parent.parent


# ============================================
# SURVIVORSHIP BIAS WARNING
# ============================================
SURVIVORSHIP_BIAS_WARNING = """
⚠️  SURVIVORSHIP BIAS WARNING ⚠️
================================
The stock universe (STOCKS_SP500_TOP50) is based on 2024 market caps.
This introduces LOOKAHEAD BIAS when backtesting from 2020.

Implications:
- Backtest returns are ARTIFICIALLY INFLATED
- Winners (NVDA, TSLA) are included; losers are excluded
- Results are NOT suitable for production trading decisions

For valid research, use point-in-time constituent data from:
- Sharadar, Compustat, Bloomberg, or similar providers

This static list is ONLY for code testing/demo purposes.
"""


# ============================================
# LOGGING CONFIGURATION
# ============================================
@dataclass
class LogConfig:
    """Settings for logging."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")
    log_file: str = "quant_alpha.log"
    
    def setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.
        
        Returns:
            Logger: Configured root logger
        """
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.level))
        
        # Clear existing handlers (allows reconfiguration)
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / self.log_file)
        file_handler.setLevel(getattr(logging, self.level))
        file_handler.setFormatter(logging.Formatter(self.format))
        
        # Stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(getattr(logging, self.level))
        stream_handler.setFormatter(logging.Formatter(self.format))
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        return logger


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
    # Note: end_date will be capped to today if in future
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
    
    @property
    def effective_end_date(self) -> str:
        """
        Returns end_date capped to today if it's in the future.
        
        Returns:
            str: Effective end date in YYYY-MM-DD format
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self.end_date > today:
            return today
        return self.end_date
    
    def create_dirs(self) -> None:
        """Create data directories if they don't exist."""
        for dir_path in [self.data_dir, self.processed_dir, self.raw_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================
# FEATURE CONFIGURATION
# ============================================
@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    
    # Momentum windows (in trading days)
    # 1w, 2w, 1m, 3m, 6m
    momentum_windows: tuple = (5, 10, 21, 63, 126)
    
    # Volatility windows (2w, 1m, 3m)
    volatility_windows: tuple = (10, 21, 63)
    
    # Mean reversion indicators
    rsi_windows: tuple = (14, 21)
    ma_windows: tuple = (10, 21, 50, 200)
    
    # Volume indicators
    volume_windows: tuple = (5, 10, 21)
    
    # Target variable
    forward_return_days: int = 10  # Predict 1-month forward returns
    
    # Data cleaning
    winsorize_limits: tuple = (0.01, 0.99)  # Remove extreme 1% outliers
    
    # Feature selection
    use_feature_selection: bool = True
    max_features: Optional[int] = 30  # Limit to top N features
    feature_importance_threshold: float = 0.001  # Min importance to keep
    
    # Cross-sectional normalization
    normalize_cross_section: bool = True  # Rank/standardize within each date


# ============================================
# MODEL CONFIGURATION
# ============================================
@dataclass 
class ModelConfig:
    """ML Model settings."""
    
    model_type: str = "lightgbm"  # "lightgbm" or "xgboost"
    random_seed: int = 42
    
    # LightGBM parameters
    lgb_params: Dict = field(default_factory=dict)
    
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=dict)
    
    # Hyperparameter tuning
    use_hyperparameter_tuning: bool = False
    cv_folds: int = 3  # For tuning only, not walk-forward
    
    def __post_init__(self):
        """Set default params only if not provided."""
        # LightGBM defaults
        # LightGBM defaults - HEAVILY REGULARIZED for noisy data
        default_lgb = {
            "objective": "regression",
            "metric": "mae",                # Changed: rmse -> mae (robust)
            "boosting_type": "gbdt",
            "n_estimators": 100,            # Changed: 300 -> 100
            "max_depth": 3,                 # Changed: 5 -> 3 (shallow)
            "learning_rate": 0.01,          # Changed: 0.05 -> 0.01 (slower)
            "num_leaves": 8,                # Changed: 31 -> 8
            "subsample": 0.5,               # Changed: 0.8 -> 0.5
            "subsample_freq": 1,
            "colsample_bytree": 0.5,        # Changed: 0.8 -> 0.5
            "reg_alpha": 5.0,               # Changed: 0.1 -> 5.0 (50x more L1)
            "reg_lambda": 50.0,             # Changed: 1.0 -> 50.0 (50x more L2)
            "min_child_samples": 100,       # Changed: 20 -> 100
            "random_state": self.random_seed,
            "verbose": -1,
            "n_jobs": -1
        }
        
        # XGBoost defaults
        default_xgb = {
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
        
        # Only update with defaults for missing keys
        for key, value in default_lgb.items():
            if key not in self.lgb_params:
                self.lgb_params[key] = value
                
        for key, value in default_xgb.items():
            if key not in self.xgb_params:
                self.xgb_params[key] = value


# ============================================
# VALIDATION CONFIGURATION
# ============================================
@dataclass
class ValidationConfig:
    """Walk-forward validation settings."""
    
    # Window strategy
    use_expanding_window: bool = True  # Recommended: use all past data
    
    # Window sizes (in months)
    # For rolling window (if use_expanding_window=False)
    train_months: int = 36  # 3 years training (increased from 18)
    
    # For expanding window
    min_train_months: int = 24  # Minimum 2 years before first prediction
    
    # Test and step
    test_months: int = 3  # 3 months testing
    step_months: int = 3  # Roll forward by 3 months (no overlap)
    
    # Purging and embargo (prevent lookahead bias)
    embargo_days: int = 21  # Must match forward_return_days
    purge_window: int = 21  # Gap between train and test
    
    # Quality checks
    min_train_samples: int = 500
    min_test_samples: int = 100
    
    # Validation metrics
    primary_metric: str = "ic"  # Information Coefficient (Spearman)
    secondary_metrics: tuple = ("rmse", "mae", "r2", "ic_ir")
    
    def validate(self, forward_return_days: int) -> List[str]:
        """
        Validate configuration consistency.
        
        Args:
            forward_return_days: From FeatureConfig
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.embargo_days < forward_return_days:
            errors.append(
                f"embargo_days ({self.embargo_days}) should be >= "
                f"forward_return_days ({forward_return_days}) to prevent lookahead"
            )
        
        if self.purge_window < forward_return_days:
            errors.append(
                f"purge_window ({self.purge_window}) should be >= "
                f"forward_return_days ({forward_return_days})"
            )
        
        if not self.use_expanding_window and self.train_months < 24:
            errors.append(
                f"train_months ({self.train_months}) too short for rolling window. "
                f"Recommend >= 24 months or use expanding window."
            )
        
        if self.step_months > self.test_months:
            errors.append(
                f"step_months ({self.step_months}) > test_months ({self.test_months}) "
                f"will create gaps in test coverage"
            )
        
        return errors


# ============================================
# BACKTEST CONFIGURATION
# ============================================
@dataclass
class BacktestConfig:
    """Backtesting settings."""
    
    # Capital
    initial_capital: float = 1_000_000  # $1M
    
    # Portfolio construction
    top_n_long: int = 10  # Long top N stocks
    top_n_short: int = 0  # Short bottom N (0 = long-only)
    weighting_scheme: str = "equal"  # "equal" or "prediction_weighted"
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # "monthly", "weekly", "daily"
    
    # Transaction costs (US market - realistic for large caps)
    commission_bps: float = 0.5   # 0.005%
    slippage_bps: float = 5.0     # 0.05%
    market_impact_bps: float = 2.0  # 0.02%
    
    # Benchmark
    benchmark: str = "SPY"  # S&P 500 ETF
    
    # Risk-free rate (time-varying is better, but this is fallback)
    # For proper implementation, fetch from FRED or similar
    risk_free_rate: Optional[float] = None  # None = use time-varying
    default_risk_free_rate: float = 0.03  # 3% fallback
    
    @property
    def total_cost_bps(self) -> float:
        """Total transaction cost in basis points (one-way)."""
        return self.commission_bps + self.slippage_bps + self.market_impact_bps
    
    @property
    def total_cost_pct(self) -> float:
        """Total transaction cost as decimal percentage."""
        return self.total_cost_bps / 10000
    
    @property
    def round_trip_cost_bps(self) -> float:
        """Round-trip transaction cost in basis points."""
        return self.total_cost_bps * 2
    
    def get_risk_free_rate(self, date: Optional[str] = None) -> float:
        """
        Get risk-free rate, potentially time-varying.
        
        Args:
            date: Date string (YYYY-MM-DD) for time-varying rate
            
        Returns:
            Risk-free rate as decimal (e.g., 0.04 for 4%)
            
        Note:
            For production, implement actual rate fetching from FRED.
            This is a simplified approximation.
        """
        if self.risk_free_rate is not None:
            return self.risk_free_rate
        
        if date is None:
            return self.default_risk_free_rate
        
        # Simplified time-varying approximation
        # In production, fetch from FRED (DGS3MO or similar)
        year = int(date[:4])
        
        approximate_rates = {
            2020: 0.005,  # Near zero (COVID)
            2021: 0.005,  # Near zero
            2022: 0.02,   # Rising rates
            2023: 0.05,   # High rates
            2024: 0.05,   # High rates
            2025: 0.045,  # Projected
        }
        
        return approximate_rates.get(year, self.default_risk_free_rate)


# ============================================
# RISK CONFIGURATION
# ============================================
@dataclass
class RiskConfig:
    """Risk management settings."""
    
    # Position limits
    max_position_pct: float = 0.15  # Max 15% per stock
    min_position_pct: float = 0.05  # Min 5% to maintain position
    
    # Sector limits
    use_sector_limits: bool = False
    max_sector_pct: float = 0.35  # Max 35% per sector
    
    # Stop losses
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.15  # -15% trailing stop
    
    # Volatility targeting
    use_vol_targeting: bool = False
    volatility_target: float = 0.15  # Target 15% annual vol
    vol_lookback_days: int = 63  # 3 months for vol estimation
    
    # Drawdown control
    max_drawdown_pct: float = 0.20  # Stop trading at -20% DD
    drawdown_recovery_pct: float = 0.10  # Resume at -10% DD
    
    def validate(self, top_n_long: int) -> List[str]:
        """
        Validate risk config against portfolio settings.
        
        Args:
            top_n_long: Number of long positions from BacktestConfig
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if top_n_long > 0:
            min_weight = 1.0 / top_n_long
            if self.max_position_pct < min_weight:
                errors.append(
                    f"max_position_pct ({self.max_position_pct:.2%}) < "
                    f"min required for {top_n_long} positions ({min_weight:.2%})"
                )
            
            if self.min_position_pct > min_weight:
                errors.append(
                    f"min_position_pct ({self.min_position_pct:.2%}) > "
                    f"equal weight ({min_weight:.2%}) - positions will always trigger rebalance"
                )
        
        if self.max_drawdown_pct <= self.drawdown_recovery_pct:
            errors.append(
                f"max_drawdown_pct ({self.max_drawdown_pct:.2%}) should be > "
                f"drawdown_recovery_pct ({self.drawdown_recovery_pct:.2%})"
            )
        
        return errors


# ============================================
# INTERPRETABILITY CONFIGURATION
# ============================================
@dataclass
class InterpretabilityConfig:
    """SHAP and model interpretability settings."""
    
    # SHAP analysis
    use_shap: bool = True
    shap_sample_size: int = 500  # Sample size for SHAP (full data is slow)
    shap_max_display: int = 20   # Max features in SHAP plots
    
    # Feature importance
    plot_top_n_features: int = 20
    
    # Partial dependence plots
    use_pdp: bool = False
    pdp_features: tuple = ("mom_126", "rsi_14", "volatility_21")
    pdp_grid_resolution: int = 50
    
    # Feature interaction analysis
    analyze_interactions: bool = False
    max_interaction_features: int = 5


# ============================================
# MASTER SETTINGS
# ============================================
@dataclass
class Settings:
    """
    Master configuration - combines all sub-settings.
    
    Usage:
        # Default settings
        settings = Settings()
        
        # Custom settings
        custom_data = DataConfig(start_date="2019-01-01")
        settings = Settings(data=custom_data)
    """
    
    # Sub-configurations (None = use defaults)
    log: Optional[LogConfig] = None
    data: Optional[DataConfig] = None
    features: Optional[FeatureConfig] = None
    model: Optional[ModelConfig] = None
    validation: Optional[ValidationConfig] = None
    backtest: Optional[BacktestConfig] = None
    risk: Optional[RiskConfig] = None
    interpretability: Optional[InterpretabilityConfig] = None
    
    # Output directories
    results_dir: Path = field(default_factory=lambda: ROOT / "results")
    plots_dir: Path = field(default_factory=lambda: ROOT / "results" / "plots")
    models_dir: Path = field(default_factory=lambda: ROOT / "results" / "models")
    reports_dir: Path = field(default_factory=lambda: ROOT / "results" / "reports")
    
    # Execution flags
    verbose: bool = True
    save_predictions: bool = True
    save_models: bool = True
    show_survivorship_warning: bool = True
    
    def __post_init__(self):
        """Initialize sub-configs only if not provided."""
        if self.log is None:
            self.log = LogConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.interpretability is None:
            self.interpretability = InterpretabilityConfig()
        
        # Show survivorship warning once
        if self.show_survivorship_warning:
            self._show_survivorship_warning()
    
    def _show_survivorship_warning(self) -> None:
        """Display survivorship bias warning."""
        warnings.warn(
            "\n" + SURVIVORSHIP_BIAS_WARNING,
            UserWarning,
            stacklevel=3
        )
    
    def create_dirs(self) -> None:
        """Create all output directories."""
        dirs = [
            self.results_dir, 
            self.plots_dir, 
            self.models_dir, 
            self.reports_dir,
            self.log.log_dir, 
            self.data.data_dir,
            self.data.processed_dir, 
            self.data.raw_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup(self) -> logging.Logger:
        """
        Full setup: create directories and configure logging.
        
        Returns:
            Logger: Configured logger instance
        """
        self.create_dirs()
        return self.log.setup_logging()
    
    def print_config(self) -> None:
        """Print current configuration summary."""
        print("\n" + "=" * 70)
        print("⚙️  ML ALPHA MODEL - CONFIGURATION")
        print("=" * 70)
        
        print(f"\n📊 DATA:")
        print(f"   Universe:         {self.data.universe}")
        print(f"   Date Range:       {self.data.start_date} → {self.data.effective_end_date}")
        print(f"   Data Source:      {self.data.data_source}")
        print(f"   Panel File:       {self.data.panel_path}")
        
        print(f"\n🔧 FEATURES:")
        print(f"   Forward Days:     {self.features.forward_return_days}")
        print(f"   Momentum Windows: {self.features.momentum_windows}")
        print(f"   Max Features:     {self.features.max_features}")
        print(f"   Cross-Sectional:  {'Normalized' if self.features.normalize_cross_section else 'Raw'}")
        
        print(f"\n🤖 MODEL:")
        print(f"   Type:             {self.model.model_type}")
        print(f"   N Estimators:     {self.model.lgb_params.get('n_estimators', 'N/A')}")
        print(f"   Learning Rate:    {self.model.lgb_params.get('learning_rate', 'N/A')}")
        print(f"   Random Seed:      {self.model.random_seed}")
        
        print(f"\n📈 VALIDATION:")
        window_type = "Expanding" if self.validation.use_expanding_window else "Rolling"
        print(f"   Window Type:      {window_type}")
        if self.validation.use_expanding_window:
            print(f"   Min Train:        {self.validation.min_train_months} months")
        else:
            print(f"   Train Window:     {self.validation.train_months} months")
        print(f"   Test Window:      {self.validation.test_months} months")
        print(f"   Step Size:        {self.validation.step_months} months")
        print(f"   Embargo Days:     {self.validation.embargo_days}")
        
        print(f"\n💼 BACKTEST:")
        print(f"   Initial Capital:  ${self.backtest.initial_capital:,.0f}")
        print(f"   Top N Long:       {self.backtest.top_n_long}")
        print(f"   Top N Short:      {self.backtest.top_n_short}")
        print(f"   Rebalance:        {self.backtest.rebalance_frequency}")
        print(f"   Total Cost:       {self.backtest.total_cost_bps:.2f} bps (one-way)")
        print(f"   Benchmark:        {self.backtest.benchmark}")
        
        print(f"\n🛡️  RISK:")
        print(f"   Max Position:     {self.risk.max_position_pct * 100:.0f}%")
        print(f"   Max Drawdown:     {self.risk.max_drawdown_pct * 100:.0f}%")
        print(f"   Vol Targeting:    {'Enabled' if self.risk.use_vol_targeting else 'Disabled'}")
        print(f"   Stop Loss:        {'Enabled' if self.risk.use_stop_loss else 'Disabled'}")
        
        print("=" * 70 + "\n")
    
    def validate_config(self) -> bool:
        """
        Validate configuration consistency across all settings.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        all_errors = []
        
        # Validation config checks
        val_errors = self.validation.validate(self.features.forward_return_days)
        all_errors.extend(val_errors)
        
        # Risk config checks
        risk_errors = self.risk.validate(self.backtest.top_n_long)
        all_errors.extend(risk_errors)
        
        # Cross-config checks
        if self.data.start_date >= self.data.effective_end_date:
            all_errors.append(
                f"start_date ({self.data.start_date}) must be before "
                f"end_date ({self.data.effective_end_date})"
            )
        
        # Check if data directory exists (warning only)
        if not self.data.data_dir.exists():
            warnings.warn(f"Data directory not found: {self.data.data_dir}")
        
        # Print results
        if all_errors:
            print("\n❌ Configuration Validation Errors:")
            for i, error in enumerate(all_errors, 1):
                print(f"   {i}. {error}")
            print()
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    def to_dict(self) -> Dict:
        """
        Convert settings to dictionary for serialization.
        
        Returns:
            Dict: All settings as nested dictionary
        """
        return {
            "data": {
                "universe": self.data.universe,
                "start_date": self.data.start_date,
                "end_date": self.data.end_date,
                "data_source": self.data.data_source,
            },
            "features": {
                "momentum_windows": self.features.momentum_windows,
                "forward_return_days": self.features.forward_return_days,
                "max_features": self.features.max_features,
            },
            "model": {
                "model_type": self.model.model_type,
                "random_seed": self.model.random_seed,
                "lgb_params": self.model.lgb_params,
            },
            "validation": {
                "use_expanding_window": self.validation.use_expanding_window,
                "train_months": self.validation.train_months,
                "test_months": self.validation.test_months,
                "embargo_days": self.validation.embargo_days,
            },
            "backtest": {
                "initial_capital": self.backtest.initial_capital,
                "top_n_long": self.backtest.top_n_long,
                "total_cost_bps": self.backtest.total_cost_bps,
            },
        }


# ============================================
# GLOBAL SETTINGS INSTANCE
# ============================================
# Create with warning suppressed for import; re-enable for actual use
settings = Settings(show_survivorship_warning=False)


# ============================================
# STOCK UNIVERSE
# ============================================
# ⚠️ SURVIVORSHIP BIAS WARNING:
# This list is based on 2024 market caps. For research validity,
# use point-in-time constituent data. See SURVIVORSHIP_BIAS_WARNING.
# This static list is ONLY for code testing/demo purposes.

STOCKS_SP500_TOP50 = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AVGO', 'ORCL', 'CRM',
    
    # Finance
    'BRK-B', 'JPM', 'V', 'MA', 'BAC',
    'GS', 'MS', 'BLK', 'SPGI', 'AXP',
    
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'MRK',
    'ABBV', 'TMO', 'ABT', 'DHR', 'BMY',
    
    # Consumer
    'PG', 'KO', 'PEP', 'COST', 'WMT',
    'MCD', 'NKE', 'SBUX', 'TGT', 'HD',
    
    # Industrial & Energy
    'XOM', 'CVX', 'CAT', 'HON', 'UPS',
    'BA', 'GE', 'RTX', 'LMT', 'DE',
]

# Alias for compatibility
STOCKS_SP500_TOP100 = STOCKS_SP500_TOP50  # Extend when needed


def get_universe(name: str = "sp500_top50") -> List[str]:
    """
    Get stock universe by name.
    
    Args:
        name: Universe identifier ("sp500_top50", "sp500", "sp500_top100")
        
    Returns:
        List of ticker symbols
        
    Raises:
        ValueError: If universe name is unknown
        
    Example:
        >>> tickers = get_universe("sp500_top50")
        >>> print(len(tickers))
        50
    """
    universes = {
        "sp500_top50": STOCKS_SP500_TOP50,
        "sp500": STOCKS_SP500_TOP50,  # Alias
        "sp500_top100": STOCKS_SP500_TOP100,
    }
    
    if name not in universes:
        available = ", ".join(universes.keys())
        raise ValueError(f"Unknown universe: '{name}'. Available: {available}")
    
    return universes[name].copy()  # Return copy to prevent modification


def get_feature_names() -> List[str]:
    """
    Generate expected feature names based on current config.
    
    Returns:
        List of feature name strings
        
    Note:
        Feature names use clear prefixes to avoid confusion:
        - mom_ : momentum
        - volatility_ : price volatility  
        - rsi_ : relative strength index
        - dist_ma_ : distance from moving average
        - volume_zscore_ : volume z-score
    """
    features = []
    cfg = settings.features
    
    # Momentum features
    for window in cfg.momentum_windows:
        features.append(f"mom_{window}")
        features.append(f"mom_{window}_rank")
    
    # Volatility features (renamed from vol_ to avoid confusion)
    for window in cfg.volatility_windows:
        features.append(f"volatility_{window}")
        features.append(f"volatility_{window}_rank")
    
    # Mean reversion - RSI
    for window in cfg.rsi_windows:
        features.append(f"rsi_{window}")
    
    # Mean reversion - Distance from MA
    for window in cfg.ma_windows:
        features.append(f"dist_ma_{window}")
    
    # Volume features (clear naming)
    for window in cfg.volume_windows:
        features.append(f"volume_zscore_{window}")
    
    return features


# ============================================
# UTILITY FUNCTIONS
# ============================================
def print_welcome() -> None:
    """Print welcome banner with project info."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         ML-BASED MULTI-FACTOR ALPHA MODEL                   ║
    ║         S&P 500 Quantitative Research                       ║
    ║                                                              ║
    ║   ⚠️  Demo/Research Only - See Survivorship Bias Warning    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


# ============================================
# MODULE TEST
# ============================================
if __name__ == "__main__":
    # Test configuration
    print_welcome()
    
    # Create settings with warning
    test_settings = Settings(show_survivorship_warning=True)
    test_settings.print_config()
    test_settings.validate_config()
    
    print(f"\n📋 Stock Universe ({len(get_universe())} stocks):")
    print(f"   {', '.join(get_universe()[:10])}...")
    
    print(f"\n📊 Expected Features ({len(get_feature_names())} total):")
    features = get_feature_names()
    print(f"   {', '.join(features[:8])}...")
    
    # Test custom config
    print("\n🔧 Testing Custom Config:")
    custom_data = DataConfig(start_date="2019-01-01", universe="sp500")
    custom_settings = Settings(data=custom_data, show_survivorship_warning=False)
    print(f"   Custom start_date: {custom_settings.data.start_date}")
    print(f"   Custom universe: {custom_settings.data.universe}")
    
    # Test time-varying risk-free rate
    print("\n📈 Time-Varying Risk-Free Rates:")
    for year in [2020, 2021, 2022, 2023, 2024]:
        rate = test_settings.backtest.get_risk_free_rate(f"{year}-06-15")
        print(f"   {year}: {rate:.2%}")
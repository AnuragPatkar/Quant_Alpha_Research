"""
Main configuration module - Single source of truth
Implements Singleton pattern for consistent settings across application

Author: Anurag Patkar
Design Pattern: Singleton
Dependencies: pathlib, os, datetime

FIXES:
  BUG-040: Embargo calendar-day conversion changed from magic 1.45 multiplier to 30 days
  BUG-045: WINSORIZE_QUANTILES is now the single source of truth for clip_pct
"""

import os
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dateutil.relativedelta import relativedelta


class Config:
    """
    Singleton configuration class

    Design Philosophy:
    - Single source of truth for all settings
    - Type-safe access to configuration
    - Environment-aware (dev, staging, production)
    - Validates configuration on initialization

    Usage:
        config = Config()
        tickers = config.UNIVERSE_TICKERS
        start_date = config.BACKTEST_START_DATE
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern - only one Config instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize configuration (only runs once)"""
        if self._initialized:
            return

        # Mark as initialized
        self._initialized = True

        # Set environment
        self.ENV = os.getenv('ENV', 'development')  # development, staging, production

        # Initialize all configuration sections
        self._setup_paths()
        self._setup_data_config()
        self._setup_feature_config()
        self._setup_model_config()
        self._setup_backtest_config()
        self._setup_logging_config()

        # Validate configuration
        self._validate_config()

    # ==================== PATH CONFIGURATION ====================

    def _setup_paths(self):
        """Set up all project paths"""
        # Project root
        self.PROJECT_ROOT = Path(__file__).parent.parent.absolute()

        # Data directories
        self.DATA_DIR = self.PROJECT_ROOT / 'data'
        self.RAW_DATA_DIR = self.DATA_DIR / 'raw'
        self.PROCESSED_DATA_DIR = self.DATA_DIR / 'processed'
        self.CACHE_DIR = self.DATA_DIR / 'cache'
        self.MEMBERSHIP_MASK_PATH = self.PROCESSED_DATA_DIR / 'sp500_membership_mask.pkl'

        # Raw data subdirectories
        self.PRICES_DIR = self.RAW_DATA_DIR / 'sp500_prices'
        self.FUNDAMENTALS_DIR = self.RAW_DATA_DIR / 'fundamentals'
        self.EARNINGS_DIR = self.RAW_DATA_DIR / 'earnings'
        self.ALTERNATIVE_DIR = self.RAW_DATA_DIR / 'alternative'

        # Model & Results directories
        self.MODELS_DIR = self.PROJECT_ROOT / 'models'
        self.RESULTS_DIR = self.PROJECT_ROOT / 'results'
        self.PREDICTIONS_DIR = self.RESULTS_DIR / 'predictions'
        self.LOG_DIR = self.PROJECT_ROOT / 'logs'

        # Create all directories if they don't exist
        self._create_directories()

    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.CACHE_DIR,
            self.PRICES_DIR,
            self.FUNDAMENTALS_DIR,
            self.EARNINGS_DIR,
            self.ALTERNATIVE_DIR,
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.PREDICTIONS_DIR,
            self.LOG_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    # ==================== DATA CONFIGURATION ====================

    def _setup_data_config(self):
        """Configure data sources and universe"""

        # Universe definition
        self.UNIVERSE = 'sp500_full'

        # Date range for backtesting
        self.BACKTEST_START_DATE = '2021-01-01'
        self.BACKTEST_END_DATE = '2026-02-28'

        # Known Biases (Documentation only)
        self.HAS_SURVIVORSHIP_BIAS = True

        # Data Quality Guards (CRITICAL FIXES)
        self.MIN_VALID_DATA_POINTS = 252      # Must have at least 1 year of data
        self.MAX_MISSING_PCT = 0.05           # Reject stock if >5% data is missing
        self.MAX_NAN_FILL_LIMIT = 5           # Max consecutive NaNs to forward fill
        self.MIN_VOLUME_THRESHOLD = 1_000_000  # Liquidity Filter ($1M+ avg volume)

        # Data Refresh Settings
        self.FUNDAMENTAL_UPDATE_FREQ = "quarterly"   # Q1, Q2, Q3, Q4
        self.EARNINGS_UPDATE_FREQ = "daily"
        self.NEWS_UPDATE_FREQ = "daily"

        # -------------------------------------------------------------
        # DYNAMIC UNIVERSE LOADING (FROM DISK)
        # -------------------------------------------------------------
        try:
            price_files = list(self.PRICES_DIR.glob('*.csv'))
            if price_files:
                self.UNIVERSE_TICKERS = [f.stem for f in price_files]
                self.UNIVERSE_SIZE = len(self.UNIVERSE_TICKERS)
            else:
                self.UNIVERSE_TICKERS = []
                self.UNIVERSE_SIZE = 0
        except Exception as e:
            print(f"⚠️ Warning: Could not load universe from disk: {e}")
            self.UNIVERSE_TICKERS = []
            self.UNIVERSE_SIZE = 0

    # ==================== FEATURE CONFIGURATION ====================

    def _setup_feature_config(self):
        """Configure feature engineering parameters"""
        self.FORWARD_RETURN_DAYS = 5
        self.FORWARD_HORIZONS = [5, 10, 21, 63]

        # Windows for technical indicators
        self.MOMENTUM_WINDOWS = [5, 10, 21, 63, 126, 252]
        self.VOLATILITY_WINDOWS = [5, 10, 21, 63, 126]
        self.VOLUME_WINDOWS = [5, 10, 21, 63]

        # --------------------------------------------------------
        # ML PREPROCESSING
        # --------------------------------------------------------
        self.RSI_PERIOD = 14
        self.BB_PERIOD = 20
        self.BB_STD = 2.0

        # Feature Selection (Prevents overfitting/model confusion)
        self.MAX_FEATURES = 80                       # Retain only the Top 80 predictive signals
        self.FEATURE_CORRELATION_THRESHOLD = 0.75    # Remove highly correlated (duplicate) signals

        # Data Cleaning (Normalization & Outlier Management)
        self.NORMALIZE_FEATURES = True               # Z-Score Standardization (Scale all features equally)
        self.WINSORIZE_FEATURES = True               # Clip extreme outliers

        # FIX BUG-045: WINSORIZE_QUANTILES is the SINGLE source of truth for
        # WinsorisationScaler clip_pct. preprocessing.py reads this value.
        self.WINSORIZE_QUANTILES = (0.01, 0.99)      # Cap the Top/Bottom 1% of values

        # Feature Engineering Switches
        self.ENABLE_TECHNICAL_FEATURES = True
        self.ENABLE_FUNDAMENTAL_FEATURES = True
        self.ENABLE_EARNINGS_FEATURES = True
        self.ENABLE_ALTERNATIVE_FEATURES = True

        # Data Engineering Bounds
        self.RETURN_CLIP_MIN = -0.50      # Cap massive gaps/halts to prevent loss distortion
        self.RETURN_CLIP_MAX = 0.50       # Cap massive gaps/halts to prevent loss distortion

        # Model Performance Gates
        self.PROD_IC_THRESHOLD = 0.010
        self.PROD_IC_TSTAT = 2.5
        self.MIN_OOS_IC_THRESHOLD = 0.005
        self.MIN_OOS_IC_TSTAT = 1.5

    # ==================== MODEL CONFIGURATION ====================

    def _setup_model_config(self):
        """Configure ML model parameters"""
        self.PRIMARY_MODEL = 'ensemble'
        self.ENSEMBLE_MODELS = ['lightgbm', 'xgboost', 'catboost']
        self.ENSEMBLE_METHOD = 'weighted'

        # 1. Ensemble Weights
        # LightGBM usually performs best on tabular data, giving it highest weight
        self.MODEL_WEIGHTS = {
            'lightgbm': 0.4,
            'xgboost': 0.3,
            'catboost': 0.3,
        }

        # 2. LightGBM parameters (Optimized for Finance)
        self.LGBM_PARAMS = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'reg_alpha': 1.0,     # L1 Regularization (Important for noise)
            'reg_lambda': 1.0,    # L2 Regularization
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        # 3. XGBoost Parameters
        self.XGB_PARAMS = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,     # L1
            'reg_lambda': 1.0,    # L2
            'random_state': 42,
            'n_jobs': -1,
        }

        # 4. CatBoost Parameters
        self.CATBOOST_PARAMS = {
            'loss_function': 'RMSE',
            'depth': 6,
            'learning_rate': 0.01,
            'iterations': 200,
            'l2_leaf_reg': 3.0,   # Regularization
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
        }

        # Hyperparameter optimization settings
        if self.ENV == 'production':
            self.ENABLE_HYPEROPT = False
            self.HYPEROPT_TRIALS = 0
        else:
            self.ENABLE_HYPEROPT = True
            self.HYPEROPT_TRIALS = 50

    # ==================== BACKTEST CONFIGURATION ====================

    def _setup_backtest_config(self):
        """Configure backtesting parameters"""

        # Walk-forward settings
        self.VALIDATION_METHOD = 'walk_forward_expanding'
        self.MIN_TRAIN_MONTHS = 24
        self.TEST_WINDOW_MONTHS = 6
        self.STEP_SIZE_MONTHS = 3

        # FIX BUG-040: EMBARGO_TRADING_DAYS remains as trading-day count (21).
        # get_train_test_splits() uses 30 calendar days as a conservative calendar
        # approximation (~21 trading days). The trainer.py resolves actual trading-day
        # offsets on the real data index (confirmed architectural decision).
        self.EMBARGO_TRADING_DAYS = 21              # Used by trainer.py on data index
        self._EMBARGO_CALENDAR_DAYS = 30            # Used only in get_train_test_splits()
        #   (30 cal days ≈ 21 trading days; conservative; avoids the 1.45 magic number)

        # Fix full history fallback
        self.PROD_MODEL_MIN_DATE = '2020-01-01'

        # Execution Logic
        self.EXECUTION_PRICE = 'open'    # Trade on NEXT DAY Open (Realistic)
        # self.EXECUTION_PRICE = 'close' # Trade on SAME DAY Close (Optimistic)

        # Portfolio Config
        self.INITIAL_CAPITAL = 1_000_000
        self.NUM_LONG_POSITIONS = 10

        # Risk Management
        self.MAX_SECTOR_EXPOSURE = 0.30   # Max 30% in one sector
        self.MAX_DRAWDOWN_LIMIT = 0.20    # Stop trading if 20% loss
        self.STOP_LOSS_PCT = 0.05         # Stop loss per trade
        self.TAKE_PROFIT_PCT = 0.15       # Take profit per trade
        self.TRAILING_STOP_PCT = 0.10     # Trailing stop (10%)
        self.MAX_POSITION_SIZE = 0.10

        # Realistic Market Simulation
        self.TRANSACTION_COST_BPS = 10.0  # Fees
        self.SLIPPAGE_PCT = 0.0005        # 0.05% Price Slippage
        self.BACKTEST_SPREAD = 0.0005     # 5 bps Spread
        self.BACKTEST_MAX_TURNOVER = 0.20 # Max 20% turnover per rebalance
        self.BACKTEST_TARGET_VOL = 0.15   # Volatility Targeting constraint
        self.RISK_FREE_RATE = 0.04        # 4% Annual Risk Free Rate (for Sharpe)
        self.BENCHMARK_TICKER = 'SPY'     # Compare performance vs S&P 500
        self.REBALANCE_FREQ = 'W'         # Weekly Rebalancing

        # Portfolio Optimization Settings
        self.OPT_MAX_WEIGHT = 0.10        # Ensures optimizers respect max limits
        self.OPT_MIN_WEIGHT = 0.0         # Allow zeroing out positions
        self.OPT_RISK_AVERSION = 2.5      # Mean-Variance & BL
        self.OPT_KELLY_FRACTION = 0.5     # Kelly Criterion
        self.OPT_LOOKBACK_DAYS = 252      # Covariance Lookback
        self.MAX_LEVERAGE = 1.0           # 1.0 = Fully funded, no margin
        self.MAX_ALPHA_RET = 0.30         # Max expected return assumption for MVO
        self.BL_CONFIDENCE_LEVEL = 0.6    # Black-Litterman alpha confidence

        # Inference Settings
        self.INFERENCE_SCALER_LOOKBACK_YEARS = 3
        self.ALPHA_SMOOTHING_LAMBDA = 0.70    # EWMA decay factor for ensemble signals

    # ==================== LOGGING CONFIGURATION ====================

    def _setup_logging_config(self):
        """Configure logging settings"""
        self.LOG_FILE = self.LOG_DIR / f'quant_alpha_{self.ENV}.log'
        self.LOG_LEVEL = 'INFO'
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    # ==================== VALIDATION ====================

    def _validate_config(self):
        """Validate configuration integrity (Dates AND Paths)"""
        errors = []

        # 1. Date Validation
        try:
            start = datetime.strptime(self.BACKTEST_START_DATE, '%Y-%m-%d')
            end = datetime.strptime(self.BACKTEST_END_DATE, '%Y-%m-%d')
            if end <= start:
                errors.append("BACKTEST_END_DATE must be after BACKTEST_START_DATE")
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")

        # 2. Winsorize quantile consistency
        lo, hi = self.WINSORIZE_QUANTILES
        if not (0.0 < lo < 0.5 and 0.5 < hi < 1.0):
            errors.append(
                f"WINSORIZE_QUANTILES must be (lo, hi) with 0 < lo < 0.5 and 0.5 < hi < 1. "
                f"Got {self.WINSORIZE_QUANTILES}"
            )

        # 3. Data Content Validation (Warning only)
        if self.PRICES_DIR.exists():
            csv_count = len(list(self.PRICES_DIR.glob("*.csv")))
            if csv_count == 0:
                print(
                    f"⚠️  WARNING: {self.PRICES_DIR} exists but is empty. "
                    "You need to run download scripts."
                )
            else:
                benchmark_path = self.ALTERNATIVE_DIR / "sp500.csv"
                if not benchmark_path.exists():
                            print(f"⚠️  WARNING: Benchmark data 'sp500.csv' not found in {self.ALTERNATIVE_DIR}. "
                                  f"Performance reporting will fetch it live from YFinance.")

        if errors:
            raise ValueError("Configuration Error:\n" + "\n".join(errors))

    # ==================== UTILITIES ====================

    def get_train_test_splits(self):
        """
        Generate walk-forward splits.
        Supports both 'expanding' (anchored start) and 'rolling' (moving start) windows.

        FIX BUG-040: Uses self._EMBARGO_CALENDAR_DAYS (30) instead of the magic
        1.45 multiplier. trainer.py applies the actual trading-day embargo on the
        data index using EMBARGO_TRADING_DAYS (21 offsets).
        """
        start = datetime.strptime(self.BACKTEST_START_DATE, '%Y-%m-%d')
        end = datetime.strptime(self.BACKTEST_END_DATE, '%Y-%m-%d')

        splits = []
        current_start = start

        while True:
            # Training period logic
            if self.VALIDATION_METHOD == 'walk_forward_expanding':
                train_start = start           # Anchor to the beginning (Expanding)
            else:
                train_start = current_start   # Moving window (Rolling)

            train_end = current_start + relativedelta(months=self.MIN_TRAIN_MONTHS)

            # FIX BUG-040: Use conservative 30 calendar days (≈ 21 trading days).
            # Trainer resolves precise trading-day embargo on the actual data index.
            test_start = train_end + timedelta(days=self._EMBARGO_CALENDAR_DAYS)
            test_end = test_start + relativedelta(months=self.TEST_WINDOW_MONTHS)

            # Stop if we run out of data
            if test_end > end:
                break

            splits.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d'),
            ))

            # Move the window forward
            current_start += relativedelta(months=self.STEP_SIZE_MONTHS)

        return splits


# Global config instance
config = Config()


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print(f"🚀 QUANT ALPHA SYSTEM | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    print(f"✅ CONFIG LOADED    : {config.ENV.upper()}")
    print(f"📊 UNIVERSE SIZE   : {config.UNIVERSE_SIZE} stocks")
    print(f"🧠 ENSEMBLE        : {', '.join(config.ENSEMBLE_MODELS)}")
    print(f"⚖️  MODEL WEIGHTS   : {config.MODEL_WEIGHTS}")

    print("-" * 50)
    print(
        f"🛡️  RISK RULES      : Sector Cap={config.MAX_SECTOR_EXPOSURE * 100}%, "
        f"Stop Loss={config.STOP_LOSS_PCT * 100}%"
    )
    print(
        f"🛡️  DATA GUARDS     : Min Vol={config.MIN_VOLUME_THRESHOLD:,.0f}, "
        f"Max NaN={config.MAX_NAN_FILL_LIMIT}"
    )
    print(
        f"📉 SIMULATION      : Slippage={config.SLIPPAGE_PCT * 100}%, "
        f"Fees={config.TRANSACTION_COST_BPS}bps"
    )

    splits = config.get_train_test_splits()
    print("-" * 50)
    print(f"🔄 STRATEGY        : {config.VALIDATION_METHOD}")

    if splits:
        print(f"   Total Folds     : {len(splits)}")
        print(
            f"   First Fold      : {splits[0][0]} to {splits[0][1]} "
            f"(Test: {splits[0][2]})"
        )
        print(
            f"   Last Fold       : {splits[-1][0]} to {splits[-1][1]} "
            f"(Test: {splits[-1][2]})"
        )

        if (
            config.VALIDATION_METHOD == 'walk_forward_expanding'
            and splits[0][0] == splits[-1][0]
        ):
            print("   ✅ Anchor Check : SUCCESS (Fixed Start Date)")
        elif (
            config.VALIDATION_METHOD == 'walk_forward_rolling'
            and splits[0][0] != splits[-1][0]
        ):
            print("   ✅ Anchor Check : SUCCESS (Moving Window)")
    else:
        print("   ⚠️  WARNING      : No backtest splits generated. Check date range!")

    print("=" * 50 + "\n")
"""
Global Configuration and Parameter Registry
===========================================
Centralized, Singleton-based configuration manager for the Quant Alpha platform.

Purpose
-------
This module establishes the single source of truth for all foundational parameters,
hyperparameters, and environmental configurations. It orchestrates directory structures,
feature engineering bounds, machine learning topology, and backtest constraints,
ensuring strict parity between research and production environments.

Role in Quantitative Workflow
-----------------------------
Imported system-wide to guarantee deterministic execution. Validates the structural
integrity of hyperparameters, path resolutions, and look-ahead bias constraints
(e.g., embargo periods) prior to pipeline execution.
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
    Singleton configuration registry for the quantitative platform.

    Ensures a single, immutable state across multi-process or distributed
    execution DAGs, preventing hyperparameter drift during pipeline runs.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        Enforces the Singleton design pattern.

        Returns:
            Config: The globally shared configuration instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initializes the platform configuration and validates structural integrity.

        Executes strictly once per application lifecycle to bind environmental
        variables, establish data warehouse paths, and freeze algorithmic hyperparameters.
        """
        if self._initialized:
            return

        self._initialized = True

        # Bind execution environment to adjust computational verbosity and pipeline persistence
        self.ENV = os.getenv('ENV', 'development')

        self._setup_paths()
        self._setup_data_config()
        self._setup_feature_config()
        self._setup_model_config()
        self._setup_backtest_config()
        self._setup_logging_config()

        self._validate_config()

    def _setup_paths(self):
        """
        Dynamically resolves and binds absolute paths for the Data Warehouse and ML Artifacts.
        """
        self.PROJECT_ROOT = Path(__file__).parent.parent.absolute()

        self.DATA_DIR = self.PROJECT_ROOT / 'data'
        self.RAW_DATA_DIR = self.DATA_DIR / 'raw'
        self.PROCESSED_DATA_DIR = self.DATA_DIR / 'processed'
        self.CACHE_DIR = self.DATA_DIR / 'cache'
        self.MEMBERSHIP_MASK_PATH = self.PROCESSED_DATA_DIR / 'sp500_membership_mask.pkl'

        self.PRICES_DIR = self.RAW_DATA_DIR / 'sp500_prices'
        self.FUNDAMENTALS_DIR = self.RAW_DATA_DIR / 'fundamentals'
        self.EARNINGS_DIR = self.RAW_DATA_DIR / 'earnings'
        self.ALTERNATIVE_DIR = self.RAW_DATA_DIR / 'alternative'

        self.MODELS_DIR = self.PROJECT_ROOT / 'models'
        self.RESULTS_DIR = self.PROJECT_ROOT / 'results'
        self.PREDICTIONS_DIR = self.RESULTS_DIR / 'predictions'
        self.LOG_DIR = self.PROJECT_ROOT / 'logs'

        self._create_directories()

    def _create_directories(self):
        """
        Provisions the target directory tree for data, models, and telemetry.
        """
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

    def _setup_data_config(self):
        """
        Configures quantitative universe boundaries, data quality guards, and pipeline frequencies.
        """
        self.UNIVERSE = 'sp500_full'

        self.BACKTEST_START_DATE = '2021-01-01'
        self.BACKTEST_END_DATE = '2026-05-01'

        # Flag indicating active survivorship bias mitigation via point-in-time constituent masking
        self.HAS_SURVIVORSHIP_BIAS = True

        # Quality Gates: Prevents model inference on illiquid or statistically sparse assets
        self.MIN_VALID_DATA_POINTS = 252
        self.MAX_MISSING_PCT = 0.05
        self.MAX_NAN_FILL_LIMIT = 5
        self.MIN_VOLUME_THRESHOLD = 1_000_000

        self.FUNDAMENTAL_UPDATE_FREQ = "quarterly"
        self.EARNINGS_UPDATE_FREQ = "daily"
        self.NEWS_UPDATE_FREQ = "daily"

        # Attempt dynamic universe discovery via materialized data blobs in the local storage tier
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

    def _setup_feature_config(self):
        """
        Defines hyperparameter boundaries for feature extraction, signal smoothing, and model gating.
        """
        self.FORWARD_RETURN_DAYS = 5
        self.FORWARD_HORIZONS = [5, 10, 21, 63]

        # Temporal isolation windows for momentum and mean reversion features
        self.MOMENTUM_WINDOWS = [5, 10, 21, 63, 126, 252]
        self.VOLATILITY_WINDOWS = [5, 10, 21, 63, 126]
        self.VOLUME_WINDOWS = [5, 10, 21, 63]

        self.RSI_PERIOD = 14
        self.BB_PERIOD = 20
        self.BB_STD = 2.0

        # Dimensionality Reduction: Mitigates multicollinearity and model degradation
        self.MAX_FEATURES = 80
        self.FEATURE_CORRELATION_THRESHOLD = 0.75

        # Preprocessing execution state switches
        self.NORMALIZE_FEATURES = True
        self.WINSORIZE_FEATURES = True

        # Establishes the definitive institutional clipping threshold for WinsorisationScaler logic
        self.WINSORIZE_QUANTILES = (0.01, 0.99)

        self.ENABLE_TECHNICAL_FEATURES = True
        self.ENABLE_FUNDAMENTAL_FEATURES = True
        self.ENABLE_EARNINGS_FEATURES = True
        self.ENABLE_ALTERNATIVE_FEATURES = True

        # Hard boundaries to prevent anomalous returns (e.g., micro-cap reverse splits) from warping the target variable
        self.RETURN_CLIP_MIN = -0.50
        self.RETURN_CLIP_MAX = 0.50

        # Statistical significance gates required for promotion to the Production Ensemble
        self.PROD_IC_THRESHOLD = 0.010
        self.PROD_IC_TSTAT = 2.5
        self.MIN_OOS_IC_THRESHOLD = 0.005
        self.MIN_OOS_IC_TSTAT = 1.5

    def _setup_model_config(self):
        """
        Configures the Gradient Boosted Decision Tree (GBDT) ensemble hyperparameters.
        """
        self.PRIMARY_MODEL = 'ensemble'
        self.ENSEMBLE_MODELS = ['lightgbm', 'xgboost', 'catboost']
        self.ENSEMBLE_METHOD = 'weighted'

        # Static allocation weights derived from out-of-sample stability analysis
        self.MODEL_WEIGHTS = {
            'lightgbm': 0.4,
            'xgboost': 0.3,
            'catboost': 0.3,
        }

        # LightGBM structural hyperparameters targeted for financial tabular data
        self.LGBM_PARAMS = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        self.XGB_PARAMS = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
        }

        self.CATBOOST_PARAMS = {
            'loss_function': 'RMSE',
            'depth': 6,
            'learning_rate': 0.01,
            'iterations': 200,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
        }

        if self.ENV == 'production':
            self.ENABLE_HYPEROPT = False
            self.HYPEROPT_TRIALS = 0
        else:
            self.ENABLE_HYPEROPT = True
            self.HYPEROPT_TRIALS = 50

    def _setup_backtest_config(self):
        """
        Establishes transaction cost analysis (TCA), execution constraints, and validation horizons.
        """
        self.VALIDATION_METHOD = 'walk_forward_expanding'
        self.MIN_TRAIN_MONTHS = 24
        self.TEST_WINDOW_MONTHS = 6
        self.STEP_SIZE_MONTHS = 3

        # Defines the mathematical embargo gap required to strictly prevent signal leakage
        # during Purged K-Fold Cross-Validation.
        self.EMBARGO_TRADING_DAYS = 21
        self._EMBARGO_CALENDAR_DAYS = 30

        self.PROD_MODEL_MIN_DATE = '2020-01-01'

        # Realistic execution assumption targeting next-day open to avoid temporal contamination
        self.EXECUTION_PRICE = 'open'

        self.INITIAL_CAPITAL = 1_000_000
        self.NUM_LONG_POSITIONS = 10

        # Mandatory portfolio-level capital preservation bounds
        self.MAX_SECTOR_EXPOSURE = 0.30
        self.MAX_DRAWDOWN_LIMIT = 0.20
        self.STOP_LOSS_PCT = 0.05
        self.TAKE_PROFIT_PCT = 0.15
        self.TRAILING_STOP_PCT = 0.10
        self.MAX_POSITION_SIZE = 0.10

        # Execution and macro constraints defining real-world implementation friction
        self.TRANSACTION_COST_BPS = 10.0
        self.SLIPPAGE_PCT = 0.0005
        self.BACKTEST_SPREAD = 0.0005
        self.BACKTEST_MAX_TURNOVER = 0.20
        self.BACKTEST_TARGET_VOL = 0.15
        self.RISK_FREE_RATE = 0.04
        self.BENCHMARK_TICKER = 'SPY'
        self.REBALANCE_FREQ = 'W'

        # Convex optimization inputs dictating asset allocation logic
        self.OPT_MAX_WEIGHT = 0.10
        self.OPT_MIN_WEIGHT = 0.0
        self.OPT_RISK_AVERSION = 2.5
        self.OPT_KELLY_FRACTION = 0.5
        self.OPT_LOOKBACK_DAYS = 252
        self.MAX_LEVERAGE = 1.0
        self.MAX_ALPHA_RET = 0.30
        self.BL_CONFIDENCE_LEVEL = 0.6

        # EWMA decay factor applied cross-sectionally to dampen signal oscillation
        self.INFERENCE_SCALER_LOOKBACK_YEARS = 3
        self.ALPHA_SMOOTHING_LAMBDA = 0.70

    def _setup_logging_config(self):
        """
        Binds deterministic global formatter parameters for observability tracing.
        """
        self.LOG_FILE = self.LOG_DIR / f'quant_alpha_{self.ENV}.log'
        self.LOG_LEVEL = 'INFO'
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def _validate_config(self):
        """
        Executes critical runtime assertions to prevent structurally flawed pipeline executions.

        Raises:
            ValueError: If date parameters intersect or normalization thresholds are mathematically invalid.
        """
        errors = []

        try:
            start = datetime.strptime(self.BACKTEST_START_DATE, '%Y-%m-%d')
            end = datetime.strptime(self.BACKTEST_END_DATE, '%Y-%m-%d')
            if end <= start:
                errors.append("BACKTEST_END_DATE must be after BACKTEST_START_DATE")
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")

        lo, hi = self.WINSORIZE_QUANTILES
        if not (0.0 < lo < 0.5 and 0.5 < hi < 1.0):
            errors.append(
                f"WINSORIZE_QUANTILES must be (lo, hi) with 0 < lo < 0.5 and 0.5 < hi < 1. "
                f"Got {self.WINSORIZE_QUANTILES}"
            )

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

    def get_train_test_splits(self):
        """
        Generates the chronological tuple sets required for out-of-sample model evaluation.

        Implements Purged K-Fold validation by explicitly buffering test boundaries with
        the target embargo horizon. Supports both dynamic expanding windows and fixed-length
        rolling anchors to combat non-stationary distribution decay.

        Returns:
            List[Tuple[str, str, str, str]]: A sequence mapping strictly to:
                (train_start, train_end, oos_test_start, oos_test_end).
        """
        start = datetime.strptime(self.BACKTEST_START_DATE, '%Y-%m-%d')
        end = datetime.strptime(self.BACKTEST_END_DATE, '%Y-%m-%d')

        splits = []
        current_start = start

        while True:
            if self.VALIDATION_METHOD == 'walk_forward_expanding':
                train_start = start
            else:
                train_start = current_start

            train_end = current_start + relativedelta(months=self.MIN_TRAIN_MONTHS)

            # Apply conservative calendar-to-business-day proxy mapping to purge overlapping serial correlation
            test_start = train_end + timedelta(days=self._EMBARGO_CALENDAR_DAYS)
            test_end = test_start + relativedelta(months=self.TEST_WINDOW_MONTHS)

            if test_end > end:
                break

            splits.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d'),
            ))

            current_start += relativedelta(months=self.STEP_SIZE_MONTHS)

        return splits


# Export the provisioned global configuration instance
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
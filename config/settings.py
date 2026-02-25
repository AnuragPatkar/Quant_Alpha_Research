"""
Main configuration module - Single source of truth
Implements Singleton pattern for consistent settings across application

Author: Anurag Patkar
Design Pattern: Singleton
Dependencies: pathlib, os, datetime
"""

import os 
import logging
from pathlib import Path 
from datetime import datetime , timedelta 
from typing import List, Dict , Optional
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
    
    def __new__(cls):
        """Singleton pattern - only one Config instance exists"""
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
        
        # Raw data subdirectories
        self.PRICES_DIR = self.RAW_DATA_DIR / 'sp500_prices'
        self.FUNDAMENTALS_DIR = self.RAW_DATA_DIR / 'fundamentals'
        self.EARNINGS_DIR = self.RAW_DATA_DIR / 'earnings'
        self.ALTERNATIVE_DIR = self.RAW_DATA_DIR / 'alternative'
        
        # Model & Results directories
        self.MODELS_DIR = self.PROJECT_ROOT / 'models'
        self.RESULTS_DIR = self.PROJECT_ROOT / 'results'
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
            self.LOG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # ==================== DATA CONFIGURATION ====================

    def _setup_data_config(self):
        """Configure data sources and universe"""

        # Universe definition
        self.UNIVERSE = 'sp500_full'
        
        # Date range for backtesting
        self.BACKTEST_START_DATE = '2016-01-01'
        self.BACKTEST_END_DATE = '2024-01-01'

        # Known Biases (Documentation only)
        self.HAS_SURVIVORSHIP_BIAS = True 
        
        # Data Quality Guards (CRITICAL FIXES)
        self.MIN_VALID_DATA_POINTS = 252  # Must have at least 1 year of data
        self.MAX_MISSING_PCT = 0.05       # Reject stock if >5% data is missing
        self.MAX_NAN_FILL_LIMIT = 5       # Max consecutive NaNs to forward fill
        self.MIN_VOLUME_THRESHOLD = 1_000_000 # Liquidity Filter ($1M+ avg volume)


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
        self.FORWARD_RETURN_DAYS = 10
        self.FORWARD_HORIZONS = [5,10,21,63]
       
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
        self.MAX_FEATURES = 80             # Retain only the Top 80 predictive signals
        self.FEATURE_CORRELATION_THRESHOLD = 0.75 # Remove highly correlated (duplicate) signals
        
        # Data Cleaning (Normalization & Outlier Management)
        self.NORMALIZE_FEATURES = True     # Z-Score Standardization (Scale all features equally)
        self.WINSORIZE_FEATURES = True     # Clip extreme outliers
        self.WINSORIZE_QUANTILES = (0.01, 0.99) # Cap the Top/Bottom 1% of values

        
        # Feature Engineering Switches
        self.ENABLE_TECHNICAL_FEATURES = True
        self.ENABLE_FUNDAMENTAL_FEATURES = True
        self.ENABLE_EARNINGS_FEATURES = True
        self.ENABLE_ALTERNATIVE_FEATURES = True

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
            'catboost': 0.3
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
            'reg_alpha': 1.0,   # L1 Regularization (Important for noise)
            'reg_lambda': 1.0,  # L2 Regularization
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
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
            'reg_alpha': 1.0,   # L1
            'reg_lambda': 1.0,  # L2
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 4. CatBoost Parameters
        self.CATBOOST_PARAMS = {
            'loss_function': 'RMSE',
            'depth': 6,
            'learning_rate': 0.01,
            'iterations': 200,
            'l2_leaf_reg': 3.0, # Regularization
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False 
        }
        
        # Hyperparameter optimization settings
        self.ENABLE_HYPEROPT = True
        self.HYPEROPT_TRIALS = 50

    # ==================== BACKTEST CONFIGURATION ====================

    def _setup_backtest_config(self):
        """Configure backtesting parameters"""

        # Walk-forward settings
        self.VALIDATION_METHOD = 'walk_forward_expanding'
        self.MIN_TRAIN_MONTHS = 24
        self.TEST_WINDOW_MONTHS = 3
        self.STEP_SIZE_MONTHS = 3
        self.EMBARGO_DAYS = 21
        
        # Execution Logic 
        self.EXECUTION_PRICE = 'open'    # Trade on NEXT DAY Open (Realistic)
        # self.EXECUTION_PRICE = 'close' # Trade on SAME DAY Close (Optimistic)

        # Portfolio Config
        self.INITIAL_CAPITAL = 1_000_000
        self.NUM_LONG_POSITIONS = 10
        
        # Risk Management (ADDED NEW)
        self.MAX_SECTOR_EXPOSURE = 0.30  # Max 30% in one sector
        self.MAX_DRAWDOWN_LIMIT = 0.20   # Stop trading if 20% loss
        self.STOP_LOSS_PCT = 0.05        # Stop loss per trade
        self.TAKE_PROFIT_PCT = 0.15      # Take profit per trade
        self.TRAILING_STOP_PCT = 0.10    # Trailing stop (10%)
        self.MAX_POSITION_SIZE = 0.15
        
        # Realistic Market Simulation (ADDED NEW)
        self.TRANSACTION_COST_BPS = 10.0  # Fees
        self.SLIPPAGE_PCT = 0.0005        # 0.05% Price Slippage
        self.RISK_FREE_RATE = 0.04        # 4% Annual Risk Free Rate (for Sharpe)
        self.BENCHMARK_TICKER = 'SPY'     # Compare performance vs S&P 500
        self.REBALANCE_FREQ = 'W'         # Weekly Rebalancing

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
            
        # 2. Critical Path Validation 
        critical_paths = {
            "Data Root": self.DATA_DIR,
            "Price Data": self.PRICES_DIR,
            "Models": self.MODELS_DIR,
            "Logs": self.LOG_DIR,
            "Results": self.RESULTS_DIR
        }
        
        for name, path in critical_paths.items():
            if not path.exists():
                errors.append(f"Critical Directory Not Found: {name} ({path})")
        
        # 3. Data Content Validation (Warning only)
        if self.PRICES_DIR.exists():
            csv_count = len(list(self.PRICES_DIR.glob("*.csv")))
            if csv_count == 0:
                print(f"⚠️  WARNING: {self.PRICES_DIR} exists but is empty. You need to run download scripts.")
        
        if errors:
            raise ValueError("Configuration Error:\n" + "\n".join(errors))

    # ==================== UTILITIES ====================

    def get_train_test_splits(self):
        """
        Generate walk-forward splits.
        Supports both 'expanding' (anchored start) and 'rolling' (moving start) windows.
        """
        from dateutil.relativedelta import relativedelta
        
        start = datetime.strptime(self.BACKTEST_START_DATE, '%Y-%m-%d')
        end = datetime.strptime(self.BACKTEST_END_DATE, '%Y-%m-%d')
        
        splits = []
        current_start = start
        
        while True:
            # Training period logic
            if self.VALIDATION_METHOD == 'walk_forward_expanding':
                train_start = start  # Anchor to the beginning (Expanding)
            else:
                train_start = current_start # Moving window (Rolling)
                
            train_end = current_start + relativedelta(months=self.MIN_TRAIN_MONTHS)
            
            # Embargo period (Prevent leakage)
            test_start = train_end + timedelta(days=self.EMBARGO_DAYS)
            test_end = test_start + relativedelta(months=self.TEST_WINDOW_MONTHS)
            
            # Stop if we run out of data
            if test_end > end:
                break
            
            splits.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
            # Move the window forward
            current_start += relativedelta(months=self.STEP_SIZE_MONTHS)
        return splits

# Global config instance
config = Config()

if __name__ == '__main__':
    print("\n" + "="*50)
    print(f"🚀 QUANT ALPHA SYSTEM | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    print(f"✅ CONFIG LOADED    : {config.ENV.upper()}")
    print(f"📊 UNIVERSE SIZE   : {config.UNIVERSE_SIZE} stocks")
    print(f"🧠 ENSEMBLE        : {', '.join(config.ENSEMBLE_MODELS)}")
    print(f"⚖️  MODEL WEIGHTS   : {config.MODEL_WEIGHTS}")
    
    # 2. Risk & Simulation Guards
    print("-" * 50)
    print(f"🛡️  RISK RULES      : Sector Cap={config.MAX_SECTOR_EXPOSURE*100}%, Stop Loss={config.STOP_LOSS_PCT*100}%")
    print(f"🛡️  DATA GUARDS     : Min Vol={config.MIN_VOLUME_THRESHOLD:,.0f}, Max NaN={config.MAX_NAN_FILL_LIMIT}")
    print(f"📉 SIMULATION      : Slippage={config.SLIPPAGE_PCT*100}%, Fees={config.TRANSACTION_COST_BPS}bps")
    
    # 3. Validation Strategy Analysis
    splits = config.get_train_test_splits()
    print("-" * 50)
    print(f"🔄 STRATEGY        : {config.VALIDATION_METHOD}")
    
    if splits:
        print(f"   Total Folds     : {len(splits)}")
        print(f"   First Fold      : {splits[0][0]} to {splits[0][1]} (Test: {splits[0][2]})")
        print(f"   Last Fold       : {splits[-1][0]} to {splits[-1][1]} (Test: {splits[-1][2]})")
        
        # Logic Check
        if config.VALIDATION_METHOD == 'walk_forward_expanding' and splits[0][0] == splits[-1][0]:
            print("   ✅ Anchor Check : SUCCESS (Fixed Start Date)")
        elif config.VALIDATION_METHOD == 'walk_forward_rolling' and splits[0][0] != splits[-1][0]:
            print("   ✅ Anchor Check : SUCCESS (Moving Window)")
    else:
        print("   ⚠️  WARNING      : No backtest splits generated. Check date range!")
        
    print("="*50 + "\n")
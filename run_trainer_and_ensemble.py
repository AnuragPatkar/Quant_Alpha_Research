import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import logging
import warnings
import joblib

# --- CONFIGURATION & SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from quant_alpha.data.DataManager import DataManager
from quant_alpha.models.trainer import WalkForwardTrainer
from quant_alpha.models.lightgbm_model import LightGBMModel
from quant_alpha.models.xgboost_model import XGBoostModel
from quant_alpha.models.catboost_model import CatBoostModel
from quant_alpha.models.feature_selector import FeatureSelector
from quant_alpha.features.utils import rank_transform, winsorize, cross_sectional_normalize
from quant_alpha.backtest.engine import BacktestEngine
from quant_alpha.backtest.metrics import print_metrics_report
from quant_alpha.backtest.attribution import SimpleAttribution, FactorAttribution
from config.settings import config

# --- RISK & PORTFOLIO PARAMETERS ---
TOP_N_STOCKS = 25
STOCK_STOP_LOSS = -0.05       # 20% se 5% kar diya
SL_SLIPPAGE_PENALTY = -0.005  # Tight SL par slippage kam hota hai
TRANSACTION_COST = 0.001      # 10 bps per trade
TURNOVER_THRESHOLD = 0.15     # Naya stock tabhi lenge agar wo top 15% rank improvement dikhaye
PORTFOLIO_DD_EXIT = -0.15     # Exit strategy when DD exceeds 15%
PORTFOLIO_DD_REENTRY = -0.05  # Re-enter when DD recovers to 5%


def load_and_build_full_dataset():
    # ---------------------------------------------------------
    # FIX: Check for Cached Master File (Speed up)
    # ---------------------------------------------------------
    cache_path = r"E:\coding\quant_alpha_research\data\cache\master_data_with_factors.parquet"
    if os.path.exists(cache_path):
        logger.info(f"⚡ Loading Cached Master Dataset from {cache_path}...")
        return pd.read_parquet(cache_path)

    logger.info("📡 Initializing DataManager and Factor Registry...")
    dm = DataManager()
    
    # 1. Load raw data (DataManager internally 4 files ko merge karta hai)
    data = dm.get_master_data() 
    
    # 🔥 CRITICAL FIX: Registry ko data dene se pehle RESET INDEX zaroori hai
    # Isse 'date' aur 'ticker' index se nikal kar columns ban jayenge
    if data.index.names[0] is not None:
        data = data.reset_index()

    # 2. Check for missing features
    # Aapke logs mein 102 columns dikh rahe hain, par factors sirf 56 hain
    if data.shape[1] < 120:
        logger.info(f"🔄 Factors missing. Computing from Registry on {data.shape[0]} rows...")
        from quant_alpha.features.registry import FactorRegistry
        registry = FactorRegistry()
        
        # Parallel computation start karein
        # Registry internally compute_all use karegi
        data = registry.compute_all(data)
            
    logger.info(f"✅ Full Dataset Ready with {data.shape[1]} columns.")
    
    # Optimization: Drop any factors that are 100% NaNs (जो computation fail huye)
    data = data.dropna(axis=1, how='all')
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info(f"💾 Saving Master Dataset to {cache_path}...")
    data.to_parquet(cache_path)
    
    return data

def calculate_ranks_robust(df):
    """Ensemble logic: Averaging percentile ranks across models."""
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    for col in pred_cols:
        r_col = f'rank_{col}'
        df[r_col] = df.groupby('date')[col].rank(pct=True, method='first')
    
    rank_cols = [f'rank_{c}' for c in pred_cols]
    df['ensemble_alpha'] = df[rank_cols].mean(axis=1, skipna=True)
    return df

def run_production_pipeline():
    logger.info("🚀 Booting Optimized Alpha-Pro Engine...")
    
    # 1. Load Data (Merging 4 Parquets + Registry Computation)
    data = load_and_build_full_dataset()
    
    if 'date' not in data.columns: data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    
    # --- FIX: Ensure Unique Index (Prevents Duplicates Error) ---
    data = data.drop_duplicates(subset=['date', 'ticker'])
    
    # ---------------------------------------------------------
    # FIX: Sort is required before shift() operations
    # ---------------------------------------------------------
    data = data.sort_values(['ticker', 'date'])
    
    # ---------------------------------------------------------
    # FIX 1: Separate Training Target from Reality-based P&L
    # ---------------------------------------------------------
    # Training Target: 5-Day Forward Alpha (Aligned with Weekly Rebalance)
    # LOGIC CHANGE: 21-day target for weekly trading causes signal lag. 5-day is sharper.
    data['raw_ret_5d'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    
    # LOGIC CHANGE: Sector Neutralization (Pure Alpha)
    # Subtract the sector's average return from the stock's return.
    # This removes "Beta" and isolates "Alpha".
    sector_mean = data.groupby(['date', 'sector'])['raw_ret_5d'].transform('mean')
    data['target'] = data['raw_ret_5d'] - sector_mean
    
    # P&L Return: Next Day Return (Reality check for Backtest)
    data['pnl_return'] = data.groupby('ticker')['close'].shift(-1) / data['close'] - 1
    
    data = data.dropna(subset=['target', 'pnl_return'])
    
    # ---------------------------------------------------------
    # FIX 2: Categorical Preservation & Vectorized Imputation
    # ---------------------------------------------------------
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'pnl_return', 'date', 'ticker', 'index', 'level_0', 'raw_ret_5d']
    
    # Identify Numeric vs Categorical features
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in exclude]
    
    # All features include categorical ones (sector, industry) for GBDT models
    all_features = [c for c in data.columns if c not in exclude]
    
    # 3. Preprocessing (Numeric Only)
    # Categoricals ko touch nahi karenge, LightGBM/CatBoost unhe natively handle kar lenge
    # FIX: Normalize BEFORE Imputation so that 0 represents the Mean (Neutral Signal)
    # NOTE: This order is intentional. Imputing 0 before normalization would bias the mean.
    data = winsorize(data, numeric_features)
    data = cross_sectional_normalize(data, numeric_features)

    # FAST Imputation: Groupby transform slow hai, isliye constant fill (0) use karein
    # Z-score normalization ke baad 0 neutral signal mana jata hai. 
    # Handle numeric and categorical separately to avoid mixed types.
    logger.info(f"🛠️ Vectorized Imputation for {len(all_features)} Features...") 
    data[numeric_features] = data[numeric_features].fillna(0)

    # Optional: Fill categorical features if they exist
    cat_features = [c for c in all_features if c not in numeric_features]
    if cat_features:
        data[cat_features] = data[cat_features].fillna('Unknown')

    # --- NEW: Feature Selection Integration ---
    logger.info("🧹 Starting Feature Selection...")
    # Define meta_cols to protect
    meta_cols = ['ticker', 'date', 'target', 'pnl_return', 'open', 'high', 'low', 'close', 'volume', 'sector', 'industry', 'raw_ret_5d']
    selector = FeatureSelector(meta_cols=meta_cols)

    # 1. Filter Variance & Correlation
    data = selector.drop_low_variance(data)
    
    # --- FIX: Strict Correlation Filter (Threshold: 0.75) ---
    # Cluster analysis showed high redundancy (0.9+). We filter strictly at 0.75.
    logger.info("🧹 Applying Strict Correlation Filter (Threshold: 0.75)...")
    numeric_df = data.select_dtypes(include=[np.number])
    
    # 1. Calculate IC for each feature to decide winner
    feature_ic = {}
    for col in numeric_df.columns:
        if col not in meta_cols:
            ic = numeric_df[col].corr(data['target'])
            feature_ic[col] = abs(ic) if not np.isnan(ic) else 0.0
            
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop (Smart Selection)
    to_drop = set()
    for col in upper.columns:
        if col in meta_cols: continue
        
        # Check correlations
        high_corr_cols = upper.index[upper[col] > 0.75].tolist()
        for other_col in high_corr_cols:
            if other_col in meta_cols: continue
            
            # Keep the one with higher IC
            if feature_ic.get(col, 0) < feature_ic.get(other_col, 0):
                to_drop.add(col)
            else:
                to_drop.add(other_col)
    
    if to_drop:
        logger.info(f"📉 Dropping {len(to_drop)} redundant features (kept higher IC ones): {list(to_drop)}")
        data = data.drop(columns=list(to_drop))

    # 2. Select Top Alpha Factors
    # Using LightGBM for importance as it's fast and effective
    selected_features = selector.select_by_importance(
        df=data,
        model_class=LightGBMModel,
        model_params={'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31},
        top_n=40, # Optimized: "Less is More" - 40 orthogonal features
        target_col='target'
    )
    logger.info(f"✨ Final Selected Features: {len(selected_features)}")

    # --- FIX: Ensure Categorical Features are preserved ---
    # FeatureSelector only selects numeric features. We must manually add back 
    # categorical features like 'sector' and 'industry' if they exist.
    potential_cats = ['sector', 'industry']
    for cat in potential_cats:
        if cat in data.columns and cat not in selected_features:
            selected_features.append(cat)
            logger.info(f"➕ Manually added categorical feature: {cat}")

    # 4. Model Training Configuration (Using 'Gold' Hyperparameters)
    # n_estimators aur regularization ko optimize kiya gaya hai
    logger.info("⚙️ Applying Final Hyperopt Parameters...")
    models_config = {
        "LightGBM": (LightGBMModel, {
            'n_estimators': 300, 'learning_rate': 0.035675689449899364, 'reg_lambda': 18.37226620326944, 
            'num_leaves': 24, 'max_depth': 6, 'importance_type': 'gain'
        }),
        "XGBoost": (XGBoostModel, {
            'n_estimators': 300, 'learning_rate': 0.03261427122370329, 'max_depth': 3, 
            'reg_lambda': 67.87878943705068, 'min_child_weight': 1, 'subsample': 0.6756485311881795
        }),
        "CatBoost": (CatBoostModel, {
            'iterations': 300, 'learning_rate': 0.03693249563204964, 'depth': 5, 
            'l2_leaf_reg': 27.699310279154073, 'subsample': 0.8996377987961148, 
            'verbose': 0, 'cat_features': ['sector', 'industry']
        })
    }

    oos_preds_master = {}
    for name, (model_class, params) in models_config.items():
        # Safety check for CatBoost features
        if 'cat_features' in params:
            params['cat_features'] = [c for c in params['cat_features'] if c in selected_features]
            
        logger.info(f"🧠 Training {name}...")
        trainer = WalkForwardTrainer(
            model_class=model_class, min_train_months=36, test_months=6, 
            step_months=6, window_type='sliding', embargo_days=12, model_params=params # Optimized: 12 days is safe for 5-day target
        )
        # Train on 'target' (21-day alpha)
        oos_preds_master[name] = trainer.train(data, selected_features, 'target')

        # --- NEW: Train & Save Production Model ---
        logger.info(f"📦 Saving Production Model: {name}")
        prod_model = model_class(params=params)
        
        # Train on full dataset (using selected features)
        prod_model.fit(data[selected_features], data['target'])
        
        # Save Bridge (Model + Features)
        os.makedirs("models/production", exist_ok=True)
        save_path = f"models/production/{name.lower()}_latest.pkl"
        payload = {'model': prod_model, 'feature_names': selected_features}
        joblib.dump(payload, save_path)
        logger.info(f"✅ Saved {name} to {save_path}")

    # 5. Robust Ensemble Blending
    ensemble_df = None
    for name, df in oos_preds_master.items():
        temp = df[['date', 'ticker', 'prediction']].rename(columns={'prediction': f'pred_{name}'})
        if ensemble_df is None:
            ensemble_df = temp
        else:
            ensemble_df = pd.merge(ensemble_df, temp, on=['date', 'ticker'], how='outer')

    if ensemble_df is None:
        logger.error("❌ No predictions generated. Aborting.")
        return

    # ---------------------------------------------------------
    # FIX 3: True Backtest Evaluation
    # ---------------------------------------------------------
    # Merge target and pnl_return back for realistic P&L calculation
    ensemble_df = pd.merge(ensemble_df, data[['date', 'ticker', 'target', 'pnl_return']], on=['date', 'ticker'], how='left')
    
    final_results = calculate_ranks_robust(ensemble_df)
    
    # --- NEW: Save Predictions for Fast Backtesting ---
    pred_cache_path = r"E:\coding\quant_alpha_research\data\cache\ensemble_predictions.parquet"
    os.makedirs(os.path.dirname(pred_cache_path), exist_ok=True)
    logger.info(f"💾 Saving Ensemble Predictions to {pred_cache_path}...")
    final_results.to_parquet(pred_cache_path)

    # ---------------------------------------------------------
    # FIX 4: Run Institutional Backtest Engine
    # ---------------------------------------------------------
    logger.info("🚀 Starting Institutional Backtest Simulation...")
    
    # Prepare Data for Engine
    # Predictions: date, ticker, prediction (using ensemble_alpha)
    backtest_preds = final_results[['date', 'ticker', 'ensemble_alpha']].rename(columns={'ensemble_alpha': 'prediction'})
    
    # FIX: Explicitly drop duplicates to satisfy BacktestEngine validation
    backtest_preds = backtest_preds.drop_duplicates(subset=['date', 'ticker'])
    
    # Prices: date, ticker, close, volume, volatility
    # Ensure volatility exists
    if 'volatility' not in data.columns:
        data['volatility'] = 0.02
    
    # FIX: Include 'sector' for RiskManager if available
    price_cols = ['date', 'ticker', 'close', 'volume', 'volatility']
    if 'sector' in data.columns:
        price_cols.append('sector')
        
    backtest_prices = data[price_cols].copy()
    # FIX: Ensure prices are unique
    backtest_prices = backtest_prices.drop_duplicates(subset=['date', 'ticker'])
    
    # Initialize Engine
    engine = BacktestEngine(
        initial_capital=1_000_000,
        commission=TRANSACTION_COST,
        spread=0.0005,
        slippage=0.0002,
        position_limit=0.10, # FIX: Relaxed to 10% to allow Inverse-Vol weighting to work
        rebalance_freq='weekly', # Optimized for speed and lower churn
        use_market_impact=True,
        target_volatility=0.15, # FIX: Lowered to 15% (Institutional Standard)
        max_adv_participation=0.02, # FIX: Cap participation at 2% of ADV to minimize impact
        trailing_stop_pct=config.TRAILING_STOP_PCT # NEW: Trailing Stop
    )
    # Note: Sector limits disabled in RiskManager default (False) which is correct for Sector-Neutral Alpha
    
    # Run Simulation
    results = engine.run(
        predictions=backtest_preds,
        prices=backtest_prices,
        top_n=TOP_N_STOCKS
    )
    
    # Print Report
    print_metrics_report(results['metrics'])
    
    # --- NEW: Save Detailed Trade Report ---
    trade_report_path = r"E:\coding\quant_alpha_research\results\detailed_trade_report.csv"
    if not results['trades'].empty:
        results['trades'].to_csv(trade_report_path, index=False)
        logger.info(f"📄 Detailed Trade Report Saved: {trade_report_path}")

    # Attribution Analysis
    logger.info("🔍 Running Attribution Analysis...")
    
    # 1. Simple Attribution (PnL Drivers)
    simple_attr = SimpleAttribution()
    simple_results = simple_attr.analyze_pnl_drivers(results['trades'])
    
    print(f"\n[ PnL Attribution ]")
    print(f"  Hit Ratio:     {simple_results.get('hit_ratio',0):.2%}")
    print(f"  Win/Loss Ratio:{simple_results.get('win_loss_ratio',0):.2f}")
    print(f"  Long PnL:      ${simple_results.get('long_pnl_contribution',0):,.0f}")
    print(f"  Short PnL:     ${simple_results.get('short_pnl_contribution',0):,.0f}")

    # 2. Factor Attribution (Alpha Predictive Power)
    factor_attr = FactorAttribution()
    
    # Prepare data for IC (MultiIndex required: date, ticker)
    ic_data = final_results.dropna(subset=['ensemble_alpha', 'pnl_return'])
    if not ic_data.empty:
        factor_vals = ic_data.set_index(['date', 'ticker'])[['ensemble_alpha']]
        fwd_rets = ic_data.set_index(['date', 'ticker'])[['pnl_return']]
        
        rolling_ic = factor_attr.calculate_rolling_ic(factor_vals, fwd_rets)
        
        print(f"\n[ Factor Analysis (Ensemble Alpha) ]")
        print(f"  Mean IC:       {rolling_ic.mean():.4f}")
        print(f"  IC Std Dev:    {rolling_ic.std():.4f}")
        print(f"  IR (IC/Std):   {rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() != 0 else 0:.2f}")

if __name__ == "__main__":
    run_production_pipeline()
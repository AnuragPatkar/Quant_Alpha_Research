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

def plot_performance_tear_sheet(df):
    """
    Realistic Backtest Evaluation using Next-Day Returns.
    Fixes the 'linear approximation' issue by using actual daily realized P&L.
    """
    logger.info("📊 Generating Reality-Based Performance Tear Sheet...")
    
    # 1. Select top stocks for each day based on Ensemble Alpha
    portfolio = df.set_index('date').groupby(level='date', group_keys=False).apply(
        lambda x: x.nlargest(TOP_N_STOCKS, 'ensemble_alpha')
    )
    
    # ---------------------------------------------------------
    # FIX: Use pnl_return (Next Day) instead of target (21-Day)
    # ---------------------------------------------------------
    # Apply Stop Loss logic on daily realized returns
    portfolio['pnl_adj'] = np.where(
        portfolio['pnl_return'] <= STOCK_STOP_LOSS, 
        STOCK_STOP_LOSS + SL_SLIPPAGE_PENALTY, 
        portfolio['pnl_return']
    )
    
    # Average daily return of the selected TOP_N_STOCKS
    daily_returns = portfolio.groupby(level='date')['pnl_adj'].mean()
    
    # Full transaction cost applied on daily rebalancing (Conservative approach)
    daily_returns -= TRANSACTION_COST

    # 2. Equity Curve with Portfolio-Level Risk Management (DD Exit/Re-entry)
    equity_curve, active = [1.0], True
    v_equity, v_peak = 1.0, 1.0
    active_flags = []

    for ret in daily_returns:
        v_equity *= (1 + ret)
        v_peak = max(v_peak, v_equity)
        v_dd = (v_equity - v_peak) / v_peak if v_peak != 0 else 0

        if active:
            # Continue investing if not in a major drawdown
            equity_curve.append(equity_curve[-1] * (1 + ret))
            if v_dd <= PORTFOLIO_DD_EXIT: 
                logger.warning(f"📉 Portfolio DD Limit Hit ({v_dd:.2%}). Moving to Cash.")
                active = False
        else:
            # Stay in cash until recovery
            equity_curve.append(equity_curve[-1])
            if v_dd >= PORTFOLIO_DD_REENTRY: 
                logger.info(f"✅ Recovery Detected ({v_dd:.2%}). Re-entering Market.")
                active = True
        
        active_flags.append(1 if active else 0)

    # 3. Metrics Calculation
    equity_series = pd.Series(equity_curve[1:], index=daily_returns.index)
    
    # Annualized Return (Geometric Mean)
    days = len(equity_series)
    ann_ret = ((equity_series.iloc[-1]) ** (252 / days) - 1) * 100 if days > 0 else 0
    
    # Annualized Volatility
    vol = equity_series.pct_change().std() * np.sqrt(252)
    
    # True Sharpe Ratio
    sharpe = (ann_ret / 100) / vol if vol > 0 else 0
    
    # Maximum Drawdown
    drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax()
    max_dd = drawdown.min() * 100

    print(f"\n⭐ REALITY-BASED RESULTS")
    print(f"--------------------------------------------------")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Ann. Return:      {ann_ret:.2f}%")
    print(f"Max Drawdown:     {max_dd:.2f}%")
    print(f"Volatility:       {vol*100:.2f}%")
    print(f"--------------------------------------------------")
    
    # 4. Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(equity_series, color='#27ae60', linewidth=2, label='Alpha-Pro Strategy')
    
    # Highlight "Cash Only" (Inactive) periods in Red
    plt.fill_between(equity_series.index, equity_series.min(), equity_series.max(), 
                     where=pd.Series(active_flags, index=equity_series.index)==0, 
                     color='red', alpha=0.1, label='In Cash (Risk Management)')
    
    plt.title(f"Strategy Equity Curve (Sharpe: {sharpe:.2f})", fontsize=14)
    plt.ylabel("Portfolio Value (Starting at 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def run_production_pipeline():
    logger.info("🚀 Booting Optimized Alpha-Pro Engine...")
    
    # 1. Load Data (Merging 4 Parquets + Registry Computation)
    data = load_and_build_full_dataset()
    
    if 'date' not in data.columns: data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    
    # ---------------------------------------------------------
    # FIX: Sort is required before shift() operations
    # ---------------------------------------------------------
    data = data.sort_values(['ticker', 'date'])
    
    # ---------------------------------------------------------
    # FIX 1: Separate Training Target from Reality-based P&L
    # ---------------------------------------------------------
    # Training Target: 21-Day Forward Alpha (Noise Filter)
    data['target'] = data.groupby('ticker')['close'].shift(-21) / data['close'] - 1
    
    # P&L Return: Next Day Return (Reality check for Tear Sheet)
    data['pnl_return'] = data.groupby('ticker')['close'].shift(-1) / data['close'] - 1
    
    data = data.dropna(subset=['target', 'pnl_return'])
    
    # ---------------------------------------------------------
    # FIX 2: Categorical Preservation & Vectorized Imputation
    # ---------------------------------------------------------
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'pnl_return', 'date', 'ticker', 'index', 'level_0']
    
    # Identify Numeric vs Categorical features
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in exclude]
    
    # All features include categorical ones (sector, industry) for GBDT models
    all_features = [c for c in data.columns if c not in exclude]
    
    # FAST Imputation: Groupby transform slow hai, isliye constant fill (0) use karein
    # Z-score normalization ke baad 0 neutral signal mana jata hai. 
    # Handle numeric and categorical separately to avoid mixed types.
    logger.info(f"🛠️ Vectorized Imputation for {len(all_features)} Features...") 
    data[numeric_features] = data[numeric_features].fillna(0)

    # Optional: Fill categorical features if they exist
    cat_features = [c for c in all_features if c not in numeric_features]
    if cat_features:
        data[cat_features] = data[cat_features].fillna('Unknown')

    # 3. Preprocessing (Numeric Only)
    # Categoricals ko touch nahi karenge, LightGBM/CatBoost unhe natively handle kar lenge
    data = winsorize(data, numeric_features)
    data = cross_sectional_normalize(data, numeric_features)

    # --- NEW: Feature Selection Integration ---
    logger.info("🧹 Starting Feature Selection...")
    # Define meta_cols to protect
    meta_cols = ['ticker', 'date', 'target', 'pnl_return']
    selector = FeatureSelector(meta_cols=meta_cols)

    # 1. Filter Variance & Correlation
    data = selector.drop_low_variance(data)
    data = selector.drop_high_correlation(data)

    # 2. Select Top Alpha Factors
    # Using LightGBM for importance as it's fast and effective
    selected_features = selector.select_by_importance(
        df=data,
        model_class=LightGBMModel,
        model_params={'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31},
        top_n=40,
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
    models_config = {
        "LightGBM": (LightGBMModel, {
            'n_estimators': 600, 'learning_rate': 0.02, 'reg_lambda': 1.0, 
            'num_leaves': 39, 'importance_type': 'gain'
        }),
        "XGBoost": (XGBoostModel, {
            'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 3, 'reg_lambda': 1.0
        }),
        "CatBoost": (CatBoostModel, {
            'iterations': 600, 'l2_leaf_reg': 3.0, 'verbose': 0, 'cat_features': ['sector', 'industry']
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
            step_months=6, window_type='sliding', embargo_days=21, model_params=params
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
    
    # Tear sheet ab pnl_return (next day) use karega inflation hatane ke liye
    plot_performance_tear_sheet(final_results)

if __name__ == "__main__":
    run_production_pipeline()
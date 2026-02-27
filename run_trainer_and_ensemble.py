import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import logging
import warnings
import joblib
from tqdm import tqdm

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
from quant_alpha.optimization.allocator import PortfolioAllocator
from config.settings import config
from quant_alpha.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_ic_time_series,
    generate_tearsheet
)

# --- IMPORT ALL FACTOR MODULES (CRITICAL FOR REGISTRY) ---
# Without these imports, FactorRegistry won't know how to calculate Fundamentals/Earnings
import quant_alpha.features.technical.momentum
import quant_alpha.features.technical.volatility
import quant_alpha.features.technical.volume
import quant_alpha.features.technical.mean_reversion
import quant_alpha.features.fundamental.value
import quant_alpha.features.fundamental.quality
import quant_alpha.features.fundamental.growth
import quant_alpha.features.fundamental.financial_health
import quant_alpha.features.earnings.surprises
import quant_alpha.features.earnings.estimates
import quant_alpha.features.earnings.revisions
import quant_alpha.features.alternative.macro
import quant_alpha.features.alternative.sentiment
import quant_alpha.features.alternative.inflation
import quant_alpha.features.composite.macro_adjusted
import quant_alpha.features.composite.system_health
import quant_alpha.features.composite.smart_signals

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
    cache_path = config.CACHE_DIR / "master_data_with_factors.parquet"
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
    # Heuristic: If columns < 120, likely missing calculated factors (Raw data is usually ~15-20 cols)
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
    
    if not pred_cols:
        logger.warning("⚠️ No prediction columns found for ensemble. Returning 0 alpha.")
        df['ensemble_alpha'] = 0.0
        return df

    for col in pred_cols:
        r_col = f'rank_{col}'
        df[r_col] = df.groupby('date')[col].rank(pct=True, method='first')
    
    rank_cols = [f'rank_{c}' for c in pred_cols]
    df['ensemble_alpha'] = df[rank_cols].mean(axis=1, skipna=True)
    return df

def sector_neutral_normalize(df, features, sector_col='sector'):
    """Normalize features within each sector group (Vectorized)."""
    if sector_col not in df.columns:
        logger.warning("Sector column missing. Falling back to global normalization.")
        return cross_sectional_normalize(df, features)
    
    logger.info(f"⚖️ Applying Sector-Neutral Normalization on {len(features)} features...")
    g = df.groupby(['date', sector_col])[features]
    df[features] = (df[features] - g.transform('mean')) / (g.transform('std') + 1e-8)
    return df

def generate_optimized_weights(predictions, prices_df, method='mean_variance'):
    """
    Runs Portfolio Optimization on a rolling basis.
    """
    logger.info(f"⚖️  Running Portfolio Optimization ({method})...")
    
    # Initialize Allocator
    allocator = PortfolioAllocator(
        method=method,
        risk_aversion=config.OPT_RISK_AVERSION,
        fraction=config.OPT_KELLY_FRACTION,
        tau=0.05
    )
    
    # Prepare Data
    # Pivot prices for covariance: Index=Date, Columns=Ticker
    price_matrix = prices_df.pivot(index='date', columns='ticker', values='close')
    
    # Get unique rebalance dates (Weekly)
    unique_dates = sorted(predictions['date'].unique())
    
    optimized_allocations = []
    
    # Rolling Optimization Loop
    for current_date in tqdm(unique_dates, desc="Optimizing Portfolio"):
        # 1. Get Alpha Signals for this date
        day_preds = predictions[predictions['date'] == current_date]
        if day_preds.empty: continue
        
        # Select Top N candidates based on Alpha Score
        top_candidates = day_preds.sort_values('ensemble_alpha', ascending=False).head(TOP_N_STOCKS)
        tickers = top_candidates['ticker'].tolist()
        
        # Expected Returns (Mapping Alpha 0-1 to expected return magnitude)
        # In production, you might scale this to realistic annualized returns (e.g. 0.10 - 0.30)
        expected_returns = top_candidates.set_index('ticker')['ensemble_alpha'].to_dict()
        
        # 2. Calculate Historical Covariance (Lookback Window)
        start_date = current_date - pd.Timedelta(days=config.OPT_LOOKBACK_DAYS)
        
        # Slice price matrix for speed
        hist_prices = price_matrix.loc[start_date:current_date, tickers]
        
        if len(hist_prices) < 60: # Need at least ~3 months for valid covariance
            # Fallback to Equal Weight if not enough history
            weights = {t: 1.0/len(tickers) for t in tickers}
        else:
            # Calculate Covariance
            returns = hist_prices.pct_change().dropna()
            if returns.empty:
                weights = {t: 1.0/len(tickers) for t in tickers}
            else:
                cov_matrix = returns.cov() * 252 # Annualized
                
                # 3. Run Allocator
                # Market Caps for Black-Litterman
                market_caps = {}
                if 'market_cap' in prices_df.columns:
                    mc_slice = prices_df[(prices_df['date'] == current_date) & (prices_df['ticker'].isin(tickers))]
                    market_caps = mc_slice.set_index('ticker')['market_cap'].to_dict()
                
                # Fill missing/zero caps with default to prevent BL crash
                for t in tickers:
                    if t not in market_caps or pd.isna(market_caps.get(t, 0)) or market_caps.get(t, 0) == 0:
                        market_caps[t] = 1e9 # Default 1B to avoid zero division
                
                weights = allocator.allocate(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    market_caps=market_caps,
                    risk_free_rate=config.RISK_FREE_RATE
                )
        
        # 4. Store Results
        for ticker, w in weights.items():
            optimized_allocations.append({'date': current_date, 'ticker': ticker, 'optimized_weight': w})
            
    return pd.DataFrame(optimized_allocations)

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
    
    # Target: Sector-Neutral Return
    # REVERT: Removed Volatility Scaling to restore 152% return logic.
    # We want the model to predict magnitude of return, not just risk-adjusted quality.
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
    # OPTIMIZATION: Use Sector Neutral Normalization instead of Global
    data = sector_neutral_normalize(data, numeric_features)

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
    
    # --- FIX: Strict Correlation Filter ---
    # Cluster analysis showed high redundancy. We filter strictly.
    corr_threshold = config.FEATURE_CORRELATION_THRESHOLD
    logger.info(f"🧹 Applying Strict Correlation Filter (Threshold: {corr_threshold})...")
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
        high_corr_cols = upper.index[upper[col] > corr_threshold].tolist()
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
            'verbose': 0, 'cat_features': ['sector', 'industry'], 'allow_writing_files': False
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
            step_months=6, window_type='sliding', embargo_days=12, model_params=params # 12 days embargo > 5 day target (Safe)
        )
        # Train on 'target' (21-day alpha)
        oos_preds_master[name] = trainer.train(data, selected_features, 'target')

        # --- NEW: Train & Save Production Model ---
        logger.info(f"📦 Saving Production Model: {name}")
        prod_model = model_class(params=params)
        
        # Train on full dataset (using selected features)
        prod_model.fit(data[selected_features], data['target'])
        
        # Save Bridge (Model + Features)
        save_dir = config.MODELS_DIR / "production"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f"{name.lower()}_latest.pkl"
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
    pred_cache_path = config.CACHE_DIR / "ensemble_predictions.parquet"
    logger.info(f"💾 Saving Ensemble Predictions to {pred_cache_path}...")
    final_results.to_parquet(pred_cache_path)

    # ---------------------------------------------------------
    # FIX 4: Run Institutional Backtest Engine
    # ---------------------------------------------------------
    logger.info("🚀 Starting Institutional Backtest Simulation...")
    
    # Prepare Data for Engine
    # Predictions: date, ticker, prediction (using ensemble_alpha)
    backtest_preds = final_results[['date', 'ticker', 'ensemble_alpha']].rename(columns={'ensemble_alpha': 'prediction'})
    # Prepare Prices for Optimization
    price_cols = ['date', 'ticker', 'close']
    backtest_prices_simple = data[price_cols].drop_duplicates(subset=['date', 'ticker'])
    
    # Prices: date, ticker, close, volume, volatility
    # Ensure volatility exists
    if 'volatility' not in data.columns:
        data['volatility'] = 0.02
    
    # FIX: Include 'sector' for RiskManager if available
    full_price_cols = ['date', 'ticker', 'close', 'open', 'volume', 'volatility']
    if 'sector' in data.columns:
        full_price_cols.append('sector')
        
    backtest_prices = data[full_price_cols].copy()
    # FIX: Ensure prices are unique
    backtest_prices = backtest_prices.drop_duplicates(subset=['date', 'ticker'])
    
    # --- NEW: Loop through ALL Optimization Methods ---
    # SPEED FIX: Only running 'mean_variance' by default. Uncomment others for full research.
    optimization_methods = ['mean_variance'] #, 'risk_parity', 'kelly', 'black_litterman']
    
    for method in optimization_methods:
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING BACKTEST WITH: {method.upper()}")
        print(f"{'='*60}")
        
        # Generate weights for this specific method
        opt_weights_df = generate_optimized_weights(final_results, backtest_prices_simple, method=method)
        
        if not opt_weights_df.empty:
            current_preds = opt_weights_df.rename(columns={'optimized_weight': 'prediction'})
            logger.info(f"✅ Using {method} Weights for Backtest.")
        else:
            current_preds = final_results[['date', 'ticker', 'ensemble_alpha']].rename(columns={'ensemble_alpha': 'prediction'})
            logger.warning(f"⚠️ {method} Optimization returned empty. Falling back to Raw Alpha Scores.")
        
        current_preds = current_preds.drop_duplicates(subset=['date', 'ticker'])

        # Initialize Engine
        engine = BacktestEngine(
            initial_capital=1_000_000,
            commission=TRANSACTION_COST,
            spread=0.0005,
            slippage=0.0002,
            position_limit=0.10, # FIX: Relaxed to 10%
            rebalance_freq='weekly',
            use_market_impact=True,
            target_volatility=0.15,
            max_adv_participation=0.02,
            trailing_stop_pct=getattr(config, 'TRAILING_STOP_PCT', 0.10),
            execution_price='open', # Trade at Next Open (Realistic)
            max_turnover=0.20       # Limit turnover to 20% per rebalance
        )
        
        # Run Simulation
        results = engine.run(
            predictions=current_preds,
            prices=backtest_prices,
            top_n=TOP_N_STOCKS
        )
        
        # Print Report for this method
        print_metrics_report(results['metrics'])
        
        # --- VISUALIZATION ---
        plot_dir = config.RESULTS_DIR / "plots" / method
        os.makedirs(plot_dir, exist_ok=True)
        logger.info(f"📊 Generating Plots for {method} in {plot_dir}...")
        
        # 1. Equity Curve & Drawdown
        plot_equity_curve(results['equity_curve'], save_path=plot_dir / "equity_curve.png")
        plot_drawdown(results['equity_curve'], save_path=plot_dir / "drawdown.png")
        
        # 2. Monthly Heatmap
        eq_df = results['equity_curve'].copy()
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        plot_monthly_heatmap(eq_df.set_index('date')['total_value'].pct_change(), save_path=plot_dir / "monthly_heatmap.png")
        
        # 3. Full Tearsheet
        generate_tearsheet(results, save_path=plot_dir / "tearsheet.pdf")
        
        # Save Trade Report
        trade_report_path = config.RESULTS_DIR / f"detailed_trade_report_{method}.csv"
        if not results['trades'].empty:
            results['trades'].to_csv(trade_report_path, index=False)
            logger.info(f"📄 Detailed Trade Report Saved: {trade_report_path}")
            
            # Attribution Analysis (Per Strategy)
            simple_attr = SimpleAttribution()
            simple_results = simple_attr.analyze_pnl_drivers(results['trades'])
            
            print(f"\n[ PnL Attribution ({method}) ]")
            print(f"  Hit Ratio:     {simple_results.get('hit_ratio',0):.2%}")
            print(f"  Win/Loss Ratio:{simple_results.get('win_loss_ratio',0):.2f}")
            print(f"  Long PnL:      ${simple_results.get('long_pnl_contribution',0):,.0f}")
            print(f"  Short PnL:     ${simple_results.get('short_pnl_contribution',0):,.0f}")

    # Factor Attribution (Alpha Predictive Power) - Runs once as Alpha is constant
    logger.info("🔍 Running Factor Analysis (Signal Quality)...")

    factor_attr = FactorAttribution()
    
    # Prepare data for IC (MultiIndex required: date, ticker)
    # FIX: Calculate IC against the correct 5-day forward return to match the training target.
    # This was a mismatch causing low IC in the report.
    ic_data = pd.merge(final_results, data[['date', 'ticker', 'raw_ret_5d']], on=['date', 'ticker'], how='left')
    ic_data = ic_data.dropna(subset=['ensemble_alpha', 'raw_ret_5d'])
    if not ic_data.empty:
        factor_vals = ic_data.set_index(['date', 'ticker'])[['ensemble_alpha']]
        fwd_rets = ic_data.set_index(['date', 'ticker'])[['raw_ret_5d']]
        
        rolling_ic = factor_attr.calculate_rolling_ic(factor_vals, fwd_rets)
        
        # --- VISUALIZATION (IC) ---
        ic_plot_path = config.RESULTS_DIR / "plots" / "factor_analysis" / "ic_time_series.png"
        os.makedirs(os.path.dirname(ic_plot_path), exist_ok=True)
        plot_ic_time_series(rolling_ic, save_path=ic_plot_path)
        
        print(f"\n[ Factor Analysis (Ensemble Alpha) ]")
        print(f"  Mean IC:       {rolling_ic.mean():.4f}")
        print(f"  IC Std Dev:    {rolling_ic.std():.4f}")
        print(f"  IR (IC/Std):   {rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() != 0 else 0:.2f}")

if __name__ == "__main__":
    run_production_pipeline()
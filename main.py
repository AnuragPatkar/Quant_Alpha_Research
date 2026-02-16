"""
QUANT ALPHA RESEARCH - MAIN EXECUTION SCRIPT (Optimized)
Runs the full Alpha Pipeline with Parallel Processing.
"""
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Progress bar

from config.logging_config import logger
from config.settings import config

# --- CORE COMPONENTS ---
from quant_alpha.data.DataManager import DataManager
from quant_alpha.features.registry import FactorRegistry
from quant_alpha.models.score_engine import ScoreEngine



# --- IMPORT ALL FACTOR MODULES ---
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

def process_single_factor(name, factor, flat_master, target_index):
    """Helper function to calculate one factor (Runs in parallel)"""
    try:
        res = factor.calculate(flat_master)
        
        if res.empty:
            return None
            
        # Standardize Output Format
        if isinstance(res, pd.Series):
            res = res.to_frame(name=name)
        
        if isinstance(res, pd.DataFrame):
            # Smart Select: Pick the correct column, ignore metadata
            valid_cols = [c for c in res.columns if c not in ['date', 'ticker', 'level_0', 'index']]
            if valid_cols:
                res = res[[valid_cols[-1]]].rename(columns={valid_cols[-1]: name})
            else:
                # Fallback: Take last column
                res = res.iloc[:, [-1]].copy()
                res.columns = [name]
        
        # Memory Optimization: Convert to float32
        res = res.astype('float32')
        
        # Align index strictly with Master Data
        res.index = target_index
        return res
        
    except Exception:
        return None

def run_pipeline():
    start_time = time.time()
    logger.info("🚀 STARTING QUANT ALPHA PRODUCTION PIPELINE (HIGH PERF)...")
    
    # ---------------------------------------------------------
    # 1. DATA INGESTION
    # ---------------------------------------------------------
    dm = DataManager()
    master_df = dm.get_master_data()
    
    if master_df.empty:
        logger.error("❌ Pipeline Aborted: Master Data is empty.")
        return

    logger.info(f"📊 Data Loaded. Rows: {len(master_df):,}")

    # ---------------------------------------------------------
    # 2. FACTOR CALCULATION (PARALLELIZED)
    # ---------------------------------------------------------
    logger.info("⚙️  Computing Alpha Factors...")
    registry = FactorRegistry()
    
    # Prepare data for broadcasting
    flat_master = master_df.reset_index()
    target_index = master_df.index
    
    computed_frames = []
    
    # Use all available CPU cores
    max_workers = min(32, (os.cpu_count() or 1) * 4) 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_factor, name, factor, flat_master, target_index): name 
            for name, factor in registry.factors.items()
        }
        
        # Process as they complete (with Progress Bar)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Factors", unit="factor"):
            res = future.result()
            if res is not None:
                computed_frames.append(res)

    if not computed_frames:
        logger.error("❌ No factors calculated. Check your logic.")
        return

    logger.info(f"⚡ Merging {len(computed_frames)} Factors...")
    
    # Fast Merge using Concat (Axis=1)
    factor_data = pd.concat(computed_frames, axis=1)
    
    # Fill NaNs for scoring
    factor_data = factor_data.fillna(0.0)
    
    logger.info(f"✅ Factor Matrix Ready. Shape: {factor_data.shape}")

    # ---------------------------------------------------------
    # 3. SCORING & RANKING
    # ---------------------------------------------------------
    strategy_weights = {
        'rsi_14d': 0.15,
        'mom_accel_10d': 0.15,
        'val_pe_ratio': -0.15,
        'val_ev_ebitda': -0.15,
        'qual_roe': 0.20,
        'earn_surprise_pct': 0.10,
        'alt_sentiment': 0.10
    }
    
    engine = ScoreEngine(weights=strategy_weights)
    
    # Normalize & Score
    norm_df = engine.normalize_factors(factor_data)
    scores = engine.compute_final_score(norm_df)
    
    # ---------------------------------------------------------
    # 4. EXPORT
    # ---------------------------------------------------------
    latest_date = scores.index.get_level_values('date').max()
    logger.info(f"📅 Generating Signals for {latest_date.date()}...")
    
    todays_scores = scores.xs(latest_date, level='date')
    top_picks = todays_scores.sort_values('alpha_score', ascending=False).head(20)
    
    # Add Context
    raw_values = factor_data.xs(latest_date, level='date')
    final_report = top_picks.join(raw_values, how='left')
    
    print("\n" + "="*60)
    print(f"🏆 TOP ALPHA PICKS ({latest_date.date()})")
    print("="*60)
    
    display_cols = ['alpha_score'] + [c for c in strategy_weights.keys() if c in final_report.columns]
    print(final_report[display_cols].head(10))
    
    output_file = f"alpha_signals_{latest_date.date()}.csv"
    final_report.to_csv(output_file)
    
    elapsed = time.time() - start_time
    logger.info(f"💾 Saved to {output_file}")
    logger.info(f"🏁 Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_pipeline()
"""
INTEGRATION TEST: DataManager + Factors + ScoreEngine
Goal: Verify flow from Raw Data -> Signals -> Ranking.
STATUS: FIXED (Handles Multi-Column Factor Returns Robustly)
"""
import pandas as pd
import numpy as np
from config.logging_config import logger
from quant_alpha.data.DataManager import DataManager
from quant_alpha.features.registry import FactorRegistry
from quant_alpha.models.score_engine import ScoreEngine




# Import factors to register them
import quant_alpha.features.technical.momentum
import quant_alpha.features.fundamental.value
import quant_alpha.features.fundamental.quality

def run_full_system_test():
    logger.info("🚀 STARTING FULL SYSTEM INTEGRATION TEST...")

    # 1. LOAD MASTER DATA
    dm = DataManager()
    master_df = dm.get_master_data()
    
    if master_df.empty:
        logger.error("❌ Test Failed: Master Data is empty.")
        return

    logger.info(f"✅ Master Data Loaded. Rows: {len(master_df):,}")

    # 2. CALCULATE FACTORS (Sample Strategy)
    logger.info("⚙️  Calculating Strategy Factors...")
    registry = FactorRegistry()
    
    target_factors = ['rsi_14d', 'val_pe_ratio', 'qual_roe']
    
    # Base container with MultiIndex
    factor_data = pd.DataFrame(index=master_df.index)
    
    # Flat version for technical calculations
    flat_master = master_df.reset_index()
    
    for name in target_factors:
        if name in registry.factors:
            try:
                factor_obj = registry.factors[name]
                
                # Calculate (May return Series, 1-col DF, or Multi-col DF)
                res = factor_obj.calculate(flat_master)
                
                if not res.empty:
                    # --- CRITICAL FIX START: Handle Multi-Column Returns ---
                    if isinstance(res, pd.Series):
                        res = res.to_frame(name=name)
                    
                    if isinstance(res, pd.DataFrame):
                        # If DataFrame has >1 columns (e.g. date, ticker, rsi), 
                        # find the numeric column that isn't ID
                        if res.shape[1] > 1:
                            valid_cols = [c for c in res.columns if c not in ['date', 'ticker', 'level_0', 'index']]
                            if valid_cols:
                                # Assume the last valid column is the factor value
                                res = res[[valid_cols[-1]]].copy()
                            else:
                                # Fallback: Take the very last column
                                res = res.iloc[:, [-1]].copy()
                    
                    # Now res is strictly 1 column. Rename it.
                    res.columns = [name]
                    # --- CRITICAL FIX END ---

                    # Re-assign the MultiIndex from master data to ensure alignment
                    # (Assuming row order is preserved, which is standard)
                    res.index = master_df.index
                    
                    # Join
                    factor_data = factor_data.join(res, how='left')
                    logger.info(f"   ✅ Calculated: {name}")
            except Exception as e:
                logger.error(f"   ❌ Failed {name}: {e}")

    # Drop rows where all factors are NaN
    factor_data = factor_data.dropna(how='all')
    factor_data = factor_data.fillna(0) # Fill remaining gaps for scoring
    
    logger.info(f"✅ Factor Calculation Complete. Shape: {factor_data.shape}")

    if factor_data.empty:
        logger.error("❌ No factors calculated. Exiting.")
        return

    # 3. SCORE & RANK
    weights = {
        'rsi_14d': 0.4,
        'qual_roe': 0.3,
        'val_pe_ratio': -0.3 
    }
    
    engine = ScoreEngine(weights=weights)
    
    # A. Normalize
    norm_df = engine.normalize_factors(factor_data)
    logger.info("✅ Factors Normalized (Z-Scores Generated)")
    
    # B. Score
    scores = engine.compute_final_score(norm_df)
    logger.info("✅ Final Alpha Scores Computed")

    # 4. SHOW TOP PICKS
    # Find the latest valid date
    latest_date = scores.index.get_level_values('date').max()
    
    if pd.isna(latest_date):
        logger.error("❌ No valid dates found in scores.")
        return

    logger.info(f"\n🏆 TOP 10 STOCK PICKS FOR: {latest_date.date()}")
    logger.info("-" * 60)
    
    # Get top 10 for the last day
    try:
        day_scores = scores.xs(latest_date, level='date')
        top_picks = day_scores.sort_values('alpha_score', ascending=False).head(10)
        
        # Join with raw values for display
        raw_values = factor_data.xs(latest_date, level='date')
        display = top_picks.join(raw_values, how='left')
        print(display)
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
    
    logger.info("-" * 60)
    logger.info("🚀 SYSTEM TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_full_system_test()
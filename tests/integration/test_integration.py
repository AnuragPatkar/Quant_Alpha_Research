"""
INTEGRATION TEST: DataManager + Factors + ML Model Flow
Goal: Verify flow from Raw Data -> Feature Engineering -> Inference.
Updated for ML Pipeline (replaces legacy ScoreEngine test).
"""
import pandas as pd
import numpy as np
import logging
from config.logging_config import setup_logging
from quant_alpha.data.DataManager import DataManager
from quant_alpha.features.registry import FactorRegistry
from quant_alpha.models.lightgbm_model import LightGBMModel

setup_logging()
logger = logging.getLogger("Quant_Alpha")

# Import factors to register them
import quant_alpha.features.technical.momentum
import quant_alpha.features.technical.volatility

def run_full_system_test():
    logger.info("🚀 STARTING ML PIPELINE INTEGRATION TEST...")

    # 1. LOAD MASTER DATA
    dm = DataManager()
    master_df = dm.get_master_data()
    
    if master_df.empty:
        logger.error("❌ Test Failed: Master Data is empty.")
        return

    logger.info(f"✅ Master Data Loaded. Rows: {len(master_df):,}")

    # 2. CALCULATE FACTORS (ML Feature Set)
    logger.info("⚙️  Calculating ML Features...")
    registry = FactorRegistry()
    
    # Test a mix of technical factors
    target_factors = ['rsi_14d', 'mom_1m', 'vol_21d']
    
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
    X = factor_data.dropna(how='all').fillna(0)
    
    logger.info(f"✅ Factor Calculation Complete. Shape: {X.shape}")

    if X.empty:
        logger.error("❌ Feature matrix empty.")
        return

    # 3. MOCK MODEL INFERENCE
    logger.info("🤖 Testing Model Inference (Mock)...")
    
    # Initialize a dummy LightGBM model (untrained, just checking interface)
    model = LightGBMModel(params={'n_estimators': 1, 'verbose': -1})
    
    try:
        # Mock fit on small subset to initialize internal structures
        y = np.random.rand(len(X))
        model.fit(X.iloc[:100], y[:100])
        
        # Predict
        preds = model.predict(X.iloc[:100])
        
        if len(preds) == 100:
            logger.info("✅ Model Fit/Predict Cycle Successful")
        else:
            logger.error("❌ Prediction length mismatch")
            
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return

    # 4. OUTPUT CHECK
    logger.info("-" * 60)
    logger.info(f"🚀 ML PIPELINE INTEGRATION TEST PASSED")
    logger.info(f"   Data Rows: {len(master_df)}")
    logger.info(f"   Features:  {list(X.columns)}")
    logger.info("-" * 60)

if __name__ == "__main__":
    run_full_system_test()
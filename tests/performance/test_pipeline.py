"""
Comprehensive Test Pipeline for ALL 120 Factor Modules
STATUS: FIXED (Datetime Precision & MultiIndex Handling)
"""

import pandas as pd
import numpy as np
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.logging_config import logger
from config.settings import config
from quant_alpha.data.price_loader import PriceLoader
from quant_alpha.data.fundamental_loader import FundamentalLoader
from quant_alpha.data.earnings_loader import EarningsLoader
from quant_alpha.data.alternative_loader import AlternativeLoader
from quant_alpha.features.registry import FactorRegistry

# Import all modules to trigger registration
import quant_alpha.features.technical.momentum
import quant_alpha.features.technical.volume
import quant_alpha.features.technical.volatility
import quant_alpha.features.technical.mean_reversion
import quant_alpha.features.fundamental.value
import quant_alpha.features.fundamental.quality
import quant_alpha.features.fundamental.growth
import quant_alpha.features.fundamental.financial_health
import quant_alpha.features.earnings.surprises
import quant_alpha.features.earnings.revisions
import quant_alpha.features.earnings.estimates
import quant_alpha.features.alternative.macro
import quant_alpha.features.alternative.sentiment
import quant_alpha.features.alternative.inflation
import quant_alpha.features.composite.macro_adjusted
import quant_alpha.features.composite.system_health
import quant_alpha.features.composite.smart_signals

warnings.filterwarnings('ignore')

class FactorTestSuite:
    def __init__(self):
        self.price_df = None
        self.fundamental_df = None
        self.earnings_df = None
        self.alternative_df = None
        self.registry = FactorRegistry()
        self.results = []
        self.errors = []
        self.quality_issues = []
        
    def test_price_factors(self):
        logger.info("="*80 + "\nTEST 1: PRICE-BASED TECHNICAL FACTORS\n" + "="*80)
        try:
            self.price_df = PriceLoader().get_data()
            if self.price_df.empty: return False
            logger.info(f"✅ Loaded {len(self.price_df):,} price records")
        except Exception as e:
            logger.error(f"❌ Price Loading Error: {e}")
            return False
        
        factors = {k: v for k, v in self.registry.factors.items() if v.category == 'technical'}
        return self._run_factor_batch(factors, self.price_df, "Technical")
    
    def test_fundamental_factors(self):
        logger.info("="*80 + "\nTEST 2: FUNDAMENTAL FACTORS\n" + "="*80)
        try:
            self.fundamental_df = FundamentalLoader().get_data()
            if self.fundamental_df.empty: return True
            logger.info(f"✅ Loaded {len(self.fundamental_df):,} records")
        except Exception as e:
            logger.error(f"❌ Fundamental Loading Error: {e}")
            return True
        
        factors = {k: v for k, v in self.registry.factors.items() if v.category == 'fundamental'}
        return self._run_factor_batch(factors, self.fundamental_df, "Fundamental")

    def test_earnings_factors(self):
        logger.info("="*80 + "\nTEST 3: EARNINGS FACTORS\n" + "="*80)
        try:
            raw_earnings = EarningsLoader().get_data()
            if raw_earnings.empty: return True
            
            # Merge Price for Valuation Factors (SUE, etc.)
            if self.price_df is not None and not self.price_df.empty:
                logger.info("🔄 Merging Price Data with Earnings...")
                p_reset = self.price_df.reset_index()
                e_reset = raw_earnings.reset_index()
                
                # Merge
                merged = pd.merge(e_reset, p_reset[['date', 'ticker', 'close']], 
                                  on=['date', 'ticker'], how='left')
                
                # CRITICAL FIX: Keep as standard DataFrame (reset_index) for the batch run
                # because some factors check "if 'date' in df.columns"
                self.earnings_df = merged
                logger.info(f"✅ Merged Earnings & Price. Shape: {self.earnings_df.shape}")
            else:
                self.earnings_df = raw_earnings.reset_index()

        except Exception as e:
            logger.error(f"❌ Earnings Loading Error: {e}")
            return False
        
        factors = {k: v for k, v in self.registry.factors.items() if v.category == 'earnings'}
        return self._run_factor_batch(factors, self.earnings_df, "Earnings")
    
    def test_alternative_factors(self):
        logger.info("="*80 + "\nTEST 4: ALTERNATIVE DATA FACTORS\n" + "="*80)
        try:
            raw_alt = AlternativeLoader().get_data()
            if raw_alt.empty: return True
            
            # Inject Dummy Ticker for Pipeline Compatibility
            if 'ticker' not in raw_alt.columns:
                logger.info("🔧 Injecting 'MACRO' ticker...")
                df = raw_alt.copy()
                if 'date' not in df.columns: df = df.reset_index()
                df['ticker'] = 'MACRO'
                # Ensure date is ns
                df['date'] = pd.to_datetime(df['date']).astype('datetime64[ns]')
                self.alternative_df = df.set_index(['date', 'ticker']).sort_index()
            else:
                self.alternative_df = raw_alt
                
            logger.info(f"✅ Loaded {len(self.alternative_df):,} records")
        except Exception as e:
            logger.error(f"❌ Alternative Loading Error: {e}")
            return True
        
        factors = {k: v for k, v in self.registry.factors.items() if v.category == 'alternative'}
        return self._run_factor_batch(factors, self.alternative_df, "Alternative")
    
    def test_composite_factors(self):
        logger.info("="*80 + "\nTEST 5: COMPOSITE FACTORS\n" + "="*80)
        
        if self.price_df is None or self.alternative_df is None:
            logger.warning("⚠️  Skipping composite (Missing Data)")
            return True
            
        logger.info("\n[MERGING] Price + Alternative Data...")
        try:
            # 1. Prepare Price (Target)
            price_flat = self.price_df.reset_index()
            # Ensure proper datetime format
            price_flat['date'] = pd.to_datetime(price_flat['date']).astype('datetime64[ns]')
            
            # 2. Prepare Macro (Source)
            alt_flat = self.alternative_df.reset_index()
            # Remove dummy ticker if present
            if 'ticker' in alt_flat.columns and (alt_flat['ticker'] == 'MACRO').all():
                alt_flat = alt_flat.drop(columns=['ticker'])
                
            alt_flat['date'] = pd.to_datetime(alt_flat['date']).astype('datetime64[ns]')
            
            # 3. Merge AsOf (Align Macro data with Stock dates)
            # direction='backward' means use the latest available macro data for the stock date
            composite_df = pd.merge_asof(
                price_flat.sort_values('date'),
                alt_flat.sort_values('date'),
                on='date',
                direction='backward'
            )
            
            # 4. Merge Fundamentals (if available)
            if self.fundamental_df is not None:
                fund_flat = self.fundamental_df.reset_index()
                # Remove duplicate columns (except ticker) to avoid merge errors
                cols_to_use = [c for c in fund_flat.columns if c not in composite_df.columns or c == 'ticker']
                
                composite_df = pd.merge(
                    composite_df, 
                    fund_flat[cols_to_use], 
                    on='ticker', 
                    how='left'
                )

            # =========== CRITICAL FIX ===========
            # Do NOT set index here. Keep 'date' and 'ticker' as COLUMNS.
            # Most composite logic expects a flat DataFrame to perform groupby('ticker').
            
            # Ensure it is sorted for any rolling logic inside factors
            composite_df = composite_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
            
            logger.info(f"✅ Merged Composite Data. Shape: {composite_df.shape}")
            
            # Check for critical columns
            if 'vix_close' in composite_df.columns:
                logger.info(f"   VIX Data Present (Nulls: {composite_df['vix_close'].isna().sum()})")
            
        except Exception as e:
            logger.error(f"❌ Failed to merge data: {e}")
            return True
        
        comp_factors = {k: v for k, v in self.registry.factors.items() if v.category == 'composite'}
        return self._run_factor_batch(comp_factors, composite_df, "Composite")
    
    def _run_factor_batch(self, factors, data, name):
        if not factors: return True
        logger.info(f"\n[TESTING] {len(factors)} {name} Factors...")
        logger.info("-"*80)
        
        success_count = 0
        
        def _compute(fname, finst, fdata):
            try:
                start = time.time()
                res = finst.calculate(fdata)
                return {'status': 'ok', 'name': fname, 'res': res, 'time': time.time() - start, 'obj': finst}
            except Exception as e:
                return {'status': 'err', 'name': fname, 'msg': str(e)}

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_compute, k, v, data): k for k, v in factors.items()}
            
            for fut in as_completed(futures):
                r = fut.result()
                fname = r['name']
                
                if r['status'] == 'ok':
                    try:
                        df = r['res']
                        col = r['obj'].name
                        if col not in df.columns: raise ValueError("Column missing in result")
                        
                        series = df[col]
                        stats = self._calculate_stats(series, fname, r['time'])
                        self.results.append(stats)
                        self._check_factor_quality(series, fname)
                        
                        logger.info(f"✅ {fname:30s} | Mean:{stats['mean']:8.4f} | Non-Null:{stats['non_null_pct']:6.2f}% | Time:{r['time']:6.2f}s")
                        success_count += 1
                    except Exception as e:
                        logger.error(f"❌ {fname}: {e}")
                        self.errors.append(f"{fname}: {e}")
                else:
                    logger.error(f"❌ {fname}: {r['msg']}")
                    self.errors.append(f"{fname}: {r['msg']}")

        logger.info("-"*80)
        return success_count == len(factors)

    def _calculate_stats(self, series, name, elapsed):
        def clean(v): return 0.0 if (pd.isna(v) or np.isinf(v)) else (0.0 if abs(v) < 1e-9 else v)
        return {
            'factor': name, 'time_seconds': elapsed,
            'mean': clean(series.mean()), 'std': clean(series.std()),
            'non_null_pct': (series.notna().sum() / len(series) * 100),
            'inf_count': np.isinf(series).sum(),
            'skewness': series.skew(), 'kurtosis': series.kurtosis()
        }
    
    def _check_factor_quality(self, series, name):
        if series.isna().all(): self.quality_issues.append(f"{name}: All NaN")
        elif series.nunique() == 1: self.quality_issues.append(f"{name}: All values identical")

    def generate_summary_report(self):
        logger.info("\n" + "="*80 + "\nFINAL SUMMARY\n" + "="*80)
        if self.errors:
            logger.info("❌ ERRORS FOUND:")
            for e in self.errors: logger.info(f" - {e}")
        else:
            logger.info("✅ ALL SYSTEMS GO")
        
        if self.quality_issues:
            logger.info("\n⚠️ QUALITY WARNINGS:")
            for q in set(self.quality_issues): logger.info(f" - {q}")

    def generate_code_review(self): pass

if __name__ == "__main__":
    suite = FactorTestSuite()
    suite.test_price_factors()
    suite.test_fundamental_factors()
    suite.test_earnings_factors()
    suite.test_alternative_factors()
    suite.test_composite_factors()
    suite.generate_summary_report()
"""
Comprehensive Test Pipeline for ALL Factor Modules
- value.py (7 factors)
- technical/momentum.py (15 factors)
- technical/volume.py (12 factors)
- technical/volatility.py (12 factors)
- technical/mean_reversion.py (13 factors)

# Total: 59+ factors

# Code Review & Quality Metrics Included
# """

import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from config.logging_config import logger
from config.settings import config
from quant_alpha.data.price_loader import PriceLoader
from quant_alpha.data.fundamental_loader import FundamentalLoader
from quant_alpha.features.registry import FactorRegistry

# --- IMPORT ALL TECHNICAL FACTORS (triggers @FactorRegistry.register decorators) ---
import quant_alpha.features.technical.momentum
import quant_alpha.features.technical.volume
import quant_alpha.features.technical.volatility
import quant_alpha.features.technical.mean_reversion

# --- IMPORT ALL FUNDAMENTAL FACTORS ---
import quant_alpha.features.fundamental.value
import quant_alpha.features.fundamental.quality
import quant_alpha.features.fundamental.growth
import quant_alpha.features.fundamental.financial_health

warnings.filterwarnings('ignore')



class FactorTestSuite:
    """Advanced test suite for all factor calculations"""
    
    def __init__(self):
        self.price_df = None
        self.fundamental_df = None
        self.registry = FactorRegistry()
        self.results = []
        self.errors = []
        self.quality_issues = []
        
    def test_price_factors(self):
        """Test all price-based technical factors"""
        logger.info("="*80)
        logger.info("TEST 1: PRICE-BASED TECHNICAL FACTORS")
        logger.info("="*80)
        
        # Load price data
        logger.info("\n[LOADING] Price Data...")
        try:
            price_loader = PriceLoader()
            self.price_df = price_loader.get_data()
            
            if self.price_df.empty:
                logger.error("❌ Price data is empty!")
                return False
            
            logger.info(f"✅ Loaded {len(self.price_df):,} price records")
            logger.info(f"   Tickers: {self.price_df['ticker'].nunique()}")
            logger.info(f"   Date Range: {self.price_df['date'].min()} to {self.price_df['date'].max()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load price data: {e}")
            self.errors.append(f"Price Loading: {str(e)}")
            return False
        
        # Get technical factors (exclude fundamental)
        technical_factors = {
            k: v for k, v in self.registry.factors.items() 
            if v.category == 'technical'
        }
        
        logger.info(f"\n[TESTING] {len(technical_factors)} Technical Factors...")
        logger.info("-"*80)
        
        def _test_single_factor(factor_name, factor, data):
            try:
                start = time.time()
                result_df = factor.calculate(data)
                elapsed = time.time() - start
                return {'status': 'success', 'name': factor_name, 'factor': factor, 'df': result_df, 'elapsed': elapsed}
            except Exception as e:
                return {'status': 'error', 'name': factor_name, 'error': str(e)}

        success = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_test_single_factor, k, v, self.price_df): k for k, v in technical_factors.items()}
            
            for future in as_completed(futures):
                res = future.result()
                factor_name = res['name']
                
                if res['status'] == 'success':
                    try:
                        factor = res['factor']
                        result_df = res['df']
                        elapsed = res['elapsed']
                        
                        factor_col = factor.name
                        if factor_col not in result_df.columns:
                            raise ValueError(f"Factor column '{factor_col}' not found")
                        
                        factor_series = result_df[factor_col]
                        stats = self._calculate_stats(factor_series, factor_name, elapsed)
                        self.results.append(stats)
                        success += 1
                        
                        # Quality checks
                        self._check_factor_quality(factor_series, factor_name)
                        
                        logger.info(f"✅ {factor_name:30s} | Mean:{stats['mean']:8.4f} | Std:{stats['std']:8.4f} | Non-Null:{stats['non_null_pct']:6.2f}% | Time:{elapsed:6.2f}s")
                    except Exception as e:
                        logger.error(f"❌ {factor_name:30s} | Error processing results: {str(e)[:60]}")
                        self.errors.append(f"{factor_name}: {str(e)}")
                else:
                    logger.error(f"❌ {factor_name:30s} | Error: {res['error'][:60]}")
                    self.errors.append(f"{factor_name}: {res['error']}")
        
        logger.info("-"*80)
        logger.info(f"\n📊 Technical Factors: {success}/{len(technical_factors)} passed ({success/len(technical_factors)*100:.1f}%)")
        return success == len(technical_factors)
    
    def test_fundamental_factors(self):
        """Test value factors with fundamental data"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: FUNDAMENTAL FACTORS (Value, Quality, Growth, Health)")
        logger.info("="*80)
        
        # Load fundamental data
        logger.info("\n[LOADING] Fundamental Data...")
        try:
            fund_loader = FundamentalLoader()
            self.fundamental_df = fund_loader.get_data()
            
            if self.fundamental_df.empty:
                logger.warning("⚠️  Fundamental data is empty - value factors will be skipped")
                logger.warning("   Ensure data/raw/fundamentals/ contains ticker folders with info.csv")
                return None
            
            logger.info(f"✅ Loaded {len(self.fundamental_df):,} fundamental records")
            logger.info(f"   Columns: {self.fundamental_df.columns.tolist()[:5]}...")
            
        except Exception as e:
            logger.error(f"❌ Failed to load fundamental data: {e}")
            self.errors.append(f"Fundamental Loading: {str(e)}")
            return None
        
        # Get fundamental factors (Value + Quality)
        fund_factors = {
            k: v for k, v in self.registry.factors.items() 
            if v.category == 'fundamental'
        }
        
        if not fund_factors:
            logger.warning("⚠️  No fundamental factors registered")
            return None
        
        logger.info(f"\n[TESTING] {len(fund_factors)} Fundamental Factors...")
        logger.info("-"*80)
        
        def _test_single_fund(factor_name, factor, data):
            try:
                start = time.time()
                result_df = factor.calculate(data)
                elapsed = time.time() - start
                return {'status': 'success', 'name': factor_name, 'factor': factor, 'df': result_df, 'elapsed': elapsed}
            except Exception as e:
                return {'status': 'error', 'name': factor_name, 'error': str(e)}

        success = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_test_single_fund, k, v, self.fundamental_df): k for k, v in fund_factors.items()}
            
            for future in as_completed(futures):
                res = future.result()
                factor_name = res['name']
                
                if res['status'] == 'success':
                    try:
                        factor = res['factor']
                        result_df = res['df']
                        elapsed = res['elapsed']
                        
                        factor_col = factor.name
                        if factor_col not in result_df.columns:
                            raise ValueError(f"Factor column '{factor_col}' not found")
                        
                        factor_series = result_df[factor_col]
                        stats = self._calculate_stats(factor_series, factor_name, elapsed)
                        self.results.append(stats)
                        
                        # Verify Configuration Fix (fill_na=False)
                        if getattr(factor, 'fill_na', True) is True:
                            self.quality_issues.append(f"Config: {factor_name} has fill_na=True (Should be False for Fundamentals)")

                        success += 1
                        
                        # Quality checks
                        self._check_factor_quality(factor_series, factor_name)
                        
                        logger.info(f"✅ {factor_name:30s} | Mean:{stats['mean']:8.4f} | Std:{stats['std']:8.4f} | Non-Null:{stats['non_null_pct']:6.2f}% | Time:{elapsed:6.2f}s")
                    except Exception as e:
                        logger.error(f"❌ {factor_name:30s} | Error processing results: {str(e)[:60]}")
                        self.errors.append(f"{factor_name}: {str(e)}")
                else:
                    logger.error(f"❌ {factor_name:30s} | Error: {res['error'][:60]}")
                    self.errors.append(f"{factor_name}: {res['error']}")
        
        logger.info("-"*80)
        if len(fund_factors) > 0:
            logger.info(f"\n📊 Fundamental Factors: {success}/{len(fund_factors)} passed ({success/len(fund_factors)*100:.1f}%)")
        return success == len(fund_factors) if fund_factors else True
    
    def _calculate_stats(self, series: pd.Series, factor_name: str, elapsed_time: float) -> dict:
        """Calculate comprehensive statistics for a factor"""
        # Helper to snap small numbers to 0.0 (Cosmetic fix for -0.0000)
        def clean_val(val):
            if pd.isna(val) or np.isinf(val): return val
            return 0.0 if abs(val) < 1e-9 else val
            
        return {
            'factor': factor_name,
            'min': clean_val(series.min()),
            'max': clean_val(series.max()),
            'mean': clean_val(series.mean()),
            'std': clean_val(series.std()),
            'median': clean_val(series.median()),
            'non_null': series.notna().sum(),
            'null': series.isna().sum(),
            'non_null_pct': (series.notna().sum() / len(series) * 100),
            'inf_count': np.isinf(series).sum(),
            'has_negative': (series < 0).sum() > 0,
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'time_seconds': elapsed_time
        }
    
    def _check_factor_quality(self, series: pd.Series, factor_name: str):
        """Check for common data quality issues"""
        issues = []
        
        # Check 1: All NaN
        if series.isna().all():
            issues.append(f"{factor_name}: All values are NaN (likely missing input columns)")
        
        # Check 2: All same value
        if series.nunique() == 1:
            issues.append(f"{factor_name}: All values are identical (no variation)")
        
        # Check 3: Too many NaN
        nan_pct = series.isna().sum() / len(series) * 100
        if nan_pct > 50:
            issues.append(f"{factor_name}: {nan_pct:.1f}% NaN values (possible missing data dependency)")
        
        # Check 4: Infinite values
        if np.isinf(series).sum() > 0:
            issues.append(f"{factor_name}: Contains {np.isinf(series).sum()} infinite values (division by zero?)")
        
        # Check 5: Extreme range
        if not series.isna().all():
            range_val = series.max() - series.min()
            if range_val > 1e6:
                issues.append(f"{factor_name}: Extremely large range [{series.min():.2e}, {series.max():.2e}]")
        
        if issues:
            for issue in issues:
                logger.warning(f"   ⚠️  {issue}")
                self.quality_issues.append(issue)
    
    def generate_code_review(self):
        """Generate comprehensive code review with negative points"""
        logger.info("\n" + "="*80)
        logger.info("CODE QUALITY REVIEW - NEGATIVE POINTS")
        logger.info("="*80)
        
        # Dynamic Review based on actual test results
        if self.errors:
            logger.info("🔴 CRITICAL ISSUES DETECTED:")
            for err in self.errors:
                logger.info(f"   - {err}")
        else:
            logger.info("✅ No Critical Runtime Errors Detected.")

        if self.quality_issues:
            logger.info("\n🟡 DATA QUALITY WARNINGS:")
            for issue in set(self.quality_issues):
                logger.info(f"   - {issue}")
        else:
            logger.info("✅ Data Quality Checks Passed.")

        # Performance Check
        slow_factors = [r for r in self.results if r['time_seconds'] > 1.0]
        if slow_factors:
            logger.info("\n🟠 PERFORMANCE WARNINGS (Slow Factors > 1.0s):")
            for f in slow_factors:
                logger.info(f"   - {f['factor']}: {f['time_seconds']:.2f}s")
        
        # Static Analysis
        static_issues = self._run_static_analysis()
        if static_issues:
            logger.info("\n🔍 STATIC ANALYSIS FINDINGS:")
            for issue in static_issues:
                logger.info(f"   - {issue}")
        
        # Recommendations
        logger.info(f"\n{'='*80}")
        logger.info("TOP 5 RECOMMENDATIONS FOR IMPROVEMENT:")
        logger.info(f"{'='*80}")
        logger.info("""
1. Integration Testing: Update pipeline to use 'FactorRegistry.compute_all()' to verify parallel execution.
2. Unit Testing: Add granular pytest cases for edge cases (empty data, single row).
3. Config Management: Move 'COLUMN_MAPPINGS' from value.py to a central YAML config.
4. Advanced Caching: Implement caching for shared rolling window calculations.
5. CI/CD: Integrate this test suite into automated build pipelines.
        """)

    def _run_static_analysis(self):
        """Check for known code patterns"""
        issues = []
        
        # Verify EPS is in base (Positive Check)
        if not hasattr(quant_alpha.features.base, 'EPS'):
            issues.append("Design: 'EPS' constant missing from base.py (Centralization required).")
            
        # Verify EPS is imported in value.py, not redefined
        import quant_alpha.features.fundamental.value as val_module
        if 'EPS' in val_module.__dict__:
            if val_module.EPS is not quant_alpha.features.base.EPS:
                 issues.append("Design: 'EPS' is redefined in value.py (Should be imported from base).")
        
        # Check for hardcoded column mappings (Source Inspection)
        try:
            import inspect
            if hasattr(val_module, 'ColumnValidator'):
                src = inspect.getsource(val_module.ColumnValidator)
                if "COLUMN_MAPPINGS = {" in src:
                     issues.append("Design: Hardcoded 'COLUMN_MAPPINGS' detected in value.py. Move to config.")
        except Exception:
            pass
            
        return issues
    
    def generate_summary_report(self):
        """Generate final test report"""
        logger.info("\n" + "="*80)
        logger.info("FINAL TEST SUMMARY REPORT")
        logger.info("="*80)
        
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            logger.info(f"\n📊 COMPUTATION STATISTICS:")
            logger.info(f"   Total Factors Tested: {len(results_df)}")
            logger.info(f"   All Non-Null: {len(results_df[results_df['non_null_pct'] == 100.0])}")
            logger.info(f"   Partial Data: {len(results_df[(results_df['non_null_pct'] > 0) & (results_df['non_null_pct'] < 100)])}")
            
            all_nan_df = results_df[results_df['non_null_pct'] == 0]
            logger.info(f"   All NaN: {len(all_nan_df)}")
            if not all_nan_df.empty:
                logger.info("\n   🔴 FAILED FACTORS (100% NaN) - CHECK MAPPINGS:")
                for name in all_nan_df['factor'].tolist():
                    logger.info(f"      - {name}")
            
            logger.info(f"   Average Computation Time: {results_df['time_seconds'].mean():.4f}s")
            logger.info(f"   Total Time: {results_df['time_seconds'].sum():.2f}s")
            
            logger.info(f"\n⚠️  DATA QUALITY METRICS:")
            logger.info(f"   Factors with Infinite Values: {len(results_df[results_df['inf_count'] > 0])}")
            logger.info(f"   Factors with High Skewness: {len(results_df[results_df['skewness'].abs() > 2])}")
            logger.info(f"   Factors with High Kurtosis: {len(results_df[results_df['kurtosis'].abs() > 5])}")
        
        if self.errors:
            logger.info(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:10], 1):
                logger.info(f"   {i}. {error[:70]}")
            if len(self.errors) > 10:
                logger.info(f"   ... and {len(self.errors)-10} more")
        
        if self.quality_issues:
            logger.info(f"\n⚠️  QUALITY ISSUES ({len(self.quality_issues)}):")
            unique_issues = list(set(self.quality_issues))
            for i, issue in enumerate(unique_issues[:5], 1):
                logger.info(f"   {i}. {issue[:70]}")
            if len(unique_issues) > 5:
                logger.info(f"   ... and {len(unique_issues)-5} more")
        
        logger.info(f"\n{'='*80}")
        success_rate = 100 - (len(self.errors) / max(1, len(self.registry.factors)) * 100)
        if success_rate >= 80:
            logger.info(f"✅ OVERALL STATUS: PASSED ({success_rate:.1f}%)")
        elif success_rate >= 50:
            logger.info(f"⚠️  OVERALL STATUS: PARTIAL ({success_rate:.1f}%)")
        else:
            logger.info(f"❌ OVERALL STATUS: FAILED ({success_rate:.1f}%)")
        logger.info(f"{'='*80}\n")


def run_all_tests():
    """Main test execution"""
    suite = FactorTestSuite()
    
    # Run tests
    test1_passed = suite.test_price_factors()
    test2_passed = suite.test_fundamental_factors()
    
    # Generate reports
    suite.generate_summary_report()
    suite.generate_code_review()
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Test suite crashed: {e}")
        exit(1)

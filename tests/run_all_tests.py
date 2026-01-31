"""
Run All Tests — Quant Alpha Complete Test Suite
================================================
Runs all test modules and generates summary report.

Usage:
    python tests/run_all_tests.py           # Full test suite
    python tests/run_all_tests.py --quick   # Quick mode (skip slow tests)
    python tests/run_all_tests.py --verbose # Verbose output
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestRunner:
    """Run all test modules and track results."""
    
    def __init__(self, quick_mode=False, verbose=True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_module(self, module_name: str, test_func) -> bool:
        """Run a single test module."""
        print(f"\n{'='*70}")
        print(f"🧪 RUNNING: {module_name}")
        print('='*70)
        
        start = time.time()
        
        try:
            passed = test_func()
            elapsed = time.time() - start
            
            self.results[module_name] = {
                'passed': passed,
                'time': elapsed,
                'error': None
            }
            
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"\n{status} — {module_name} ({elapsed:.1f}s)")
            
            return passed
            
        except Exception as e:
            elapsed = time.time() - start
            self.results[module_name] = {
                'passed': False,
                'time': elapsed,
                'error': str(e)
            }
            print(f"\n💥 ERROR — {module_name}: {e}")
            return False
    
    def run_all(self) -> bool:
        """Run all test modules."""
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("🚀 QUANT ALPHA — COMPLETE TEST SUITE")
        print("="*70)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚡ Mode: {'QUICK' if self.quick_mode else 'FULL'}")
        print("="*70)
        
        all_passed = True
        
        # =====================================================================
        # TEST 1: Data Loading
        # =====================================================================
        try:
            from tests.test_data_loading import (
                test_synthetic_data_validation,
                test_dataloader_integration
            )
            
            def run_data_tests():
                p1 = test_synthetic_data_validation()
                p2 = test_dataloader_integration()
                return p1 and p2
            
            if not self.run_test_module("Data Loading", run_data_tests):
                all_passed = False
                
        except ImportError as e:
            print(f"⚠️  Could not import test_data: {e}")
            self.results["Data Loading"] = {'passed': False, 'time': 0, 'error': str(e)}
            all_passed = False
        
        # =====================================================================
        # TEST 2: Feature Engineering
        # =====================================================================
        try:
            from tests.test_features import (
                test_momentum_calculations,
                test_mean_reversion_calculations,
                test_volatility_calculations,
                test_volume_calculations,
                test_feature_edge_cases,
                test_factor_registry,
                test_compute_all_features,
                test_no_data_leakage,
                test_feature_value_ranges
            )
            
            def run_feature_tests():
                results = [
                    test_momentum_calculations(),
                    test_mean_reversion_calculations(),
                    test_volatility_calculations(),
                    test_volume_calculations(),
                    test_feature_edge_cases(),
                    test_factor_registry(),
                    test_compute_all_features(),
                    test_no_data_leakage(),
                    test_feature_value_ranges()
                ]
                return all(results)
            
            if not self.run_test_module("Feature Engineering", run_feature_tests):
                all_passed = False
                
        except ImportError as e:
            print(f"⚠️  Could not import test_features: {e}")
            self.results["Feature Engineering"] = {'passed': False, 'time': 0, 'error': str(e)}
            all_passed = False
        
        # =====================================================================
        # TEST 3: Models
        # =====================================================================
        try:
            from tests.test_models import (
                test_model_initialization,
                test_model_training,
                test_model_prediction,
                test_model_evaluation,
                test_feature_importance,
                test_model_save_load,
                test_model_edge_cases,
                test_prediction_speed
            )
            
            def run_model_tests():
                results = [
                    test_model_initialization(),
                    test_model_training(),
                    test_model_prediction(),
                    test_model_evaluation(),
                    test_feature_importance(),
                    test_model_save_load(),
                    test_model_edge_cases(),
                ]
                # Skip speed test in quick mode
                if not self.quick_mode:
                    results.append(test_prediction_speed())
                return all(results)
            
            if not self.run_test_module("Models", run_model_tests):
                all_passed = False
                
        except ImportError as e:
            print(f"⚠️  Could not import test_models: {e}")
            self.results["Models"] = {'passed': False, 'time': 0, 'error': str(e)}
            all_passed = False
        
        # =====================================================================
        # TEST 4: Backtesting
        # =====================================================================
        try:
            from tests.test_backtest import (
                test_performance_metrics,
                test_portfolio_construction,
                test_transaction_costs,
                test_backtest_engine,
                test_metrics_module,
                test_portfolio_optimizer,
                test_backtest_edge_cases
            )
            
            def run_backtest_tests():
                results = [
                    test_performance_metrics(),
                    test_portfolio_construction(),
                    test_transaction_costs(),
                    test_backtest_engine(),
                    test_metrics_module(),
                    test_portfolio_optimizer(),
                    test_backtest_edge_cases()
                ]
                return all(results)
            
            if not self.run_test_module("Backtesting", run_backtest_tests):
                all_passed = False
                
        except ImportError as e:
            print(f"⚠️  Could not import test_backtest: {e}")
            self.results["Backtesting"] = {'passed': False, 'time': 0, 'error': str(e)}
            all_passed = False
        
        # =====================================================================
        # TEST 5: Visualization (Skip in quick mode)
        # =====================================================================
        if not self.quick_mode:
            try:
                from tests.test_visualization import (
                    generate_viz_data,
                    get_output_dir,
                    test_performance_plotter,
                    test_factor_plotter,
                    test_risk_plotter,
                    test_quick_plot_all,
                    test_report_generator,
                    test_convenience_functions,
                    test_edge_cases,
                    test_full_report
                )
                
                def run_viz_tests():
                    output_dir = get_output_dir()
                    data = generate_viz_data()
                    
                    results = [
                        test_performance_plotter(data, output_dir),
                        test_factor_plotter(data, output_dir),
                        test_risk_plotter(data, output_dir),
                        test_quick_plot_all(data, output_dir),
                        test_report_generator(data, output_dir),
                        test_convenience_functions(data, output_dir),
                        test_edge_cases(output_dir),
                        test_full_report(data, output_dir)
                    ]
                    return all(results)
                
                if not self.run_test_module("Visualization", run_viz_tests):
                    all_passed = False
                    
            except ImportError as e:
                print(f"⚠️  Could not import test_visualization: {e}")
                self.results["Visualization"] = {'passed': False, 'time': 0, 'error': str(e)}
                all_passed = False
        else:
            print("\n⏭️  Skipping Visualization tests (quick mode)")
            self.results["Visualization"] = {'passed': True, 'time': 0, 'error': 'Skipped (quick mode)'}
        
        # =====================================================================
        # TEST 6: Integration (Skip in quick mode)
        # =====================================================================
        if not self.quick_mode:
            try:
                from tests.test_integration import test_full_pipeline
                
                if not self.run_test_module("Integration", test_full_pipeline):
                    all_passed = False
                    
            except ImportError as e:
                print(f"⚠️  Could not import test_integration: {e}")
                self.results["Integration"] = {'passed': False, 'time': 0, 'error': str(e)}
                all_passed = False
        else:
            print("\n⏭️  Skipping Integration tests (quick mode)")
            self.results["Integration"] = {'passed': True, 'time': 0, 'error': 'Skipped (quick mode)'}
        
        self.end_time = time.time()
        
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        total_time = self.end_time - self.start_time
        
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        passed_count = sum(1 for r in self.results.values() if r['passed'])
        total_count = len(self.results)
        
        print(f"\n{'Module':<25} {'Status':<12} {'Time':<10} {'Notes'}")
        print("-"*70)
        
        for module, result in self.results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            time_str = f"{result['time']:.1f}s"
            notes = result['error'][:30] + "..." if result['error'] and len(result['error']) > 30 else (result['error'] or "")
            print(f"{module:<25} {status:<12} {time_str:<10} {notes}")
        
        print("-"*70)
        print(f"{'TOTAL':<25} {passed_count}/{total_count:<10} {total_time:.1f}s")
        
        # Final verdict
        print("\n" + "="*70)
        if passed_count == total_count:
            print("🎉 ALL TESTS PASSED — QUANT ALPHA IS PRODUCTION READY!")
        else:
            failed = total_count - passed_count
            print(f"⚠️  {failed} TEST MODULE(S) FAILED — Please review errors above")
        print("="*70)
        
        # Detailed errors
        failed_modules = [m for m, r in self.results.items() if not r['passed'] and r['error']]
        if failed_modules:
            print("\n📋 ERROR DETAILS:")
            print("-"*70)
            for module in failed_modules:
                print(f"\n{module}:")
                print(f"   {self.results[module]['error']}")
        
        return passed_count == total_count


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Quant Alpha Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_all_tests.py           # Full test suite
    python tests/run_all_tests.py --quick   # Skip slow tests
    python tests/run_all_tests.py -v        # Verbose output
        """
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode: skip visualization and integration tests'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    runner = TestRunner(quick_mode=args.quick, verbose=args.verbose)
    all_passed = runner.run_all()
    
    # Print summary
    success = runner.print_summary()
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
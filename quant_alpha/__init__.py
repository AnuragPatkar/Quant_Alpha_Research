"""
Quant Alpha Research Package
============================
ML-Based Multi-Factor Alpha Model for Stock Return Prediction

This package provides a complete framework for:
    - Feature engineering (momentum, mean reversion, microstructure)
    - ML model training with walk-forward validation
    - Backtesting with realistic costs
    - Performance analysis and reporting
    - Research tools (alpha decay, significance testing)

Modules:
    - data: Data loading and preprocessing
    - features: Factor/feature engineering
    - models: ML models and training
    - backtest: Portfolio simulation
    - research: Alpha research tools
    - visualization: Plots and reports

Quick Start:
    >>> from quant_alpha import AlphaResearchPipeline
    >>> pipeline = AlphaResearchPipeline()
    >>> results = pipeline.run()

Author: Anurag Patkar
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Anurag Patkar"
__email__ = "your.email@example.com"
__license__ = "MIT"

# =============================================================================
# IMPORTS - Core Classes
# =============================================================================

# Data
try:
    from .data.loader import DataLoader
except ImportError:
    DataLoader = None

# Features
try:
    from .features.registry import FeatureRegistry
    from .features.base import FeatureBase
    from .features.momentum import MomentumFeatures
    from .features.mean_reversion import MeanReversionFeatures
    from .features.microstructure import MicrostructureFeatures
except ImportError:
    FeatureRegistry = None
    FeatureBase = None
    MomentumFeatures = None
    MeanReversionFeatures = None
    MicrostructureFeatures = None

# Models
try:
    from .models.boosting import LightGBMModel
    from .models.trainer import WalkForwardTrainer
except ImportError:
    LightGBMModel = None
    WalkForwardTrainer = None

# Backtest
try:
    from .backtest.engine import Backtester, BacktestResult, BacktestConfig
    from .backtest.metrics import PerformanceMetrics, RiskMetrics, calculate_metrics
    from .backtest.portfolio import PortfolioAnalyzer
except ImportError:
    Backtester = None
    BacktestResult = None
    BacktestConfig = None
    PerformanceMetrics = None
    RiskMetrics = None
    calculate_metrics = None
    PortfolioAnalyzer = None

# Research
try:
    from .research.analysis import AlphaAnalyzer
    from .research.significance import SignificanceTester
    from .research.regime import RegimeDetector
except ImportError:
    AlphaAnalyzer = None
    SignificanceTester = None
    RegimeDetector = None

# Visualization
try:
    from .visualization.plots import (
        PerformancePlotter,
        FactorPlotter,
        RiskPlotter,
        plot_equity_curve,
        plot_returns_distribution,
        quick_plot_all
    )
    from .visualization.reports import (
        ReportGenerator,
        generate_report,
        print_metrics
    )
except ImportError:
    PerformancePlotter = None
    FactorPlotter = None
    RiskPlotter = None
    plot_equity_curve = None
    plot_returns_distribution = None
    quick_plot_all = None
    ReportGenerator = None
    generate_report = None
    print_metrics = None


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class AlphaResearchPipeline:
    """
    Complete alpha research pipeline.
    
    Orchestrates the full workflow:
        1. Load data
        2. Engineer features
        3. Train model
        4. Generate predictions
        5. Backtest strategy
        6. Generate reports
    
    Example:
        >>> pipeline = AlphaResearchPipeline(
        ...     data_path='data/prices.parquet',
        ...     output_dir='results'
        ... )
        >>> results = pipeline.run()
        >>> pipeline.generate_report()
    
    Attributes:
        config: Pipeline configuration
        data_loader: Data loading component
        feature_registry: Feature engineering component
        model: ML model component
        trainer: Walk-forward trainer
        backtester: Backtesting engine
        results: Latest results dictionary
    """
    
    def __init__(
        self,
        data_path: str = None,
        output_dir: str = "output",
        config: dict = None,
        verbose: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            data_path: Path to price data
            output_dir: Output directory for results
            config: Custom configuration dict
            verbose: Print progress
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.config = config or {}
        self.verbose = verbose
        self.results = None
        
        # Initialize components (lazy loading)
        self._data_loader = None
        self._feature_registry = None
        self._model = None
        self._trainer = None
        self._backtester = None
    
    @property
    def data_loader(self):
        """Lazy load data loader."""
        if self._data_loader is None and DataLoader is not None:
            self._data_loader = DataLoader()
        return self._data_loader
    
    @property
    def feature_registry(self):
        """Lazy load feature registry."""
        if self._feature_registry is None and FeatureRegistry is not None:
            self._feature_registry = FeatureRegistry()
        return self._feature_registry
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None and LightGBMModel is not None:
            self._model = LightGBMModel()
        return self._model
    
    @property
    def trainer(self):
        """Lazy load trainer."""
        if self._trainer is None and WalkForwardTrainer is not None:
            self._trainer = WalkForwardTrainer()
        return self._trainer
    
    @property
    def backtester(self):
        """Lazy load backtester."""
        if self._backtester is None and Backtester is not None:
            self._backtester = Backtester(verbose=self.verbose)
        return self._backtester
    
    def run(
        self,
        start_date: str = None,
        end_date: str = None,
        generate_report: bool = True
    ) -> dict:
        """
        Run complete pipeline.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            generate_report: Auto-generate report
            
        Returns:
            Dictionary with all results
        """
        if self.verbose:
            self._print_header()
        
        results = {}
        
        try:
            # Step 1: Load Data
            if self.verbose:
                print("\n📊 Step 1: Loading data...")
            
            if self.data_loader is not None and self.data_path:
                data = self.data_loader.load(self.data_path)
                results['data'] = data
            else:
                print("   ⚠️ Data loader not available or no path specified")
                return results
            
            # Step 2: Engineer Features
            if self.verbose:
                print("\n🔧 Step 2: Engineering features...")
            
            if self.feature_registry is not None:
                features = self.feature_registry.compute_all(data)
                results['features'] = features
            else:
                print("   ⚠️ Feature registry not available")
            
            # Step 3: Train Model
            if self.verbose:
                print("\n🤖 Step 3: Training model...")
            
            if self.trainer is not None and 'features' in results:
                train_results = self.trainer.train(results['features'])
                results['model_results'] = train_results
                results['predictions'] = train_results.get('predictions')
            else:
                print("   ⚠️ Trainer not available")
            
            # Step 4: Backtest
            if self.verbose:
                print("\n💼 Step 4: Backtesting...")
            
            if self.backtester is not None and 'predictions' in results:
                backtest_result = self.backtester.run(results['predictions'])
                if backtest_result is not None:
                    results['backtest'] = backtest_result.to_dict()
            else:
                print("   ⚠️ Backtester not available")
            
            # Step 5: Generate Report
            if generate_report and 'backtest' in results:
                if self.verbose:
                    print("\n📄 Step 5: Generating report...")
                self.generate_report(results)
            
            self.results = results
            
            if self.verbose:
                self._print_footer(results)
            
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def generate_report(
        self,
        results: dict = None,
        formats: list = ['pdf', 'json', 'png']
    ) -> dict:
        """
        Generate reports from results.
        
        Args:
            results: Results dict (uses self.results if None)
            formats: Report formats to generate
            
        Returns:
            Dictionary mapping format to file path
        """
        results = results or self.results
        
        if results is None:
            print("❌ No results to report. Run pipeline first.")
            return {}
        
        if 'backtest' not in results:
            print("❌ No backtest results to report.")
            return {}
        
        if ReportGenerator is None:
            print("❌ Report generator not available.")
            return {}
        
        generator = ReportGenerator(output_dir=self.output_dir)
        return generator.generate_full_report(
            results['backtest'],
            formats=formats,
            report_name=f"alpha_research_{__version__}"
        )
    
    def _print_header(self):
        """Print pipeline header."""
        print("\n" + "="*60)
        print("  🚀 QUANT ALPHA RESEARCH PIPELINE")
        print("="*60)
        print(f"  Version: {__version__}")
        print(f"  Author: {__author__}")
        print(f"  Output: {self.output_dir}")
        print("="*60)
    
    def _print_footer(self, results: dict):
        """Print pipeline footer."""
        print("\n" + "="*60)
        print("  ✅ PIPELINE COMPLETE")
        print("="*60)
        
        if 'backtest' in results:
            metrics = results['backtest'].get('metrics', {})
            print(f"  Total Return:  {metrics.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown:  {metrics.get('max_drawdown', 0):.2%}")
        
        print(f"\n  📂 Output: {self.output_dir}")
        print("="*60 + "\n")


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Package info
    '__version__',
    '__author__',
    
    # Main pipeline
    'AlphaResearchPipeline',
    
    # Data
    'DataLoader',
    
    # Features
    'FeatureRegistry',
    'FeatureBase',
    'MomentumFeatures',
    'MeanReversionFeatures',
    'MicrostructureFeatures',
    
    # Models
    'LightGBMModel',
    'WalkForwardTrainer',
    
    # Backtest
    'Backtester',
    'BacktestResult',
    'BacktestConfig',
    'PerformanceMetrics',
    'RiskMetrics',
    'PortfolioAnalyzer',
    'calculate_metrics',
    
    # Research
    'AlphaAnalyzer',
    'SignificanceTester',
    'RegimeDetector',
    
    # Visualization
    'PerformancePlotter',
    'FactorPlotter',
    'RiskPlotter',
    'ReportGenerator',
    'plot_equity_curve',
    'plot_returns_distribution',
    'quick_plot_all',
    'generate_report',
    'print_metrics'
]


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def version():
    """Return package version."""
    return __version__


def info():
    """Print package information."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    QUANT ALPHA RESEARCH                      ║
╠══════════════════════════════════════════════════════════════╣
║  Version:     {__version__:<47} ║
║  Author:      {__author__:<47} ║
║  License:     {__license__:<47} ║
╠══════════════════════════════════════════════════════════════╣
║  MODULES                                                     ║
║  ────────────────────────────────────────────────────────    ║
║  • data          - Data loading and preprocessing            ║
║  • features      - Factor/feature engineering                ║
║  • models        - ML models and training                    ║
║  • backtest      - Portfolio simulation                      ║
║  • research      - Alpha research tools                      ║
║  • visualization - Plots and reports                         ║
╠══════════════════════════════════════════════════════════════╣
║  QUICK START                                                 ║
║  ────────────────────────────────────────────────────────    ║
║  >>> from quant_alpha import AlphaResearchPipeline           ║
║  >>> pipeline = AlphaResearchPipeline('data/prices.parquet') ║
║  >>> results = pipeline.run()                                ║
╚══════════════════════════════════════════════════════════════╝
""")


def check_installation():
    """Check if all components are properly installed."""
    components = {
        'DataLoader': DataLoader,
        'FeatureRegistry': FeatureRegistry,
        'LightGBMModel': LightGBMModel,
        'WalkForwardTrainer': WalkForwardTrainer,
        'Backtester': Backtester,
        'PerformanceMetrics': PerformanceMetrics,
        'PerformancePlotter': PerformancePlotter,
        'ReportGenerator': ReportGenerator
    }
    
    print("\n📦 QUANT ALPHA - Installation Check")
    print("="*50)
    
    all_ok = True
    for name, component in components.items():
        status = "✅" if component is not None else "❌"
        if component is None:
            all_ok = False
        print(f"  {status} {name}")
    
    print("="*50)
    
    if all_ok:
        print("  ✅ All components installed correctly!")
    else:
        print("  ⚠️ Some components missing. Check imports.")
    
    print()
    return all_ok


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    info()
    check_installation()
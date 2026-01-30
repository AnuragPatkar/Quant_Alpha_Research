"""
Utility Functions
=================
Common utility functions for ML Alpha Research scripts.

Features:
    - File I/O utilities
    - Pipeline status tracking
    - Performance formatting
    - Timing utilities
    - Validation helpers
    - Dashboard generation

Author: Anurag Patkar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import time
import pickle
import functools
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

# Add project root
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import config
try:
    from config import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(
    name: str = "quant_alpha",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file name
        log_dir: Optional log directory
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%H:%M:%S'))
    logger.addHandler(console)
    
    # File handler
    if log_file:
        if log_dir:
            log_path = Path(log_dir) / log_file
        else:
            log_path = ROOT / "logs" / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# TIMING
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"⏱️  {self.name} started...")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"✅ {self.name} completed in {self.format_time(self.elapsed)}")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to human readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {seconds%60:.0f}s"
        else:
            return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def timeit(func: Callable) -> Callable:
    """Decorator for timing functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️  {func.__name__} took {Timer.format_time(time.time() - start)}")
        return result
    return wrapper


class ProgressTracker:
    """Simple progress tracker for loops."""
    
    def __init__(self, total: int, prefix: str = "Progress", bar_length: int = 30):
        self.total = total
        self.prefix = prefix
        self.bar_length = bar_length
        self.start_time = time.time()
    
    def update(self, current: int, suffix: str = ""):
        pct = current / self.total
        filled = int(self.bar_length * pct)
        bar = "█" * filled + "░" * (self.bar_length - filled)
        
        elapsed = time.time() - self.start_time
        eta = (elapsed / current) * (self.total - current) if current > 0 else 0
        
        status = f"\r{self.prefix}: [{bar}] {pct:.1%} ({current}/{self.total})"
        if suffix:
            status += f" | {suffix}"
        status += f" | ETA: {Timer.format_time(eta)}"
        
        print(status, end="", flush=True)
        if current == self.total:
            print()


# =============================================================================
# FILE I/O
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_file_exists(file_path: Path, description: str = "File") -> bool:
    """Check if a file exists and print status."""
    file_path = Path(file_path)
    if file_path.exists():
        print(f"✅ {description} found: {file_path.name}")
        return True
    else:
        print(f"❌ {description} missing: {file_path}")
        return False


def load_json_safely(file_path: Path) -> Optional[Dict]:
    """Safely load JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load {file_path.name}: {e}")
        return None


def save_json_safely(data: Dict, file_path: Path, description: str = "Data") -> bool:
    """Safely save data to JSON file."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=_json_serializer)
        print(f"💾 {description} saved: {file_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to save {description}: {e}")
        return False


def _json_serializer(obj):
    """JSON serializer for complex types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_results(
    data: Union[pd.DataFrame, pd.Series, Dict],
    path: Union[str, Path],
    verbose: bool = True
) -> Path:
    """Save results to file (auto-detect format)."""
    path = Path(path)
    ensure_dir(path.parent)
    
    suffix = path.suffix.lower()
    
    try:
        if isinstance(data, pd.DataFrame):
            if suffix == '.parquet':
                data.to_parquet(path)
            elif suffix == '.pkl':
                data.to_pickle(path)
            else:
                data.to_csv(path)
        elif isinstance(data, pd.Series):
            data.to_frame().to_csv(path)
        elif isinstance(data, dict):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=_json_serializer)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
        if verbose:
            print(f"💾 Saved: {path}")
        return path
        
    except Exception as e:
        print(f"❌ Failed to save {path}: {e}")
        raise


def load_results(path: Union[str, Path], verbose: bool = True) -> Any:
    """Load results from file (auto-detect format)."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix = path.suffix.lower()
    
    try:
        if suffix == '.csv':
            data = pd.read_csv(path, index_col=0, parse_dates=True)
        elif suffix == '.parquet':
            data = pd.read_parquet(path)
        elif suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif suffix in ['.pkl', '.pickle']:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = pd.read_csv(path)
        
        if verbose:
            print(f"📂 Loaded: {path}")
        return data
        
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        raise


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataframe(df: pd.DataFrame, required_cols: List[str], name: str = "DataFrame") -> bool:
    """Validate DataFrame has required columns."""
    if df is None or len(df) == 0:
        print(f"❌ {name} is empty")
        return False
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ {name} missing columns: {missing}")
        return False
    
    return True


def validate_predictions(df: pd.DataFrame) -> bool:
    """Validate predictions DataFrame."""
    # Try multiple column name variants
    pred_cols = ['prediction', 'predictions', 'pred']
    has_pred = any(col in df.columns for col in pred_cols)
    
    required = ['date', 'ticker', 'forward_return']
    if not validate_dataframe(df, required, "Predictions"):
        return False
    
    if not has_pred:
        print(f"❌ Missing prediction column (tried: {pred_cols})")
        return False
    
    return True


# =============================================================================
# PRINTING
# =============================================================================

def print_header(title: str, width: int = 60, char: str = "="):
    """Print formatted header."""
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_section(title: str, width: int = 50, char: str = "-"):
    """Print section header."""
    print(f"\n{title}")
    print(char * width)


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print formatted metrics."""
    print_section(title)
    for key, value in metrics.items():
        if isinstance(value, float):
            if any(x in key.lower() for x in ['return', 'drawdown', 'rate', 'pct']):
                print(f"  {key:.<30} {value:>10.2%}")
            elif 'ratio' in key.lower():
                print(f"  {key:.<30} {value:>10.2f}")
            else:
                print(f"  {key:.<30} {value:>10.4f}")
        else:
            print(f"  {key:.<30} {str(value):>10}")


def format_performance_metrics(metrics: Dict[str, float]) -> str:
    """Format performance metrics as string."""
    parts = []
    for key, value in metrics.items():
        if 'ic' in key.lower():
            parts.append(f"{key}: {value:.4f}")
        elif any(x in key.lower() for x in ['rate', 'return']):
            parts.append(f"{key}: {value:.2%}")
        else:
            parts.append(f"{key}: {value:.3f}")
    return " | ".join(parts)


# =============================================================================
# PIPELINE STATUS
# =============================================================================

def get_pipeline_status() -> Dict[str, bool]:
    """Check status of all pipeline components."""
    if not SETTINGS_AVAILABLE:
        print("⚠️ Settings not available")
        return {}
    
    status = {}
    
    # Data files
    status['raw_data'] = settings.data.panel_path.exists() if hasattr(settings.data, 'panel_path') else False
    status['features_data'] = (settings.data.processed_dir / "features_dataset.pkl").exists()
    
    # Results files
    status['validation_results'] = (settings.results_dir / "validation_results.csv").exists()
    status['feature_importance'] = (settings.results_dir / "feature_importance.csv").exists()
    status['backtest_results'] = (settings.results_dir / "backtest_results.csv").exists()
    status['analysis_report'] = (settings.results_dir / "analysis_report.txt").exists()
    
    return status


def print_pipeline_status():
    """Print comprehensive pipeline status."""
    print_header("ML ALPHA PIPELINE STATUS")
    
    status = get_pipeline_status()
    
    if not status:
        print("⚠️ Could not determine status")
        return
    
    completed = sum(status.values())
    total = len(status)
    
    print(f"📈 Progress: {completed}/{total} components ready ({completed/total*100:.0f}%)")
    
    print_section("Component Status")
    for component, is_ready in status.items():
        icon = "✅" if is_ready else "❌"
        print(f"   {icon} {component.replace('_', ' ').title()}")
    
    print_section("Next Steps")
    if not status.get('raw_data'):
        print("   1. Add data to data/processed/ folder")
    elif not status.get('features_data'):
        print("   1. Run: python scripts/run_research.py")
    elif not status.get('validation_results'):
        print("   1. Run: python scripts/run_research.py")
    elif not status.get('backtest_results'):
        print("   1. Run: python scripts/run_backtest.py")
    elif not status.get('analysis_report'):
        print("   1. Run: python scripts/run_analysis.py")
    else:
        print("   ✅ All components ready!")
        print("   🎯 Optional: python scripts/run_optimization.py")


def create_summary_dashboard():
    """Create a summary dashboard of all results."""
    print("\n" + "🎯 " * 20)
    print("ML ALPHA MODEL - SUMMARY DASHBOARD")
    print("🎯 " * 20)
    
    print_pipeline_status()
    
    if not SETTINGS_AVAILABLE:
        return
    
    try:
        # Validation results
        val_path = settings.results_dir / "validation_results.csv"
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            print_section("Model Performance")
            if 'test_ic' in val_df.columns:
                print(f"   Average IC: {val_df['test_ic'].mean():.4f}")
            if 'test_hit_rate' in val_df.columns:
                print(f"   Hit Rate: {val_df['test_hit_rate'].mean():.2%}")
            print(f"   Validation Folds: {len(val_df)}")
        
        # Feature importance
        imp_path = settings.results_dir / "feature_importance.csv"
        if imp_path.exists():
            imp_df = pd.read_csv(imp_path)
            print_section("Feature Analysis")
            print(f"   Total Features: {len(imp_df)}")
            if len(imp_df) > 0:
                print(f"   Top Feature: {imp_df.iloc[0]['feature']}")
        
        # Backtest metrics
        bt_path = settings.results_dir / "backtest_metrics.json"
        if bt_path.exists():
            bt_metrics = load_json_safely(bt_path)
            if bt_metrics:
                print_section("Backtest Performance")
                if 'sharpe_ratio' in bt_metrics:
                    print(f"   Sharpe Ratio: {bt_metrics['sharpe_ratio']:.3f}")
                if 'total_return' in bt_metrics:
                    print(f"   Total Return: {bt_metrics['total_return']:.2%}")
                if 'max_drawdown' in bt_metrics:
                    print(f"   Max Drawdown: {bt_metrics['max_drawdown']:.2%}")
    
    except Exception as e:
        print(f"⚠️ Could not load some metrics: {e}")
    
    print("\n" + "🎯 " * 20)


# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    packages = ['pandas', 'numpy', 'lightgbm', 'matplotlib', 'seaborn', 'scipy', 'rich', 'optuna', 'openpyxl']
    
    status = {}
    for package in packages:
        try:
            __import__(package)
            status[package] = True
        except ImportError:
            status[package] = False
    
    return status


def print_dependency_status():
    """Print dependency status."""
    deps = check_dependencies()
    
    print_header("Dependencies")
    
    required = ['pandas', 'numpy', 'lightgbm', 'scipy']
    optional = ['matplotlib', 'seaborn', 'rich', 'optuna', 'openpyxl']
    
    print("\n  Required:")
    for pkg in required:
        print(f"    {'✅' if deps.get(pkg) else '❌'} {pkg}")
    
    print("\n  Optional:")
    for pkg in optional:
        print(f"    {'✅' if deps.get(pkg) else '⚪'} {pkg}")
    
    missing = [p for p in required if not deps.get(p)]
    if missing:
        print(f"\n  ⚠️ Missing required: pip install {' '.join(missing)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Utility Functions')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--dashboard', action='store_true', help='Show summary dashboard')
    parser.add_argument('--deps', action='store_true', help='Check dependencies')
    
    args = parser.parse_args()
    
    if args.dashboard:
        create_summary_dashboard()
    elif args.status:
        print_pipeline_status()
    elif args.deps:
        print_dependency_status()
    else:
        print("Available utilities:")
        print("  --status     Show pipeline status")
        print("  --dashboard  Show summary dashboard")
        print("  --deps       Check dependencies")
"""
Utility Functions
=================
Common utility functions for ML Alpha Research scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import settings


def check_file_exists(file_path: Path, description: str = "File") -> bool:
    """
    Check if a file exists and print status.
    
    Args:
        file_path: Path to check
        description: Description for user-friendly message
        
    Returns:
        True if file exists, False otherwise
    """
    if file_path.exists():
        print(f"✅ {description} found: {file_path.name}")
        return True
    else:
        print(f"❌ {description} missing: {file_path}")
        return False


def load_json_safely(file_path: Path) -> Optional[Dict]:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary if successful, None if failed
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load {file_path.name}: {e}")
        return None


def save_json_safely(data: Dict, file_path: Path, description: str = "Data") -> bool:
    """
    Safely save data to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save to
        description: Description for user-friendly message
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 {description} saved: {file_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to save {description}: {e}")
        return False


def get_pipeline_status() -> Dict[str, bool]:
    """
    Check status of all pipeline components.
    
    Returns:
        Dictionary with component status
    """
    status = {}
    
    # Data files
    status['raw_data'] = check_file_exists(
        settings.data.panel_path, "Raw data"
    )
    
    status['features_data'] = check_file_exists(
        settings.data.processed_dir / "features_dataset.pkl", "Features dataset"
    )
    
    # Results files
    status['validation_results'] = check_file_exists(
        settings.results_dir / "validation_results.csv", "Validation results"
    )
    
    status['feature_importance'] = check_file_exists(
        settings.results_dir / "feature_importance.csv", "Feature importance"
    )
    
    status['backtest_results'] = check_file_exists(
        settings.results_dir / "backtest_results.csv", "Backtest results"
    )
    
    status['analysis_report'] = check_file_exists(
        settings.results_dir / "analysis_report.txt", "Analysis report"
    )
    
    return status


def print_pipeline_status():
    """Print comprehensive pipeline status."""
    print("\n" + "="*60)
    print("📊 ML ALPHA PIPELINE STATUS")
    print("="*60)
    
    status = get_pipeline_status()
    
    # Count completed steps
    completed = sum(status.values())
    total = len(status)
    
    print(f"📈 Progress: {completed}/{total} components ready")
    print(f"📊 Completion: {completed/total*100:.0f}%")
    
    # Detailed status
    print(f"\n📋 Component Status:")
    for component, is_ready in status.items():
        icon = "✅" if is_ready else "❌"
        print(f"   {icon} {component.replace('_', ' ').title()}")
    
    # Next steps
    print(f"\n💡 Next Steps:")
    if not status['raw_data']:
        print("   1. Load data: Check data/processed/ folder")
    elif not status['features_data']:
        print("   1. Run: python scripts/run_research.py")
    elif not status['validation_results']:
        print("   1. Run: python scripts/run_research.py")
    elif not status['backtest_results']:
        print("   1. Run: python scripts/run_backtest.py")
    elif not status['analysis_report']:
        print("   1. Run: python scripts/run_analysis.py")
    else:
        print("   ✅ All components ready!")
        print("   🎯 Optional: python scripts/run_optimization.py")
    
    print("="*60)


def format_performance_metrics(metrics: Dict[str, float]) -> str:
    """
    Format performance metrics for display.
    
    Args:
        metrics: Dictionary of metric name -> value
        
    Returns:
        Formatted string
    """
    formatted = []
    
    for metric, value in metrics.items():
        if 'ic' in metric.lower():
            formatted.append(f"{metric}: {value:.4f}")
        elif 'rate' in metric.lower() or 'ratio' in metric.lower():
            formatted.append(f"{metric}: {value:.2%}")
        elif 'return' in metric.lower():
            formatted.append(f"{metric}: {value:.2%}")
        else:
            formatted.append(f"{metric}: {value:.3f}")
    
    return " | ".join(formatted)


def create_summary_dashboard():
    """Create a summary dashboard of all results."""
    print("\n" + "🎯 "*25)
    print("ML ALPHA MODEL - SUMMARY DASHBOARD")
    print("🎯 "*25)
    
    # Pipeline status
    print_pipeline_status()
    
    # Load and display key metrics
    try:
        # Validation results
        validation_path = settings.results_dir / "validation_results.csv"
        if validation_path.exists():
            validation_df = pd.read_csv(validation_path)
            avg_ic = validation_df['test_ic'].mean()
            print(f"\n📈 Model Performance:")
            print(f"   Average IC: {avg_ic:.4f}")
            print(f"   Hit Rate: {validation_df['test_hit_rate'].mean():.2%}")
            print(f"   Validation Folds: {len(validation_df)}")
        
        # Feature importance
        importance_path = settings.results_dir / "feature_importance.csv"
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            print(f"\n🔧 Feature Analysis:")
            print(f"   Total Features: {len(importance_df)}")
            print(f"   Top Feature: {importance_df.iloc[0]['feature']}")
        
        # Backtest results
        backtest_path = settings.results_dir / "backtest_metrics.json"
        if backtest_path.exists():
            backtest_metrics = load_json_safely(backtest_path)
            if backtest_metrics:
                print(f"\n💼 Backtest Performance:")
                if 'sharpe_ratio' in backtest_metrics:
                    print(f"   Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.3f}")
                if 'total_return' in backtest_metrics:
                    print(f"   Total Return: {backtest_metrics['total_return']:.2%}")
    
    except Exception as e:
        print(f"⚠️ Could not load some metrics: {e}")
    
    print("\n" + "🎯 "*25)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Utility Functions')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--dashboard', action='store_true', help='Show summary dashboard')
    
    args = parser.parse_args()
    
    if args.dashboard:
        create_summary_dashboard()
    elif args.status:
        print_pipeline_status()
    else:
        print("Available utilities:")
        print("  --status     Show pipeline status")
        print("  --dashboard  Show summary dashboard")
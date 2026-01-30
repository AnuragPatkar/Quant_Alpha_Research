"""
Scripts Module
==============
Executable scripts for ML Alpha Research.

Scripts:
    - run_research.py: Complete ML research pipeline
    - run_backtest.py: Portfolio backtesting simulation
    - run_analysis.py: Performance analysis and reporting
    - run_optimization.py: Hyperparameter optimization
    - utils.py: Shared utility functions

Author: Anurag Patkar
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Anurag Patkar"

SCRIPTS = {
    'run_research.py': 'Complete ML research pipeline (Data → Features → Model → Results)',
    'run_backtest.py': 'Portfolio backtesting simulation with realistic costs',
    'run_analysis.py': 'Comprehensive performance analysis and reporting',
    'run_optimization.py': 'Hyperparameter optimization (Grid/Random/Optuna)',
    'utils.py': 'Shared utility functions for all scripts'
}


def list_scripts():
    """List all available scripts."""
    print("\n📜 Available Scripts:")
    print("=" * 70)
    for script, description in SCRIPTS.items():
        print(f"  • {script:25s} - {description}")
    print("=" * 70)
    print("\nUsage: python scripts/<script_name> [options]")
    print("Help:  python scripts/<script_name> --help\n")


if __name__ == "__main__":
    list_scripts()
"""
main.py
=======
Central Command-Line Interface (CLI) for the Quant Alpha Research Platform.

Purpose
-------
Serves as the unified entry point and orchestration layer for the quantitative research
lifecycle. It abstracts complex underlying scripts into a consistent command structure,
managing the transition from Data Ingestion $\rightarrow$ Alpha Research $\rightarrow$
Production Deployment.

Usage:
-------
    $ python main.py <command> [options]

Importance
----------
1.  **Operational Discipline**: Enforces a standardized interface for all research activities,
    reducing "fat-finger" errors in production.
2.  **Process Isolation**: Dispatches heavy computational tasks (e.g., Model Training,
    Backtesting) to separate subprocesses. This ensures a clean memory heap for each stage,
    preventing memory fragmentation and side-effects from global state (e.g., Numba JIT
    compilation artifacts).
3.  **Environment Control**: Pre-configures critical environment variables for BLAS/LAPACK
    threading (e.g., `OMP_NUM_THREADS`) to prevent CPU oversubscription during
    matrix operations.

Tools & Frameworks
------------------
*   **`argparse`**: Implements the Subcommand Pattern for robust CLI argument parsing.
*   **`subprocess`**: Manages child processes for script execution, ensuring GIL release
    and memory isolation.
*   **`logging`**: Provides centralized telemetry and error reporting.
*   **`os` / `sys`**: Handles low-level environment configuration for numerical libraries
    (NumPy, SciPy, Numba).

Commands:
---------
    pipeline    : Orchestrates the full lifecycle (Data -> Train -> Backtest).
    data        : Ingests and normalizes market data (Prices, Fundamentals).
    validate    : Performs statistical validation (IC, Turnover) on alpha factors.
    train       : Executes Walk-Forward Cross-Validation for ML models.
    predict     : Generates alpha signals (Inference) using production models.
    monitor     : Checks for concept drift (PSI) and signal staleness.
    backtest    : Simulates strategy performance (Vectorized & Event-Driven).
    optimize    : Solves for optimal portfolio weights (Mean-Variance, Risk Parity).
    report      : Compiles executive summaries and performance metrics.
    hyperopt    : Runs Bayesian Optimization for hyperparameter tuning.
    deploy      : Manages model artifacts (Archival, Pruning, Health Checks).
    test        : Executes the automated test suite (Unit & Integration).
    clean       : Purges temporary artifacts and caches.

Examples:
---------
    # Run the entire pipeline: data -> validate -> train -> deploy -> backtest -> report
    $ python main.py pipeline --all

    # Train models using parallel processing (for multi-core machines)
    $ python main.py train --parallel-models

    # Generate new alpha signals for the most recent day of data
    $ python main.py predict --last-day-only

    # Generate a target portfolio of 50 stocks using risk parity optimization
    $ python main.py optimize --method risk_parity --top-n 50

    # Check the health of currently deployed models and the signal cache
    $ python main.py deploy --action check

    # Run production monitoring for signal drift (Population Stability Index)
    $ python main.py monitor

    # Launch a hyperparameter search for the models
    $ python main.py hyperopt

    # Run the full unit and integration test suite
    $ python main.py test
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
from pathlib import Path

# --- 1. Environment Configuration (Critical Initialization) ---
# Configure linear algebra backends (MKL, OpenBLAS) prior to importing numerical libraries.
# This prevents thread oversubscription (Context Switching Thrashing) when multiple
# parallel processes are spawned.
#
# Default: 4 threads.
# Rationale: Limits CPU contention in multi-process environments (e.g., concurrent backtests).
# Specific scripts (e.g., `train_models.py`) may override this for dedicated HPC tasks.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["NUMBA_CACHE_DIR"] = ".numba_cache"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config

# --- 2. Command Handlers ---

def _run_script(script_name: str, args: list[str]):
    """
    Executes a target script within an isolated subprocess.

    Architectural Note:
        Using `subprocess.run` guarantees a pristine memory heap for each task.
        This prevents:
        1.  Global state pollution (e.g., Singleton configurations, Numba JIT caches).
        2.  Memory fragmentation from long-running processes (e.g., Training).
        3.  GIL (Global Interpreter Lock) contention between disparate modules.
    """
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        logging.error(f"Script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)] + args
    logging.info(f"🚀 Running: {script_name} {' '.join(args)}")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"✅ Finished: {script_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed: {script_name} (Exit Code: {e.returncode})")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logging.warning("\n🛑 Interrupted by user.")
        sys.exit(130)

def handle_pipeline(args):
    """
    Orchestrates the full end-to-end research pipeline.
    
    Mapping Logic:
        Translates high-level CLI flags (e.g., `--all`) into granular arguments
        required by the underlying `run_pipeline.py` script.
    """
    script_args = []
    # run_pipeline.py does not accept --all; map it to specific steps
    if args.all:
        script_args.extend(["--validate", "--train", "--deploy", "--backtest"])
    # run_pipeline.py uses --full-rebuild, not --force-rebuild
    if args.force_rebuild: script_args.append("--full-rebuild")
    if args.parallel_models: script_args.append("--parallel-models")
    if args.skip_data: script_args.append("--skip-data")
    
    _run_script("run_pipeline.py", script_args)

def handle_data(args):
    """Dispatches the Market Data Ingestion process."""
    _run_script("update_data.py", [])

def handle_validate(args):
    """Dispatches the Factor Validation (IC/Turnover analysis) process."""
    _run_script("validate_factors.py", [])

def handle_train(args):
    """Dispatches the Walk-Forward Model Training process."""
    script_args = []
    if args.force_rebuild: script_args.append("--force-rebuild")
    if args.parallel_models: script_args.append("--parallel-models")
    if args.all: script_args.append("--all")

    _run_script("train_models.py", script_args)

def handle_predict(args):
    """Dispatches the Alpha Signal Generation (Inference) process."""
    script_args = []
    if args.last_day_only: script_args.append("--last-day-only")

    _run_script("generate_predictions.py", script_args)

def handle_monitor(args):
    """Dispatches the Production Health Monitoring process (Drift/Staleness)."""
    script_args = []
    if args.psi_threshold: script_args.extend(["--psi-threshold", str(args.psi_threshold)])

    _run_script("monitor_production.py", script_args)

def handle_backtest(args):
    """Dispatches the Historical Simulation Engine."""
    script_args = []
    if args.method: script_args.extend(["--method", args.method])
    if args.top_n: script_args.extend(["--top-n", str(args.top_n)])

    _run_script("run_backtest.py", script_args)

def handle_optimize(args):
    """Dispatches the Portfolio Construction & Optimization Engine."""
    script_args = []
    if args.capital: script_args.extend(["--capital", str(args.capital)])
    if args.method: script_args.extend(["--method", args.method])
    if args.target_vol: script_args.extend(["--target-vol", str(args.target_vol)])
    if args.top_n: script_args.extend(["--top-n", str(args.top_n)])

    _run_script("optimize_portfolio.py", script_args)

def handle_report(args):
    """Dispatches the Executive Reporting module."""
    script_args = []
    if args.output: script_args.extend(["--output-file", args.output])

    _run_script("create_report.py", script_args)

def handle_hyperopt(args):
    """Dispatches the Bayesian Hyperparameter Optimization process."""
    _run_script("run_hyperopt.py", [])

def handle_deploy(args):
    """Dispatches the Model Lifecycle Manager (Archive/Prune)."""
    script_args = []
    # The --all flag takes precedence over a specific action.
    if args.all:
        script_args.append("--all")
    else:
        # The parser for this command sets a default of 'check'.
        script_args.extend(["--action", args.action])

    # These options are relevant for 'prune' and 'all' actions.
    if args.keep: script_args.extend(["--keep", str(args.keep)])
    if args.dry_run: script_args.append("--dry-run")

    _run_script("deploy_model.py", script_args)

def handle_test(args):
    """
    Executes the automated test suite via Pytest.
    
    Scope:
        - Unit Tests: Verify individual component logic (e.g., Factor calculations).
        - Integration Tests: Validate cross-module workflows (e.g., Data -> Model -> Signal).
    """
    logger = logging.getLogger("TestRunner")
    logger.info("🚀 Running Tests (pytest)...")
    try:
        cmd = [sys.executable, "-m", "pytest", "tests"]
        if args.verbose:
            cmd.append("-v")
        subprocess.run(cmd, check=True)
        logger.info("✅ Tests passed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Tests failed (Exit Code: {e.returncode})")
        sys.exit(e.returncode)

def handle_clean(args):
    """Purges temporary artifacts, caches, and logs to reset the environment."""
    logger = logging.getLogger("Cleaner")
    logger.info("🧹 Cleaning cache and temporary files...")

    targets = [
        PROJECT_ROOT / ".numba_cache",
        PROJECT_ROOT / "results" / "logs",
    ]

    if args.all:
        targets.append(config.CACHE_DIR)
        logger.warning(f"⚠️  Also cleaning data cache: {config.CACHE_DIR}")
    
    for t in targets:
        if t.exists():
            try:
                if t.is_dir():
                    shutil.rmtree(t)
                else:
                    t.unlink()
                logger.info(f"   Removed: {t}")
            except Exception as e:
                logger.error(f"   Failed to remove {t}: {e}")
        else:
            logger.info(f"   Skipped (not found): {t}")

    logger.info("✅ Clean complete.")

# --- 3. Main Entry Point ---

def setup_logging():
    """
    Initializes the logging subsystem with minimal overhead.
    
    Optimization:
        Configures logging locally to avoid importing heavy numerical dependencies
        (Pandas, NumPy) located in `quant_alpha.utils`. This ensures the CLI
        starts in $O(1)$ time relative to the project size.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    parser = argparse.ArgumentParser(prog="main.py", description="Quant Alpha Research Platform CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full end-to-end pipeline")
    p_pipe.add_argument("--all", action="store_true", help="Run all steps including backtest/report")
    p_pipe.add_argument("--force-rebuild", action="store_true", help="Force full data rebuild")
    p_pipe.add_argument("--parallel-models", action="store_true", help="Train models in parallel")
    p_pipe.add_argument("--skip-data", action="store_true", help="Skip data update step")
    p_pipe.set_defaults(func=handle_pipeline)

    # Data
    p_data = subparsers.add_parser("data", help="Update market data")
    p_data.set_defaults(func=handle_data)

    # Validate
    p_val = subparsers.add_parser("validate", help="Run factor validation")
    p_val.set_defaults(func=handle_validate)

    # Train
    p_train = subparsers.add_parser("train", help="Train alpha models")
    p_train.add_argument("--force-rebuild", action="store_true", help="Force full data rebuild")
    p_train.add_argument("--parallel-models", action="store_true", help="Train models in parallel")
    p_train.add_argument("--all", action="store_true", help="Run post-training analysis")
    p_train.set_defaults(func=handle_train)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Generate alpha signals")
    p_pred.add_argument("--last-day-only", action="store_true", help="Predict only for the most recent date")
    p_pred.set_defaults(func=handle_predict)

    # Monitor
    p_mon = subparsers.add_parser("monitor", help="Run production monitoring")
    p_mon.add_argument("--psi-threshold", type=float, help="PSI threshold for drift alert")
    p_mon.set_defaults(func=handle_monitor)

    # Backtest
    p_bt = subparsers.add_parser("backtest", help="Run backtest simulation")
    p_bt.add_argument("--method", type=str, default="mean_variance", help="Optimization method")
    p_bt.add_argument("--top-n", type=int, default=25, help="Number of stocks")
    p_bt.set_defaults(func=handle_backtest)

    # Optimize
    p_opt = subparsers.add_parser("optimize", help="Run portfolio optimization")
    p_opt.add_argument("--capital", type=float, help="Initial capital")
    p_opt.add_argument("--method", type=str, default="mean_variance", help="Optimization method")
    p_opt.add_argument("--target-vol", type=float, default=0.15, help="Target volatility")
    p_opt.add_argument("--top-n", type=int, default=25, help="Number of assets")
    p_opt.set_defaults(func=handle_optimize)

    # Report
    p_rep = subparsers.add_parser("report", help="Generate executive report")
    p_rep.add_argument("--output", type=str, help="Output file path")
    p_rep.set_defaults(func=handle_report)

    # Hyperopt
    p_hyper = subparsers.add_parser("hyperopt", help="Run hyperparameter optimization")
    p_hyper.set_defaults(func=handle_hyperopt)

    # Deploy
    p_dep = subparsers.add_parser("deploy", help="Manage model deployment")
    p_dep.add_argument("--action", type=str, choices=["check", "archive", "prune"], default="check", help="Action to perform (defaults to 'check')")
    p_dep.add_argument("--all", action="store_true", help="Run check -> archive -> prune sequence. Overrides --action.")
    p_dep.add_argument("--keep", type=int, default=5, help="Archives to keep when pruning")
    p_dep.add_argument("--dry-run", action="store_true", help="Simulate pruning without deleting")
    p_dep.set_defaults(func=handle_deploy)

    # Test
    p_test = subparsers.add_parser("test", help="Run unit and integration tests")
    p_test.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_test.set_defaults(func=handle_test)

    # Clean
    p_clean = subparsers.add_parser("clean", help="Clean cache and temporary files")
    p_clean.add_argument("--all", action="store_true", help="Also clean data cache (requires rebuild)")
    p_clean.set_defaults(func=handle_clean)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

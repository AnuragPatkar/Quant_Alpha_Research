r"""
Central Command-Line Interface (CLI) Orchestrator
=================================================
Unified entry point and orchestration layer for the Quant Alpha Research Platform.

Purpose
-------
Abstracts complex underlying quantitative scripts into a consistent, easily 
executable command structure. Manages the transition of data states from 
Ingestion $\rightarrow$ Alpha Research $\rightarrow$ Production Deployment.

Role in Quantitative Workflow
-----------------------------
Enforces operational discipline by providing a standardized interface for all 
research activities, mitigating manual execution errors. Dispatches heavy 
computational tasks (e.g., Model Training, Backtesting) to isolated subprocesses 
to guarantee clean memory heaps and prevent global state pollution.

Dependencies
------------
- **Argparse**: Robust Subcommand Pattern parsing for granular execution control.
- **Subprocess**: Child process management ensuring GIL release and memory isolation.
- **OS/Sys**: Low-level environment configuration for BLAS/LAPACK threading limits.
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
from pathlib import Path

# Configures linear algebra backends (MKL, OpenBLAS) prior to importing numerical libraries.
# Limits CPU contention in multi-process environments preventing Context Switching Thrashing.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["NUMBA_CACHE_DIR"] = ".numba_cache"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config

def _run_script(script_name: str, args: list[str]):
    """
    Executes a target quantitative script within an isolated subprocess environment.

    Architectural Note:
        Using `subprocess.run` guarantees a pristine memory heap for each task.
        This explicitly prevents global state pollution (e.g., Singleton 
        configurations, Numba JIT caches), memory fragmentation from long-running 
        processes, and Global Interpreter Lock (GIL) contention.

    Args:
        script_name (str): The filename of the target script located in the `scripts/` directory.
        args (list[str]): A list of string arguments to pass to the target executable.

    Returns:
        None

    Raises:
        SystemExit: If the target script is missing, returns a non-zero exit code, 
            or an unhandled execution exception occurs.
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
    Orchestrates the full end-to-end research pipeline via sub-script dispatch.
    
    Translates high-level CLI flags into granular arguments required by the 
    underlying `run_pipeline.py` script, managing the full execution Directed 
    Acyclic Graph (DAG).

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing 
            execution modifiers.

    Returns:
        None
    """
    script_args = []
    # Maps aggregate flags to specific execution parameters
    if args.all:
        script_args.extend(["--validate", "--train", "--deploy", "--backtest"])
    if args.force_rebuild: script_args.append("--full-rebuild")
    if args.parallel_models: script_args.append("--parallel-models")
    if args.skip_data: script_args.append("--skip-data")
    
    _run_script("run_pipeline.py", script_args)

def handle_data(args):
    """
    Dispatches the Market Data Ingestion process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    _run_script("update_data.py", [])

def handle_validate(args):
    """
    Dispatches the Factor Validation (Information Coefficient/Turnover analysis) process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    _run_script("validate_factors.py", [])

def handle_train(args):
    """
    Dispatches the Walk-Forward Model Training process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    script_args = []
    if args.force_rebuild: script_args.append("--force-rebuild")
    if args.parallel_models: script_args.append("--parallel-models")
    if args.all: script_args.append("--all")

    _run_script("train_models.py", script_args)

def handle_predict(args):
    """
    Dispatches the Alpha Signal Generation (Inference) process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    script_args = []
    if args.last_day_only: script_args.append("--last-day-only")

    _run_script("generate_predictions.py", script_args)

def handle_monitor(args):
    """
    Dispatches the Production Health Monitoring process (Drift/Staleness evaluation).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    script_args = []
    if args.psi_threshold: script_args.extend(["--psi-threshold", str(args.psi_threshold)])

    _run_script("monitor_production.py", script_args)

def handle_backtest(args):
    """
    Dispatches the Historical Simulation Engine.

    Args:
        args (argparse.Namespace): Parsed command-line arguments mapping 
            optimization boundaries.

    Returns:
        None
    """
    script_args = []
    if args.method: script_args.extend(["--method", args.method])
    if args.top_n: script_args.extend(["--top-n", str(args.top_n)])

    _run_script("run_backtest.py", script_args)

def handle_optimize(args):
    """
    Dispatches the Portfolio Construction & Optimization Engine.

    Args:
        args (argparse.Namespace): Parsed command-line arguments mapping 
            capital and target bounds.

    Returns:
        None
    """
    script_args = []
    if args.capital: script_args.extend(["--capital", str(args.capital)])
    if args.method: script_args.extend(["--method", args.method])
    if args.target_vol: script_args.extend(["--target-vol", str(args.target_vol)])
    if args.top_n: script_args.extend(["--top-n", str(args.top_n)])

    _run_script("optimize_portfolio.py", script_args)

def handle_report(args):
    """
    Dispatches the Executive Reporting module.

    Args:
        args (argparse.Namespace): Parsed command-line arguments defining output targets.

    Returns:
        None
    """
    script_args = []
    if args.output: script_args.extend(["--output-file", args.output])

    _run_script("create_report.py", script_args)

def handle_hyperopt(args):
    """
    Dispatches the Bayesian Hyperparameter Optimization process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    _run_script("run_hyperopt.py", [])

def handle_deploy(args):
    """
    Dispatches the Model Lifecycle Manager (Archive/Prune procedures).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    script_args = []
    # Resolves precedence logic where global flags override specific action parameters
    if args.all:
        script_args.append("--all")
    else:
        script_args.extend(["--action", args.action])

    if args.keep: script_args.extend(["--keep", str(args.keep)])
    if args.dry_run: script_args.append("--dry-run")

    _run_script("deploy_model.py", script_args)

def handle_test(args):
    """
    Executes the automated test suite via Pytest.
    
    Runs both unit tests verifying individual component logic (e.g., Factor calculations)
    and integration tests validating cross-module workflows (e.g., Data -> Model -> Signal).

    Args:
        args (argparse.Namespace): Parsed command-line arguments controlling test verbosity.

    Returns:
        None
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

def handle_typecheck(args):
    """
    Executes static type checking via Mypy to assert structural safety.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    logger = logging.getLogger("TypeCheck")

    try:
        import mypy
    except ImportError:
        logger.error("❌ 'mypy' is not installed. Run: pip install -e .[dev]")
        sys.exit(1)

    logger.info("🔍 Running Static Type Checking (mypy)...")
    try:
        cmd = [sys.executable, "-m", "mypy", "."]
        subprocess.run(cmd, check=True)
        logger.info("✅ Type check passed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Type check failed (Exit Code: {e.returncode})")
        sys.exit(e.returncode)

def handle_clean(args):
    """
    Purges temporary artifacts, caches, and logs to reset the environment state.

    Args:
        args (argparse.Namespace): Parsed command-line arguments defining purge scope.

    Returns:
        None
    """
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

def setup_logging():
    """
    Initializes the logging subsystem with minimal initialization overhead.
    
    Configures logging locally to avoid importing heavy numerical dependencies 
    (Pandas, NumPy) located in `quant_alpha.utils`. This ensures the CLI 
    starts in $O(1)$ time relative to the project size.

    Args:
        None

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    """
    Primary execution routine defining the CLI argument parser boundaries.

    Constructs the Subcommand Pattern parser and dynamically routes execution 
    to the appropriate handler function based on user input.

    Args:
        None

    Returns:
        None
    """
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

    # Typecheck
    p_type = subparsers.add_parser("typecheck", help="Run static type checking")
    p_type.set_defaults(func=handle_typecheck)

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

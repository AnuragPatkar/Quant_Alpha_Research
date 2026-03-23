"""
End-to-End Quantitative Research Pipeline Orchestrator
======================================================
Automated workflow engine for the full Alpha Research lifecycle, from data ingestion
to portfolio construction and reporting.

Purpose
-------
This module serves as the **Master Controller** for the quantitative system. It 
sequentially executes the individual components of the ML-driven alpha platform, 
ensuring dependency management and state consistency between steps. It replaces 
manual script execution with a unified CLI interface capable of running the entire
production DAG (Directed Acyclic Graph) or specific sub-graphs.

Workflow Sequence:
1.  **Data Update**: Incremental fetch of OHLCV and fundamental data.
2.  **Factor Validation**: Statistical efficacy checks ($IC$, $t$-stat).
3.  **Model Training**: Walk-forward GBDT ensemble training (LightGBM/XGB/CatBoost).
4.  **Deployment**: Artifact promotion, health checks, and archival.
5.  **Inference**: Generation of OOS alpha signals ($S_t$).
6.  **Monitoring**: Drift detection ($PSI$) and signal integrity checks.
7.  **Simulation**: Historical backtesting of the generated signals.
8.  **Optimization**: Construction of optimal target portfolios ($w^*$).
9.  **Reporting**: Generation of executive summaries and visualizations.

Usage
-----
Executed via CLI to trigger the full pipeline or specific phases.

.. code-block:: bash

    # 1. Full Production Run (Data -> Inference -> Orders -> Report)
    python scripts/run_pipeline.py

    # 2. Retraining Run (Data -> Validation -> Train -> Deploy)
    python scripts/run_pipeline.py --validate --train --deploy

    # 3. Simulation Run (Backtest with specific settings)
    python scripts/run_pipeline.py --skip-data --backtest --opt-method risk_parity

Importance
----------
-   **Operational Continuity**: Guarantees that downstream steps (e.g., Inference)
    always run on the freshest available data from upstream steps.
-   **Error Containment**: Implements "Fail-Fast" logic; if critical steps like
    Data Update fail, the pipeline aborts to prevent corrupted signal generation.
-   **Reproducibility**: Enforces a deterministic execution order for research
    experiments and production runs.

Tools & Frameworks
------------------
-   **Subprocess**: Spawns isolated processes for each step to prevent namespace
    pollution and memory leaks between heavy ML tasks.
-   **Argparse**: CLI parameter parsing for granular workflow control.
-   **Logging**: Centralized execution tracking and error reporting.
"""
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quant_alpha.utils import setup_logging

setup_logging()
logger = logging.getLogger("Quant_Alpha")

def run_step(script_name: str, args: Optional[list[str]] = None):
    """
    Executes a quantitative script within an isolated subprocess environment.
    
    Isolation Strategy:
    Using `subprocess` guarantees that each pipeline step initializes with a clean 
    memory heap and namespace. This strictly prevents memory leaks and cross-contamination 
    of global singletons across computationally intensive ML tasks.

    Args:
        script_name (str): The filename of the target script located in the `scripts/` directory.
        args (Optional[list[str]], optional): A list of string arguments to pass to the 
            target executable. Defaults to None.

    Returns:
        None

    Raises:
        SystemExit: If the target script is missing, returns a non-zero exit code, 
            or an unhandled execution exception occurs.
    """
    if args is None:
        args = []
    
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        logger.error(f"❌ Script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)] + args
    logger.info(f"🚀 [PIPELINE] Running {script_name} {' '.join(args)}...")
    
    try:
        # Fail-Fast mechanism: Immediately aborts the pipeline upon a non-zero exit code, 
        # preventing the propagation of corrupted states to downstream execution nodes.
        subprocess.run(cmd, check=True)
        logger.info(f"✅ [PIPELINE] {script_name} completed.\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ [PIPELINE] {script_name} failed (exit code {e.returncode}). Aborting pipeline.")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"❌ [PIPELINE] Execution failed: {e}")
        sys.exit(1)

def run():
    """
    Main execution routine for the pipeline orchestrator.

    Parses command-line arguments to determine the active Directed Acyclic Graph (DAG) 
    components, then sequentially dispatches execution while strictly enforcing 
    dependency boundaries.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Quant Alpha Research Pipeline Orchestrator")
    
    parser.add_argument("--skip-data", action="store_true", help="Skip data download/update step.")
    parser.add_argument("--validate", action="store_true", help="Run factor validation (validate_factors.py).")
    parser.add_argument("--train", action="store_true", help="Run model training (train_models.py).")
    parser.add_argument("--deploy", action="store_true", help="Run deployment checks & archival (deploy_model.py).")
    parser.add_argument("--full-rebuild", action="store_true", help="Force full rebuild of data and models.")
    parser.add_argument("--parallel-models", action="store_true", help="Train models in parallel (requires 16GB+ RAM).")
    parser.add_argument("--backtest", action="store_true", help="Run backtest simulation (run_backtest.py).")
    
    parser.add_argument("--capital", type=float, default=1000000, help="Capital for portfolio optimization.")
    parser.add_argument("--opt-method", type=str, default="mean_variance", help="Optimization method (mean_variance, risk_parity, etc).")
    parser.add_argument("--target-vol", type=float, default=0.15, help="Target annualized volatility (default: 0.15).")
    parser.add_argument("--top-n", type=int, default=25, help="Number of assets to hold (default: 25).")
    
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  QUANT ALPHA PIPELINE START")
    logger.info("=" * 60)

    # 1. Data Ingestion
    # Orchestrates updates to the local Data Lake, guaranteeing downstream models 
    # consume the most current point-in-time universe data.
    if not args.skip_data:
        run_step("update_data.py")
    else:
        logger.info("⏭️  [PIPELINE] Skipping data update.")

    # 2. Factor Quality Assurance
    # Validates alpha factor cross-sectional efficacy prior to model ingestion.
    if args.validate:
        run_step("validate_factors.py")
    else:
        logger.info("⏭️  [PIPELINE] Skipping factor validation (use --validate to run).")

    # 3. Model Training & Validation
    # Executes purged K-Fold walk-forward training for the GBDT ensemble.
    if args.train or args.full_rebuild:
        train_args = []
        if args.full_rebuild:
            train_args.append("--force-rebuild")
        if args.parallel_models:
            train_args.append("--parallel-models")
        
        run_step("train_models.py", train_args)
    else:
        logger.info("⏭️  [PIPELINE] Skipping model training (use --train to run).")

    # 4. Model Deployment
    # Conducts artifact health checks and enforces promotion criteria.
    if args.deploy:
        run_step("deploy_model.py", ["--all"])
    else:
        logger.info("⏭️  [PIPELINE] Skipping deployment checks (use --deploy to run).")

    # 5. Inference
    # Applies localized scaling and generates out-of-sample alpha signals.
    run_step("generate_predictions.py")

    # 6. Production Monitoring
    # Quantifies distributional drift and guards against model collapse.
    run_step("monitor_production.py")

    # 7. Historical Simulation
    # Evaluates strategy integrity with transaction costs and slippage logic.
    if args.backtest:
        run_step("run_backtest.py", [
            "--method", args.opt_method,
            "--top-n", str(args.top_n)
        ])
    else:
        logger.info("⏭️  [PIPELINE] Skipping backtest (use --backtest to run).")

    # 8. Portfolio Construction
    # Solves the optimal convex allocation mapping given institutional constraints.
    opt_args = [
        "--capital", str(args.capital),
        "--method", args.opt_method,
        "--target-vol", str(args.target_vol),
        "--top-n", str(args.top_n)
    ]
    run_step("optimize_portfolio.py", opt_args)

    # 9. Executive Reporting
    # Summarizes state diagnostics into management-level artifacts.
    run_step("create_report.py")

    logger.info("=" * 60)
    logger.info("🏁 QUANT ALPHA PIPELINE COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    run()
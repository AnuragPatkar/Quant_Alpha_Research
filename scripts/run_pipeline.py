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

# --- PROJECT SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quant_alpha.utils import setup_logging

setup_logging()
logger = logging.getLogger("Quant_Alpha")

def run_step(script_name: str, args: Optional[list[str]] = None):
    """
    Executes a script in a dedicated subprocess.
    
    Isolation Strategy:
    Using `subprocess` ensures that each step starts with a clean memory heap and
    namespace, preventing memory leaks and side effects (e.g., modified singletons)
    from propagating between heavy ML tasks.
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
        # Fail-Fast: check=True raises CalledProcessError on non-zero exit codes,
        # halting the pipeline immediately to prevent cascading failures.
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
    Main entry point for the pipeline.
    Parses command line arguments and executes steps in order.
    """
    parser = argparse.ArgumentParser(description="Quant Alpha Research Pipeline Orchestrator")
    
    # Workflow Control Flags
    parser.add_argument("--skip-data", action="store_true", help="Skip data download/update step.")
    parser.add_argument("--validate", action="store_true", help="Run factor validation (validate_factors.py).")
    parser.add_argument("--train", action="store_true", help="Run model training (train_models.py).")
    parser.add_argument("--deploy", action="store_true", help="Run deployment checks & archival (deploy_model.py).")
    parser.add_argument("--full-rebuild", action="store_true", help="Force full rebuild of data and models.")
    parser.add_argument("--parallel-models", action="store_true", help="Train models in parallel (requires 16GB+ RAM).")
    parser.add_argument("--backtest", action="store_true", help="Run backtest simulation (run_backtest.py).")
    
    # Portfolio Construction Parameters
    parser.add_argument("--capital", type=float, default=1000000, help="Capital for portfolio optimization.")
    parser.add_argument("--opt-method", type=str, default="mean_variance", help="Optimization method (mean_variance, risk_parity, etc).")
    parser.add_argument("--target-vol", type=float, default=0.15, help="Target annualized volatility (default: 0.15).")
    parser.add_argument("--top-n", type=int, default=25, help="Number of assets to hold (default: 25).")
    
    # Parse arguments
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  QUANT ALPHA PIPELINE START")
    logger.info("=" * 60)

    # 1. Data Ingestion (Dependency Root)
    if not args.skip_data:
        run_step("update_data.py")
    else:
        logger.info("⏭️  [PIPELINE] Skipping data update.")

    # 2. Factor Quality Assurance (Optional)
    if args.validate:
        run_step("validate_factors.py")
    else:
        logger.info("⏭️  [PIPELINE] Skipping factor validation (use --validate to run).")

    # 3. Model Training & Validation (Optional / Rebuild)
    if args.train or args.full_rebuild:
        train_args = []
        if args.full_rebuild:
            train_args.append("--force-rebuild")
        if args.parallel_models:
            train_args.append("--parallel-models")
        
        run_step("train_models.py", train_args)
    else:
        logger.info("⏭️  [PIPELINE] Skipping model training (use --train to run).")

    # 4. Model Deployment (Gatekeeping)
    if args.deploy:
        run_step("deploy_model.py", ["--all"])
    else:
        logger.info("⏭️  [PIPELINE] Skipping deployment checks (use --deploy to run).")

    # 5. Inference (Production Signal Generation)
    run_step("generate_predictions.py")

    # 6. Production Monitoring (Drift Detection)
    run_step("monitor_production.py")

    # 7. Historical Simulation (Optional)
    if args.backtest:
        run_step("run_backtest.py", [
            "--method", args.opt_method,
            "--top-n", str(args.top_n)
        ])
    else:
        logger.info("⏭️  [PIPELINE] Skipping backtest (use --backtest to run).")

    # 8. Portfolio Construction (Order Generation)
    opt_args = [
        "--capital", str(args.capital),
        "--method", args.opt_method,
        "--target-vol", str(args.target_vol),
        "--top-n", str(args.top_n)
    ]
    run_step("optimize_portfolio.py", opt_args)

    # 9. Executive Reporting
    run_step("create_report.py")

    logger.info("=" * 60)
    logger.info("🏁 QUANT ALPHA PIPELINE COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    run()
"""
run_pipeline.py
===============
Orchestrates the full Quant Alpha Pipeline (ML-based).
Replaces the old heuristic pipeline.

Sequence:
1. update_data.py        (Fetch latest market data)
2. validate_factors.py   (Optional: Quality Assurance)
3. train_models.py       (Optional: Retrain models)
4. deploy_model.py       (Optional: Health Check & Archive)
5. generate_predictions.py (Inference: Generate Alpha Signals)
6. run_backtest.py       (Optional: Simulation)
7. optimize_portfolio.py (Portfolio Construction: Generate Orders)
8. create_report.py      (Reporting: Executive Summary)
"""
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# --- PROJECT SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("Quant_Alpha")

def run_step(script_name: str, args: list[str] = None):
    """Runs a script as a subprocess to ensure clean state."""
    if args is None:
        args = []
    
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        logger.error(f"❌ Script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)] + args
    logger.info(f"🚀 [PIPELINE] Running {script_name} {' '.join(args)}...")
    
    try:
        # check=True raises CalledProcessError on non-zero exit code
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
    
    # Pipeline Control Flags
    parser.add_argument("--skip-data", action="store_true", help="Skip data download/update step.")
    parser.add_argument("--validate", action="store_true", help="Run factor validation (validate_factors.py).")
    parser.add_argument("--train", action="store_true", help="Run model training (train_models.py).")
    parser.add_argument("--deploy", action="store_true", help="Run deployment checks & archival (deploy_model.py).")
    parser.add_argument("--full-rebuild", action="store_true", help="Force full rebuild of data and models.")
    parser.add_argument("--parallel-models", action="store_true", help="Train models in parallel (requires 16GB+ RAM).")
    parser.add_argument("--backtest", action="store_true", help="Run backtest simulation (run_backtest.py).")
    
    # Optimization Flags
    parser.add_argument("--capital", type=float, default=1000000, help="Capital for portfolio optimization.")
    parser.add_argument("--opt-method", type=str, default="mean_variance", help="Optimization method (mean_variance, risk_parity, etc).")
    parser.add_argument("--target-vol", type=float, default=0.15, help="Target annualized volatility (default: 0.15).")
    parser.add_argument("--top-n", type=int, default=25, help="Number of assets to hold (default: 25).")
    
    # Parse arguments
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  QUANT ALPHA PIPELINE START")
    logger.info("=" * 60)

    # 1. Data Update
    if not args.skip_data:
        run_step("update_data.py")
    else:
        logger.info("⏭️  [PIPELINE] Skipping data update.")

    # 2. Factor Validation
    if args.validate:
        run_step("validate_factors.py")
    else:
        logger.info("⏭️  [PIPELINE] Skipping factor validation (use --validate to run).")

    # 3. Model Training (Optional)
    if args.train or args.full_rebuild:
        train_args = []
        if args.full_rebuild:
            train_args.append("--force-rebuild")
        if args.parallel_models:
            train_args.append("--parallel-models")
        
        run_step("train_models.py", train_args)
    else:
        logger.info("⏭️  [PIPELINE] Skipping model training (use --train to run).")

    # 4. Deployment (Health Check & Archive)
    if args.deploy:
        run_step("deploy_model.py", ["--all"])
    else:
        logger.info("⏭️  [PIPELINE] Skipping deployment checks (use --deploy to run).")

    # 5. Inference (Generate Predictions)
    run_step("generate_predictions.py")

    # 6. Backtest
    if args.backtest:
        run_step("run_backtest.py", [
            "--method", args.opt_method,
            "--top-n", str(args.top_n)
        ])
    else:
        logger.info("⏭️  [PIPELINE] Skipping backtest (use --backtest to run).")

    # 7. Portfolio Optimization
    opt_args = [
        "--capital", str(args.capital),
        "--method", args.opt_method,
        "--target-vol", str(args.target_vol),
        "--top-n", str(args.top_n)
    ]
    run_step("optimize_portfolio.py", opt_args)

    # 8. Reporting
    run_step("create_report.py")

    logger.info("=" * 60)
    logger.info("🏁 QUANT ALPHA PIPELINE COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    run()
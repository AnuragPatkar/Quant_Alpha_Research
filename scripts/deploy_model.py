"""
Production Model Deployment & Lifecycle Management
==================================================
Orchestration engine for the promotion, validation, and maintenance of alpha
models in the live trading environment.

Purpose
-------
This module serves as the **Gatekeeper** for the production inference pipeline.
It enforces strict quality assurance protocols before allowing models to
influence capital allocation. Key responsibilities include:
1.  **Health Verification**: Conducting "smoke tests" on serialized artifacts
    to ensure loadability and schema compatibility.
2.  **Degeneracy Detection**: Identifying model collapse (zero-variance
    predictions) prior to deployment.
3.  **Lifecycle Management**: Handling the archival of superseded models with
    audit-grade manifests and automated disk space management (pruning).
4.  **Staleness Monitoring**: Ensuring the live signal cache
    (`ensemble_predictions.parquet`) remains within the valid look-forward window.

Role in Quantitative Workflow
-----------------------------
Executed via the CLI or triggered by the CI/CD pipeline.

.. code-block:: bash

    # 1. Health Check (Pre-Trading)
    python scripts/deploy_model.py --action check

    # 2. Archive & Prune (Post-Retraining)
    python scripts/deploy_model.py --all --keep 5

Importance
----------
-   **Operational Risk**: Prevents "silent failures" where stale or corrupted
    models generate invalid signals, potentially leading to unintended market exposure.
-   **Auditability**: Generates a persistent `manifest.json` for every deployment,
    logging performance metrics ($IC$, $t$-stat) and feature sets for compliance.
-   **Resource Optimization**: Manages filesystem constraints by pruning legacy
    archives based on modification time ($mtime$), preventing storage exhaustion.

Tools & Frameworks
------------------
-   **Joblib**: Efficient serialization/deserialization of model artifacts.
-   **Pandas**: Time-series alignment and Parquet I/O for signal cache inspection.
-   **NumPy**: Statistical computations (variance, standard deviation) for checks.
-   **Pathlib**: Object-oriented filesystem paths for robust cross-platform file handling.
"""

import sys
import json
import shutil
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config

from quant_alpha.utils import setup_logging
setup_logging()

logger = logging.getLogger("Quant_Alpha")


# ==============================================================================
# SERIALIZATION COMPATIBILITY LAYER
# ==============================================================================
def weighted_symmetric_mae(y_true, y_pred):
    r"""
    Custom gradient/hessian for the Weighted Symmetric Mean Absolute Error objective.
    Must match the definition in ``train_models.py`` exactly to ensure deserialization integrity.

    Mathematical Form:
    .. math::
        L(y, \\hat{y}) = -w \cdot \\tanh(y - \\hat{y})

    Note: This function is required in the global namespace for ``joblib`` to resolve
    the function pointer when unpickling models trained in a ``__main__`` context.
    """
    residuals = y_true - y_pred
    weights   = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad      = -weights * np.tanh(residuals)
    hess      = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess


# Deserialization Hook (Monkey Patch):
# Injects the custom objective function into the `__main__` namespace to prevent
# `AttributeError` during the unpickling of legacy artifacts.
try:
    import __main__
    setattr(__main__, "weighted_symmetric_mae", weighted_symmetric_mae)  # type: ignore
except Exception:
    pass   # Defensive: joblib.load() try/except block handles failure gracefully

# Static bindings for algorithmic promotion thresholds
_PROD_IC_THRESHOLD  = getattr(config, "PROD_IC_THRESHOLD",    0.010)
_PROD_IC_TSTAT      = getattr(config, "PROD_IC_TSTAT",         2.5)
_ENS_IC_THRESHOLD   = getattr(config, "MIN_OOS_IC_THRESHOLD",  0.005)
_ENS_IC_TSTAT       = getattr(config, "MIN_OOS_IC_TSTAT",      1.5)


def _gate_tier(ic: float, ic_std: float, n_dates: int) -> str:
    r"""
    Classifies model quality based on Information Coefficient (IC) statistics.

    Statistical Significance ($t$-test):
    .. math::
        t = \\frac{\\mu_{IC}}{\\sigma_{IC} / \sqrt{N}}

    Args:
        ic (float): The mean out-of-sample Information Coefficient.
        ic_std (float): The standard deviation of the daily Information Coefficient.
        n_dates (int): The number of unique trading days in the out-of-sample period.

    Returns:
        str: The deployment tier classification ('✅ PROD', '🟡 ENSEMBLE', or '❌ GATED').
    """
    tstat = ic / (ic_std / (n_dates ** 0.5)) if n_dates > 0 and ic_std > 0 else 0.0
    if ic >= _PROD_IC_THRESHOLD and tstat >= _PROD_IC_TSTAT:
        return "✅ PROD"
    if ic >= _ENS_IC_THRESHOLD and tstat >= _ENS_IC_TSTAT:
        return "🟡 ENSEMBLE"
    return "❌ GATED"


def _prediction_cache_age() -> tuple[str | None, int | None]:
    """
    Audits the temporal freshness of the active signal cache (`ensemble_predictions.parquet`).
    
    Crucial for preventing Look-Ahead Bias by ensuring signals are actionable.

    Args:
        None

    Returns:
        tuple[str | None, int | None]: A tuple containing the last signal date as a string 
            and the number of trading days elapsed. Returns (None, None) upon missing cache.
    """
    cache_path = config.CACHE_DIR / "ensemble_predictions.parquet"
    if not cache_path.exists():
        return None, None
    try:
        preds      = pd.read_parquet(cache_path, columns=["date"])
        last_sig   = pd.to_datetime(preds["date"]).max().date()
        today      = pd.Timestamp(datetime.now().date())
        bdays      = pd.bdate_range(str(last_sig), today)
        age_tdays  = max(0, len(bdays) - 1)
        return str(last_sig), age_tdays
    except Exception:
        return None, None


def _build_dummy_df(features: list, n_rows: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic DataFrame for Fuzz Testing the inference engine.

    Used to detect schema mismatches or serialization corruption before production load.

    Args:
        features (list): The list of feature column names expected by the model.
        n_rows (int, optional): The number of synthetic rows to generate. Defaults to 50.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A synthetic dataset populated with Gaussian noise for numericals 
            and "Unknown" for categoricals to test pipeline resilience.
    """
    KNOWN_CATS = {
        "sector", "industry", "ticker", "exchange",
        "country", "currency", "gics_sector", "gics_industry",
    }
    rng = np.random.default_rng(seed=seed)

    data = {}
    for col in features:
        is_cat = col.lower() in KNOWN_CATS or col.lower().endswith(
            ("_sector", "_industry", "_type", "_category")
        )
        if is_cat:
            data[col] = pd.Categorical(["Unknown"] * n_rows)
        else:
            data[col] = rng.standard_normal(n_rows)

    return pd.DataFrame(data)

class DeploymentManager:
    def __init__(self):
        self.prod_dir    = config.MODELS_DIR / "production"
        self.archive_dir = config.MODELS_DIR / "archive"
        
    def verify_deployment(self) -> bool:
        """
        Executes a comprehensive Health Check (Smoke Test) on production artifacts.

        Validation Steps:
        1.  **Latency Audit**: Verifies `ensemble_predictions.parquet` is within the
            re-training window ($T < 5$ days).
        2.  **Deserialization Check**: Attempts to load (`joblib.load`) all model artifacts.
        3.  **Schema Compatibility**: Constructs a synthetic DataFrame with correct
            dtypes (Categorical vs. Float) to prevent inference engine crashes.
        4.  **Variance Constraint**: Asserts prediction variance $\\sigma^2_{pred} > \\epsilon$
            to detect model collapse (constant output) or gradient explosion.

        Args:
            None

        Returns:
            bool: True if all systems are nominal, False otherwise.
        """
        models = list(self.prod_dir.glob("*_latest.pkl"))
        if not models:
            logger.error("[CHECK] ❌ No models found in production directory.")
            return False

        sig_date, sig_age = _prediction_cache_age()
        if sig_date is None:
            logger.warning(
                "[CHECK] ⚠️  ensemble_predictions.parquet not found. "
                "Signals may not have been generated yet."
            )
        elif sig_age is not None and sig_age > 21:
            logger.error(
                f"[CHECK] 🚨 Signal cache is {sig_age} trading days old "
                f"(last: {sig_date}). RETRAIN URGENTLY before live trading."
            )
        elif sig_age is not None and sig_age > 5:
            logger.warning(
                f"[CHECK] ⚠️  Signal cache is {sig_age} trading days old "
                f"(last: {sig_date}). Consider retraining."
            )
        else:
            logger.info(
                f"[CHECK] ✅ Signal cache fresh: {sig_date} ({sig_age} trading days old)."
            )

        logger.info(f"[CHECK] Verifying {len(models)} production model(s)...")

        passed = 0
        for model_path in sorted(models):
            name = model_path.stem.replace("_latest", "").capitalize()
            try:
                try:
                    payload = joblib.load(model_path)
                except AttributeError as exc:
                    logger.error(
                        f"   ❌ {name}: Pickle load failed — custom objective not "
                        f"found in __main__: {exc}. "
                        "Fix: move weighted_symmetric_mae to quant_alpha/objectives.py "
                        "and import it in both train_models.py and deploy_model.py."
                    )
                    continue

                if not isinstance(payload, dict) or "model" not in payload:
                    logger.error(f"   ❌ {name}: Invalid payload (missing 'model' key).")
                    continue

                model      = payload["model"]
                features   = payload.get("feature_names", [])
                trained_to = payload.get("trained_to", "Unknown")
                metrics    = payload.get("oos_metrics", {})

                # Extract OOS metrics to ensure mathematical continuity during promotion
                ic      = metrics.get("ic_mean",  0.0)
                ic_std  = metrics.get("ic_std",   1e-8)
                n_dates = metrics.get("n_dates",  1)
                tstat    = ic / (ic_std / (n_dates ** 0.5)) if n_dates > 0 else 0.0
                ann_icir = (ic / ic_std) * (252 ** 0.5) if ic_std > 0 else 0.0
                tier     = _gate_tier(ic, ic_std, n_dates)

                # Evaluate inference stability to prevent deployment of collapsed models
                smoke_ok  = True
                smoke_msg = "No features stored — inference test skipped."
                if features:
                    dummy = _build_dummy_df(features, n_rows=50)
                    try:
                        preds = model.predict(dummy)
                        preds = np.asarray(preds, dtype=float)

                        pred_std = float(np.std(preds))
                        if pred_std < 1e-6:
                            smoke_ok  = False
                            smoke_msg = (
                                f"DEGENERATE — all predictions ≈ {preds.mean():.4f} "
                                f"(std={pred_std:.2e}). Model may have collapsed."
                            )
                        else:
                            smoke_msg = (
                                f"OK — pred range [{preds.min():.3f}, {preds.max():.3f}], "
                                f"std={pred_std:.4f}"
                            )
                    except Exception as exc:
                        smoke_ok  = False
                        smoke_msg = f"PREDICT FAILED: {exc}"

                status_icon = "✅" if smoke_ok else "❌"
                logger.info(
                    f"   {status_icon} {name:<12} | "
                    f"IC={ic:+.4f}  t={tstat:.1f}  AnnICIR={ann_icir:.2f}  {tier} | "
                    f"Features={len(features)}  TrainedTo={trained_to} | "
                    f"Smoke: {smoke_msg}"
                )

                if smoke_ok:
                    passed += 1

            except Exception as exc:
                logger.error(f"   ❌ {name}: Unexpected error: {exc}", exc_info=False)

        all_ok = passed == len(models)
        icon = "✅" if all_ok else "⚠️ "
        logger.info(
            f"[CHECK] {icon} Verification complete: {passed}/{len(models)} models healthy."
        )
        return all_ok

    def archive_current_models(self) -> Path | None:
        """
        Creates an immutable snapshot of current production artifacts.

        Generates a `manifest.json` to provide a regulatory audit trail, recording:
        - Model performance metrics ($IC$, $t$-stat).
        - Feature usage and training cutoff dates.
        - Signal cache latency at the time of archival.

        Args:
            None

        Returns:
            Path: The directory of the created archive, or None if empty.
        """
        models = list(self.prod_dir.glob("*_latest.pkl"))
        if not models:
            logger.warning("[ARCHIVE] No production models found to archive.")
            return None

        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.archive_dir / f"deployment_{timestamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[ARCHIVE] Snapshotting {len(models)} model(s) → {snapshot_dir.name}..."
        )

        manifest = {
            "archived_at":    datetime.now().isoformat(),
            "archive_name":   snapshot_dir.name,
            "models":         [],
        }

        sig_date, sig_age = _prediction_cache_age()
        manifest["signal_cache"] = {
            "last_date":        sig_date,
            "trading_days_old": sig_age,
        }

        for model_path in sorted(models):
            entry = {"file": model_path.name, "status": "ok"}
            try:
                shutil.copy2(model_path, snapshot_dir)

                # Persist statistical telemetry to the manifest for compliance auditing
                payload = joblib.load(snapshot_dir / model_path.name)
                metrics = payload.get("oos_metrics", {})
                features = payload.get("feature_names", [])
                ic      = metrics.get("ic_mean", 0.0)
                ic_std  = metrics.get("ic_std",  1e-8)
                n_d     = metrics.get("n_dates", 1)
                tstat   = ic / (ic_std / (n_d ** 0.5)) if n_d > 0 else 0.0
                ann_icir = (ic / ic_std) * (252 ** 0.5) if ic_std > 0 else 0.0

                entry.update({
                    "model_name":    model_path.stem.replace("_latest", ""),
                    "trained_to":    payload.get("trained_to", "Unknown"),
                    "ic_mean":       round(ic, 6),
                    "ic_std":        round(ic_std, 6),
                    "t_stat":        round(tstat, 2),
                    "ann_icir":      round(ann_icir, 2),
                    "gate_tier":     _gate_tier(ic, ic_std, n_d),
                    "n_features":    len(features),
                    "feature_names": features,
                })
                logger.info(f"   Archived: {model_path.name}")
            except AttributeError:
                entry["status"] = "copied_no_metrics (custom objective unpickle failed)"
                logger.warning(f"   Archived {model_path.name} (could not read metrics).")
            except Exception as exc:
                entry["status"] = f"FAILED: {exc}"
                logger.error(f"   Failed to archive {model_path.name}: {exc}")

            manifest["models"].append(entry)

        manifest_path = snapshot_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"[ARCHIVE] Manifest written → {manifest_path}")
        logger.info("[ARCHIVE] Archive complete.")
        return snapshot_dir

    def prune_archives(self, keep_last: int = 5, dry_run: bool = False) -> None:
        """
        Enforces the Archive Retention Policy to mitigate storage exhaustion.

        Relies on filesystem modification time (st_mtime) to identify and purge 
        legacy model snapshots, providing capacity alerts if storage remains constrained.

        Args:
            keep_last (int, optional): The number of recent archives to retain. Defaults to 5.
            dry_run (bool, optional): If True, simulates the pruning process without deleting 
                any files. Defaults to False.

        Returns:
            None
        """
        archives = sorted(
            [d for d in self.archive_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime
        )

        def _dir_size_mb(path: Path) -> float:
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6

        total_size_mb = sum(_dir_size_mb(a) for a in archives)
        logger.info(
            f"[PRUNE] Found {len(archives)} archive(s) | "
            f"Total size: {total_size_mb:.0f} MB"
        )

        if len(archives) <= keep_last:
            logger.info(
                f"[PRUNE] {len(archives)} ≤ keep_last={keep_last}. Nothing to prune."
            )
            return

        to_delete = archives[:-keep_last]
        to_keep   = archives[-keep_last:]

        logger.info(
            f"[PRUNE] Will {'preview' if dry_run else 'delete'} "
            f"{len(to_delete)} archive(s), keep {len(to_keep)} most recent."
        )

        freed_mb = 0.0
        for archive in to_delete:
            size_mb = _dir_size_mb(archive)
            freed_mb += size_mb
            age_str = datetime.fromtimestamp(
                archive.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M")

            if dry_run:
                logger.info(
                    f"   [DRY-RUN] Would delete: {archive.name} "
                    f"({size_mb:.0f} MB, created {age_str})"
                )
            else:
                try:
                    shutil.rmtree(archive)
                    logger.info(
                        f"   Deleted: {archive.name} "
                        f"({size_mb:.0f} MB, created {age_str})"
                    )
                except Exception as exc:
                    logger.error(f"   Failed to delete {archive.name}: {exc}")

        action = "Would free" if dry_run else "Freed"
        remaining_mb = total_size_mb - (0 if dry_run else freed_mb)

        logger.info(f"[PRUNE] {action} {freed_mb:.0f} MB. Remaining: {remaining_mb:.0f} MB")

        if remaining_mb > 5_000:
            logger.warning(
                f"[PRUNE] ⚠️  Archive folder is {remaining_mb/1000:.1f} GB. "
                "Consider reducing --keep or moving archives to cold storage."
            )

def main():
    """
    Primary execution routine for the deployment manager.

    Parses command-line arguments to orchestrate health checks, artifact archival, 
    and filesystem pruning.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Production Model Deployment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/deploy_model.py --action check
  python scripts/deploy_model.py --action archive
  python scripts/deploy_model.py --action prune --keep 3
  python scripts/deploy_model.py --action prune --keep 3 --dry-run
  python scripts/deploy_model.py --all
        """,
    )
    parser.add_argument(
        "--action", type=str,
        choices=["check", "archive", "prune"],
        help="Specific action to perform."
    )
    parser.add_argument(
        "--all", action="store_true",
        help=(
            "Execute full lifecycle: Check -> Archive -> Prune. "
            "Execution Order: Validate -> Archive -> Prune."
        )
    )
    parser.add_argument(
        "--keep", type=int, default=5,
        help="Number of archives to keep when pruning (default: 5)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="For --action prune: show what would be deleted without removing anything."
    )
    args = parser.parse_args()

    manager = DeploymentManager()

    if args.all:
        logger.info("[ALL] Step 1/3 — Health Check")
        healthy = manager.verify_deployment()
        if not healthy:
            logger.warning(
                "[ALL] ⚠️  Health check flagged issues. Proceeding with archive, "
                "but review check output before live trading."
            )

        logger.info("[ALL] Step 2/3 — Archive")
        archive_path = manager.archive_current_models()

        logger.info("[ALL] Step 3/3 — Prune")
        if archive_path is not None:
            manager.prune_archives(keep_last=args.keep, dry_run=args.dry_run)
        else:
            logger.info("[ALL] Skipping prune — no new archive was created.")

    elif args.action == "check":
        manager.verify_deployment()
    elif args.action == "archive":
        manager.archive_current_models()
    elif args.action == "prune":
        manager.prune_archives(keep_last=args.keep, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
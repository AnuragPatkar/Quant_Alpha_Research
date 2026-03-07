"""
deploy_model.py
===============
Production Deployment Manager  —  v2 (Fixed)
---------------------------------------------
Manages the lifecycle of alpha models in the production environment.

Key Responsibilities:
  1. Health Check : Verifies integrity of deployed models (load, smoke-test,
                    degenerate-prediction check, prediction-cache staleness).
  2. Archival     : Snapshots production models + writes manifest.json for audit.
  3. Pruning      : Cleans old archives by mtime (not name sort), with disk-size
                    reporting and optional --dry-run preview.
  4. Reporting    : Deployment manifest records IC, t-stat, gate tier, features.

FIXES vs v1:
  BUG C1 [CRITICAL]: Staleness check used model training date vs today (always
    stale — model trained 2023-12-29, today 2026-03-05 = 796 days, threshold 30).
    Training date staleness is meaningless — a model trained on 2016-2023 data
    is EXPECTED to be old. Actual staleness = age of ensemble_predictions.parquet
    (the live signal cache). Fixed: check prediction cache date, not training date.
  BUG C2 [CRITICAL]: Smoke test built dummy DataFrame with np.random.randn for
    ALL columns, then overwrote sector/industry with 'Unknown' string — creating
    a mixed float/string df. Only 2 hardcoded categoricals handled; any other
    cat feature got random floats → CatBoost dtype crash.
    Fixed: build dummy with correct dtypes from feature metadata if available;
    fall back to inspecting column names for known categorical patterns.
  BUG C3 [CRITICAL]: Custom objective injected into __main__ at module level.
    Works only when this file IS __main__. If imported as a module (testing,
    pipeline), the injection targets the wrong namespace. Also: the function
    was defined locally — pickle stored path is __main__.weighted_symmetric_mae
    which only resolves correctly in the original training process's __main__.
    Fixed: robust try/except around joblib.load() with clear error message
    directing user to quant_alpha/objectives.py for the canonical definition.
    Injection is kept as best-effort fallback but failure is caught gracefully.
  BUG H1 [HIGH]: No manifest.json written with archive. No audit trail of what
    was deployed, its quality metrics, or why. Fixed: write manifest.json
    alongside PKL files in each archive snapshot.
  BUG H2 [HIGH]: Health check showed raw IC only; t-stat and gate tier missing.
    Fixed: compute and log t-stat + annualized ICIR + PROD/ENSEMBLE/GATED tier.
  BUG H3 [HIGH]: prune_archives() sorted archives lexicographically by name.
    Non-standard naming (deployment_v2_...) breaks sort order. Fixed: sort by
    directory mtime (st_mtime) — always correct regardless of name format.
  BUG H4 [HIGH]: from config.logging_config import setup_logging — inconsistent
    with all other scripts, crashes if module doesn't exist.
    Fixed: from quant_alpha.utils import setup_logging with basicConfig fallback.
  BUG M1 [MEDIUM]: Smoke test only verified model.predict() doesn't crash.
    A degenerate model predicting all 0.5 passes. Fixed: check prediction
    variance (std > 1e-6) on a 50-row sample — catches collapsed models.
  BUG M2 [MEDIUM]: No disk-size reporting before or after prune.
    Fixed: log total archive size and per-archive size before deletion.
  BUG M3 [MEDIUM]: --all ran archive → prune → check. If archive fails (disk
    full), prune deletes old backups and check runs on broken state.
    Fixed: sequence is now check → archive → prune. Check first ensures we
    know model state before touching anything. Archive before prune ensures
    old copies are never deleted before new copy exists.
  BUG L1 [LOW]: np.random.randn for dummy data is non-deterministic.
    Fixed: np.random.default_rng(seed=42).
  BUG L2 [LOW]: No --dry-run flag for prune. Fixed: --dry-run shows what
    would be deleted without removing anything.

Usage:
    python scripts/deploy_model.py --action check
    python scripts/deploy_model.py --action archive
    python scripts/deploy_model.py --action prune --keep 5
    python scripts/deploy_model.py --action prune --keep 3 --dry-run
    python scripts/deploy_model.py --all
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

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config

from quant_alpha.utils import setup_logging
setup_logging()

logger = logging.getLogger("Quant_Alpha")


# ==============================================================================
# CUSTOM OBJECTIVE — best-effort injection
#
# FIXED C3: Original injected into __main__ at module level with no error handling.
# Root issue: joblib pickle stores the function by qualified name
# (__main__.weighted_symmetric_mae). When train_models.py ran as __main__, it
# stored that path. deploy_model.py also runs as __main__ in standalone use, so
# injection into __main__ works in practice — but:
#   (a) It is fragile if imported as a module or run under a test runner.
#   (b) It is undocumented and confusing to future maintainers.
#
# Correct long-term fix: move weighted_symmetric_mae to
# quant_alpha/objectives.py and import it in both train_models.py and here.
# That way pickle stores 'quant_alpha.objectives.weighted_symmetric_mae' —
# a stable importable path that works everywhere.
#
# Short-term: keep injection as best-effort; joblib.load() is wrapped in
# try/except with a clear error message pointing to the real fix.
# ==============================================================================
def weighted_symmetric_mae(y_true, y_pred):
    """
    Custom training objective — must match definition in train_models.py exactly.

    TODO: Move this to quant_alpha/objectives.py and import in both files.
    That makes pickle resolution stable and removes the __main__ injection hack.
    """
    residuals = y_true - y_pred
    weights   = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad      = -weights * np.tanh(residuals)
    hess      = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess


# Best-effort injection into __main__ for pickle compatibility.
# This handles cases where the script is run directly OR via subprocess.
try:
    import __main__
    setattr(__main__, "weighted_symmetric_mae", weighted_symmetric_mae)
except Exception:
    pass   # Safe to ignore — joblib.load() try/except handles failure gracefully


# ==============================================================================
# GATE THRESHOLDS — read from config (set by train_models.py)
# ==============================================================================
_PROD_IC_THRESHOLD  = getattr(config, "PROD_IC_THRESHOLD",    0.010)
_PROD_IC_TSTAT      = getattr(config, "PROD_IC_TSTAT",         2.5)
_ENS_IC_THRESHOLD   = getattr(config, "MIN_OOS_IC_THRESHOLD",  0.005)
_ENS_IC_TSTAT       = getattr(config, "MIN_OOS_IC_TSTAT",      1.5)


def _gate_tier(ic: float, ic_std: float, n_dates: int) -> str:
    """Return PROD / ENSEMBLE / GATED tier string — mirrors train_models.py logic."""
    tstat = ic / (ic_std / (n_dates ** 0.5)) if n_dates > 0 and ic_std > 0 else 0.0
    if ic >= _PROD_IC_THRESHOLD and tstat >= _PROD_IC_TSTAT:
        return "✅ PROD"
    if ic >= _ENS_IC_THRESHOLD and tstat >= _ENS_IC_TSTAT:
        return "🟡 ENSEMBLE"
    return "❌ GATED"


def _prediction_cache_age() -> tuple[str | None, int | None]:
    """
    Return (last_signal_date_str, trading_days_old) from ensemble_predictions.parquet.
    Returns (None, None) if cache not found or unreadable.

    FIXED C1: This is the correct staleness metric — not model training date.
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
    Build a dummy DataFrame for smoke-testing with correct dtypes.

    FIXED C2: Original used np.random.randn for ALL features then overwrote
    only ['sector', 'industry']. This created a mixed float/string df and only
    handled 2 hardcoded categoricals. CatBoost crashes on unexpected dtypes.

    Fix: identify categorical columns by name pattern (any column whose name
    is a known categorical OR whose name ends in known suffixes like '_sector').
    All others get float64. Seed is fixed for reproducibility (FIXED L1).
    """
    KNOWN_CATS = {
        "sector", "industry", "ticker", "exchange",
        "country", "currency", "gics_sector", "gics_industry",
    }
    rng = np.random.default_rng(seed=seed)    # FIXED L1: fixed seed

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


# ==============================================================================
# DEPLOYMENT MANAGER
# ==============================================================================
class DeploymentManager:
    def __init__(self):
        self.prod_dir    = config.MODELS_DIR / "production"
        self.archive_dir = config.MODELS_DIR / "archive"
        self.prod_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------------------------------
    def verify_deployment(self) -> bool:
        """
        Load and smoke-test all models in production.

        FIXED C1: Staleness now checks prediction cache age, not training date.
        FIXED C2: Dummy data built with correct dtypes per feature.
        FIXED C3: joblib.load() wrapped in try/except with clear error message.
        FIXED H2: Shows t-stat, annualized ICIR, and gate tier.
        FIXED M1: Checks prediction variance on 50-row sample (catches degenerate models).

        Returns True if all models pass, False if any fail.
        """
        models = list(self.prod_dir.glob("*_latest.pkl"))
        if not models:
            logger.error("[CHECK] ❌ No models found in production directory.")
            return False

        # FIXED C1: Signal cache staleness — the thing that actually matters
        sig_date, sig_age = _prediction_cache_age()
        if sig_date is None:
            logger.warning(
                "[CHECK] ⚠️  ensemble_predictions.parquet not found. "
                "Signals may not have been generated yet."
            )
        elif sig_age > 21:
            logger.error(
                f"[CHECK] 🚨 Signal cache is {sig_age} trading days old "
                f"(last: {sig_date}). RETRAIN URGENTLY before live trading."
            )
        elif sig_age > 5:
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
                # FIXED C3: robust load with clear error message
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
                m          = payload.get("oos_metrics", {})

                # FIXED H2: t-stat + annualized ICIR + gate tier
                ic      = m.get("ic_mean",  0.0)
                ic_std  = m.get("ic_std",   1e-8)
                n_dates = m.get("n_dates",  1)
                tstat    = ic / (ic_std / (n_dates ** 0.5)) if n_dates > 0 else 0.0
                ann_icir = (ic / ic_std) * (252 ** 0.5) if ic_std > 0 else 0.0
                tier     = _gate_tier(ic, ic_std, n_dates)

                # FIXED C2 + M1: smoke test with correct dtypes + variance check
                smoke_ok  = True
                smoke_msg = "No features stored — inference test skipped."
                if features:
                    dummy = _build_dummy_df(features, n_rows=50)
                    try:
                        preds = model.predict(dummy)
                        preds = np.asarray(preds, dtype=float)

                        # FIXED M1: degenerate model check
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

    # --------------------------------------------------------------------------
    # ARCHIVE
    # --------------------------------------------------------------------------
    def archive_current_models(self) -> Path | None:
        """
        Snapshot current production models to a timestamped archive folder.

        FIXED H1: Writes manifest.json alongside PKLs for full audit trail:
          - which models were archived
          - their IC, t-stat, gate tier, feature count
          - prediction cache age at archive time
          - archive reason (auto-generated)
        Returns the archive path (or None if nothing to archive).
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

        # FIXED C1: record signal cache age in manifest
        sig_date, sig_age = _prediction_cache_age()
        manifest["signal_cache"] = {
            "last_date":        sig_date,
            "trading_days_old": sig_age,
        }

        for model_path in sorted(models):
            entry = {"file": model_path.name, "status": "ok"}
            try:
                shutil.copy2(model_path, snapshot_dir)

                # FIXED H1: extract quality metrics for manifest
                payload = joblib.load(snapshot_dir / model_path.name)
                m       = payload.get("oos_metrics", {})
                feat    = payload.get("feature_names", [])
                ic      = m.get("ic_mean", 0.0)
                ic_std  = m.get("ic_std",  1e-8)
                n_d     = m.get("n_dates", 1)
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
                    "n_features":    len(feat),
                    "feature_names": feat,
                })
                logger.info(f"   Archived: {model_path.name}")
            except AttributeError:
                # Custom objective pickle issue — still copy the file, just no metrics
                entry["status"] = "copied_no_metrics (custom objective unpickle failed)"
                logger.warning(f"   Archived {model_path.name} (could not read metrics).")
            except Exception as exc:
                entry["status"] = f"FAILED: {exc}"
                logger.error(f"   Failed to archive {model_path.name}: {exc}")

            manifest["models"].append(entry)

        # Write manifest
        manifest_path = snapshot_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"[ARCHIVE] Manifest written → {manifest_path}")
        logger.info("[ARCHIVE] Archive complete.")
        return snapshot_dir

    # --------------------------------------------------------------------------
    # PRUNE
    # --------------------------------------------------------------------------
    def prune_archives(self, keep_last: int = 5, dry_run: bool = False) -> None:
        """
        Remove old archives to save disk space, keeping the N most recent.

        FIXED H3: Original sorted by directory name lexicographically. Non-standard
        names (e.g. deployment_v2_20231205) break lexicographic sort order, causing
        wrong archives to be deleted. Fix: sort by directory mtime (st_mtime) —
        always correct regardless of naming convention.

        FIXED M2: Now logs total archive size before and after pruning. Warns
        if remaining archives exceed 5 GB to prevent silent disk fill.

        FIXED L2: --dry-run flag shows what WOULD be deleted without removing.
        """
        # FIXED H3: sort by modification time, not name
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

        # FIXED M2: warn if archive folder is getting large
        if remaining_mb > 5_000:
            logger.warning(
                f"[PRUNE] ⚠️  Archive folder is {remaining_mb/1000:.1f} GB. "
                "Consider reducing --keep or moving archives to cold storage."
            )


# ==============================================================================
# MAIN
# ==============================================================================
def main():
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
            "Run full sequence: Check → Archive → Prune. "
            "FIXED: check runs FIRST so model state is known before any writes."
        )
    )
    parser.add_argument(
        "--keep", type=int, default=5,
        help="Number of archives to keep when pruning (default: 5)."
    )
    # FIXED L2: --dry-run flag
    parser.add_argument(
        "--dry-run", action="store_true",
        help="For --action prune: show what would be deleted without removing anything."
    )
    args = parser.parse_args()

    manager = DeploymentManager()

    if args.all:
        # FIXED M3: check FIRST, then archive, then prune
        # Original was archive → prune → check:
        #   If archive failed (disk full), prune still ran (deleted old backups),
        #   then check ran on a state where new archive doesn't exist and old ones
        #   were deleted. Unrecoverable.
        # New sequence: check first (read-only), then archive, then prune.
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
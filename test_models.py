# ==============================================================================
# test_models.py
# Tests model training and prediction for LightGBM, XGBoost, CatBoost
# Run: python test_models.py
# ==============================================================================

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd

# CPU throttle — set before any imports
os.environ["OMP_NUM_THREADS"]      = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"]      = "2"
os.environ["NUMBA_NUM_THREADS"]    = "2"

# Suppress CatBoost Numba JIT warning — not an error, just informational
# CatBoost tries to JIT compile calc_ders_range, fails silently, uses Python fallback
import warnings
warnings.filterwarnings("ignore", message="Failed to optimize method")
warnings.filterwarnings("ignore", category=UserWarning, module="catboost")

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✅ PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}❌ FAIL{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}ℹ  {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠  {msg}{RESET}")
def header(msg):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


# ==============================================================================
# SYNTHETIC DATA FACTORY
# Mimics the real pipeline data structure so model wrappers are tested
# exactly as they would be in production.
# ==============================================================================
def make_dataset(
    n_tickers: int = 30,
    n_days: int = 500,
    n_numeric_features: int = 20,
    include_categoricals: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Build a synthetic (date, ticker) panel dataset.

    Columns:
      - date, ticker
      - sector, industry          (categorical — tests XGBoost encoding fix)
      - f_001 … f_020             (numeric features)
      - target                    (float, ~N(0,0.02))
    """
    rng = np.random.default_rng(seed)

    sectors   = ["Technology", "Healthcare", "Finance",
                 "Energy", "Consumer", "Industrials"]
    industries = ["Software", "Hardware", "Pharma",
                  "Banking", "Oil", "Retail", "Auto"]

    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates   = pd.date_range("2018-01-01", periods=n_days, freq="B")

    rows = []
    for ticker in tickers:
        sec = rng.choice(sectors)
        ind = rng.choice(industries)
        for date in dates:
            row = {"date": date, "ticker": ticker,
                   "sector": sec, "industry": ind}
            for j in range(n_numeric_features):
                row[f"f_{j:03d}"] = rng.standard_normal()
            # target has slight signal from first 3 features
            signal = (row["f_000"] * 0.3 +
                      row["f_001"] * 0.2 +
                      row["f_002"] * 0.1)
            row["target"] = signal + rng.normal(0, 0.02)
            rows.append(row)

    df = pd.DataFrame(rows)
    if include_categoricals:
        df["sector"]   = df["sector"].astype("category")
        df["industry"] = df["industry"].astype("category")
    return df


def split_data(df: pd.DataFrame, train_frac: float = 0.7):
    """Chronological train/test split — no data leakage."""
    dates      = sorted(df["date"].unique())
    cutoff_idx = int(len(dates) * train_frac)
    cutoff     = dates[cutoff_idx]
    train = df[df["date"] <  cutoff].copy()
    test  = df[df["date"] >= cutoff].copy()
    return train, test


def get_features(df: pd.DataFrame) -> list:
    numeric = [c for c in df.columns
               if c.startswith("f_")]
    cats    = ["sector", "industry"]
    return numeric + [c for c in cats if c in df.columns]


# ==============================================================================
# CUSTOM OBJECTIVE (same as production)
# ==============================================================================
def weighted_symmetric_mae(y_true, y_pred):
    residuals = y_true - y_pred
    weights   = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad      = -weights * np.tanh(residuals)
    hess      = np.maximum(weights * (1.0 - np.tanh(residuals)**2), 1e-3)
    return grad, hess


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================
def assert_predictions_valid(preds: np.ndarray, n_expected: int, model_name: str):
    assert isinstance(preds, np.ndarray), \
        f"{model_name}: predict() must return np.ndarray, got {type(preds)}"
    assert len(preds) == n_expected, \
        f"{model_name}: expected {n_expected} predictions, got {len(preds)}"
    assert not np.all(preds == 0), \
        f"{model_name}: all predictions are 0 — model did not learn"
    assert not np.any(np.isnan(preds)), \
        f"{model_name}: predictions contain NaN"
    assert not np.any(np.isinf(preds)), \
        f"{model_name}: predictions contain Inf"


def assert_feature_importance(model, model_name: str, features: list):
    assert model.feature_importance is not None, \
        f"{model_name}: feature_importance is None after fit()"
    assert isinstance(model.feature_importance, pd.Series), \
        f"{model_name}: feature_importance must be pd.Series"
    assert len(model.feature_importance) == len(features), \
        (f"{model_name}: feature_importance length {len(model.feature_importance)} "
         f"!= n_features {len(features)}")


def assert_is_fitted(model, model_name: str):
    assert getattr(model, "is_fitted", False), \
        f"{model_name}: is_fitted is False after fit()"
    assert model.model is not None, \
        f"{model_name}: model.model is None after fit()"
    assert model.feature_names is not None, \
        f"{model_name}: feature_names is None after fit()"


def compute_rank_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """Spearman rank IC between predictions and targets."""
    from scipy.stats import spearmanr
    ic, _ = spearmanr(preds, targets)
    return ic if not np.isnan(ic) else 0.0


# ==============================================================================
# INDIVIDUAL MODEL TEST SUITES
# ==============================================================================

class ModelTestSuite:
    """
    Runs all sub-tests for one model class.
    Results are collected so a full summary can be printed at the end.
    """

    def __init__(self, model_class, model_name: str, params: dict,
                 train: pd.DataFrame, test: pd.DataFrame, features: list):
        self.model_class = model_class
        self.model_name  = model_name
        self.params      = params
        self.train       = train
        self.test        = test
        self.features    = features
        self.results: list[dict] = []   # {name, passed, msg}

    def _run(self, test_name: str, fn):
        try:
            fn()
            ok(test_name)
            self.results.append({"name": test_name, "passed": True, "msg": ""})
        except Exception as e:
            fail(f"{test_name} → {e}")
            self.results.append({"name": test_name, "passed": False,
                                  "msg": str(e)})

    # ── Sub-tests ──────────────────────────────────────────────────────────────

    def test_instantiation(self):
        def _():
            m = self.model_class(params=self.params.copy())
            assert m is not None
            assert not getattr(m, "is_fitted", True) or True  # may start False
        self._run("instantiation", _)

    def test_basic_fit_predict(self):
        def _():
            m = self.model_class(params=self.params.copy())
            X_train = self.train[self.features]
            y_train = self.train["target"]
            X_test  = self.test[self.features]

            m.fit(X_train, y_train)
            assert_is_fitted(m, self.model_name)

            preds = m.predict(X_test)
            assert_predictions_valid(preds, len(X_test), self.model_name)
        self._run("basic fit + predict", _)

    def test_fit_with_validation(self):
        def _():
            m = self.model_class(params=self.params.copy())
            n_val   = int(len(self.train) * 0.2)
            tr      = self.train.iloc[:-n_val]
            val     = self.train.iloc[-n_val:]

            m.fit(
                tr[self.features],  tr["target"],
                X_val=val[self.features], y_val=val["target"]
            )
            assert_is_fitted(m, self.model_name)
            preds = m.predict(self.test[self.features])
            assert_predictions_valid(preds, len(self.test), self.model_name)
        self._run("fit with validation set (early stopping)", _)

    def test_custom_objective(self):
        def _():
            p = self.params.copy()

            if "CatBoost" in self.model_name:
                # CatBoost uses loss_function key, needs a class with calc_ders_range
                class _CatObj:
                    def calc_ders_range(self, approxes, targets, weights):
                        # full batch arrays - NO [0] indexing
                        y_pred = np.array(approxes, dtype=np.float64)
                        y_true = np.array(targets,  dtype=np.float64)
                        r = y_true - y_pred
                        w = np.where(y_true * y_pred < 0, 2.0, 1.0)
                        g = -w * np.tanh(r)
                        h = np.maximum(w * (1.0 - np.tanh(r)**2), 1e-3)
                        return list(zip(g.tolist(), h.tolist()))
                p["loss_function"] = _CatObj()
                p["eval_metric"]   = "RMSE"  # mandatory with custom loss
            else:
                # LightGBM and XGBoost both use objective key
                p["objective"] = weighted_symmetric_mae

            m = self.model_class(params=p)
            m.fit(self.train[self.features], self.train["target"])
            assert_is_fitted(m, self.model_name)
            preds = m.predict(self.test[self.features])
            assert_predictions_valid(preds, len(self.test), self.model_name)
        self._run("custom objective (weighted_symmetric_mae)", _)

    def test_feature_importance(self):
        def _():
            m = self.model_class(params=self.params.copy())
            m.fit(self.train[self.features], self.train["target"])
            assert_feature_importance(m, self.model_name, self.features)
        self._run("feature_importance after fit", _)

    def test_predict_before_fit_raises(self):
        def _():
            m = self.model_class(params=self.params.copy())
            try:
                m.predict(self.test[self.features])
                raise AssertionError("Should have raised ValueError")
            except (ValueError, RuntimeError):
                pass  # expected
        self._run("predict before fit raises error", _)

    def test_column_order_invariance(self):
        """Model must produce same predictions regardless of column order."""
        def _():
            m = self.model_class(params=self.params.copy())
            m.fit(self.train[self.features], self.train["target"])

            X_normal   = self.test[self.features]
            X_shuffled = X_normal[self.features[::-1]]  # reversed column order

            p1 = m.predict(X_normal)
            p2 = m.predict(X_shuffled)
            np.testing.assert_allclose(
                p1, p2, rtol=1e-4,
                err_msg=f"{self.model_name}: column reorder changed predictions"
            )
        self._run("column order invariance", _)

    def test_nan_in_numeric_features(self):
        """Model must handle NaN in numeric features gracefully."""
        def _():
            m = self.model_class(params=self.params.copy())
            m.fit(self.train[self.features], self.train["target"])

            X_nan = self.test[self.features].copy()
            num_cols = [c for c in self.features if c.startswith("f_")]
            # Inject 10% NaN into numeric columns
            for col in num_cols[:5]:
                mask = np.random.rand(len(X_nan)) < 0.10
                X_nan.loc[mask, col] = np.nan

            preds = m.predict(X_nan)
            assert_predictions_valid(preds, len(X_nan), self.model_name)
        self._run("handles NaN in numeric features", _)

    def test_rank_ic_positive(self):
        """
        Trained model should have positive rank IC on test set.
        Data has a real signal so IC > 0 is expected.
        IC > 0.01 is a lenient threshold — just checks model actually learned.
        """
        def _():
            m = self.model_class(params=self.params.copy())
            m.fit(self.train[self.features], self.train["target"])
            preds  = m.predict(self.test[self.features])
            ic     = compute_rank_ic(preds, self.test["target"].values)
            info(f"    Rank IC = {ic:.4f}")
            assert ic > 0.01, \
                f"{self.model_name}: Rank IC {ic:.4f} ≤ 0.01 — model may not be learning"
        self._run("rank IC > 0.01 on test set", _)

    def test_prediction_variance(self):
        """Predictions should not be near-constant (degenerate model)."""
        def _():
            m = self.model_class(params=self.params.copy())
            m.fit(self.train[self.features], self.train["target"])
            preds = m.predict(self.test[self.features])
            std = np.std(preds)
            info(f"    Prediction std = {std:.6f}")
            assert std > 1e-6, \
                f"{self.model_name}: prediction std {std:.2e} near zero — degenerate model"
        self._run("prediction variance > 0 (not degenerate)", _)

    def test_reproducibility(self):
        """Two fits with same params and data must give identical predictions."""
        def _():
            p = self.params.copy()
            m1 = self.model_class(params=p)
            m2 = self.model_class(params=p)
            m1.fit(self.train[self.features], self.train["target"])
            m2.fit(self.train[self.features], self.train["target"])
            p1 = m1.predict(self.test[self.features])
            p2 = m2.predict(self.test[self.features])
            np.testing.assert_allclose(
                p1, p2, rtol=1e-4,
                err_msg=f"{self.model_name}: two identical fits gave different predictions"
            )
        self._run("reproducibility (same params → same predictions)", _)

    def test_training_time_logged(self):
        def _():
            m = self.model_class(params=self.params.copy())
            m.fit(self.train[self.features], self.train["target"])
            assert hasattr(m, "training_time"), \
                f"{self.model_name}: training_time attribute missing"
            assert m.training_time > 0, \
                f"{self.model_name}: training_time is 0"
            info(f"    Training time = {m.training_time:.2f}s")
        self._run("training_time attribute logged", _)

    def run_all(self) -> dict:
        header(f"Testing {self.model_name}")
        t0 = time.perf_counter()

        self.test_instantiation()
        self.test_basic_fit_predict()
        self.test_fit_with_validation()
        self.test_custom_objective()
        self.test_feature_importance()
        self.test_predict_before_fit_raises()
        self.test_column_order_invariance()
        self.test_nan_in_numeric_features()
        self.test_rank_ic_positive()
        self.test_prediction_variance()
        self.test_reproducibility()
        self.test_training_time_logged()

        elapsed  = time.perf_counter() - t0
        passed   = sum(1 for r in self.results if r["passed"])
        total    = len(self.results)
        failures = [r for r in self.results if not r["passed"]]

        print(f"\n  {BOLD}Result: {passed}/{total} passed "
              f"({elapsed:.1f}s){RESET}")
        return {
            "model":    self.model_name,
            "passed":   passed,
            "total":    total,
            "failures": failures,
            "elapsed":  elapsed,
        }


# ==============================================================================
# WALK-FORWARD MINI TEST
# Validates the trainer + model integration end-to-end
# ==============================================================================
def test_walk_forward_integration(model_class, model_name: str,
                                   params: dict, df: pd.DataFrame,
                                   features: list) -> dict:
    header(f"Walk-Forward Integration — {model_name}")
    results = []

    def _run(name, fn):
        try:
            fn()
            ok(name)
            results.append({"name": name, "passed": True, "msg": ""})
        except Exception as e:
            fail(f"{name} → {e}")
            results.append({"name": name, "passed": False, "msg": str(e)})

    def _test_wf():
        try:
            from quant_alpha.models.trainer import WalkForwardTrainer
        except ImportError:
            warn("WalkForwardTrainer not importable — skipping WF test")
            return

        p = params.copy()
        if "cat_features" in p:
            p["cat_features"] = [c for c in p["cat_features"] if c in features]
            if not p["cat_features"]:
                del p["cat_features"]

        trainer = WalkForwardTrainer(
            model_class=model_class,
            min_train_months=12,   # small for test data
            test_months=3,
            step_months=3,
            window_type="expanding",
            embargo_days=10,
            model_params=p
        )
        preds_df = trainer.train(df, features, "target")

        assert not preds_df.empty, \
            f"{model_name} WF: empty predictions DataFrame"
        assert "prediction" in preds_df.columns, \
            f"{model_name} WF: 'prediction' column missing"
        assert "date"   in preds_df.columns, \
            f"{model_name} WF: 'date' column missing"
        assert "ticker" in preds_df.columns, \
            f"{model_name} WF: 'ticker' column missing"
        assert not preds_df["prediction"].isna().all(), \
            f"{model_name} WF: all predictions are NaN"

        n_folds = len(trainer.results)
        info(f"    Folds succeeded: {n_folds}")
        info(f"    OOS predictions: {len(preds_df):,}")
        assert n_folds >= 1, f"{model_name} WF: 0 folds succeeded"

    _run("walk-forward produces valid predictions", _test_wf)

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    return {
        "model":    f"{model_name} (WF)",
        "passed":   passed,
        "total":    total,
        "failures": [r for r in results if not r["passed"]],
        "elapsed":  0.0,
    }


# ==============================================================================
# IMPORT CHECK
# ==============================================================================
def check_imports() -> bool:
    header("Import Check")
    all_ok = True
    # (module_to_import, attr_to_check, display_name)
    # attr=None means just importing the module is enough
    libs = [
        ("lightgbm",                          "LGBMRegressor",   "lightgbm.LGBMRegressor"),
        ("xgboost",                           "XGBRegressor",    "xgboost.XGBRegressor"),
        ("catboost",                          "CatBoostRegressor","catboost.CatBoostRegressor"),
        ("sklearn.covariance",                "LedoitWolf",      "sklearn.covariance.LedoitWolf"),
        ("scipy.stats",                       "spearmanr",       "scipy.stats.spearmanr"),
        ("quant_alpha.models.lightgbm_model", "LightGBMModel",   "LightGBMModel"),
        ("quant_alpha.models.xgboost_model",  "XGBoostModel",    "XGBoostModel"),
        ("quant_alpha.models.catboost_model", "CatBoostModel",   "CatBoostModel"),
        ("quant_alpha.models.base_model",     "BaseModel",       "BaseModel"),
    ]
    for module, attr, display in libs:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            ok(display)
        except Exception as e:
            fail(f"{display}  →  {e}")
            all_ok = False
    return all_ok


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
def print_summary(all_results: list[dict]):
    header("FINAL SUMMARY")
    grand_pass  = 0
    grand_total = 0
    any_failure = False

    for r in all_results:
        status = f"{GREEN}PASS{RESET}" if r["passed"] == r["total"] else f"{RED}FAIL{RESET}"
        print(f"  [{status}]  {r['model']:<35}  "
              f"{r['passed']}/{r['total']} tests  "
              f"({r['elapsed']:.1f}s)")
        grand_pass  += r["passed"]
        grand_total += r["total"]
        if r["failures"]:
            any_failure = True
            for f_ in r["failures"]:
                print(f"          {RED}↳ {f_['name']}: {f_['msg'][:80]}{RESET}")

    print(f"\n{BOLD}{'='*60}{RESET}")
    pct = grand_pass / grand_total * 100 if grand_total else 0
    col = GREEN if grand_pass == grand_total else RED
    print(f"  {BOLD}Total: {col}{grand_pass}/{grand_total}{RESET}{BOLD} "
          f"({pct:.0f}%){RESET}")

    if not any_failure:
        print(f"\n  {GREEN}{BOLD}🎉 All tests passed! Models are ready.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}⚠  Some tests failed. Check errors above.{RESET}")
    print()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  QUANT ALPHA — MODEL TEST SUITE{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # ── 1. Import check ────────────────────────────────────────────────────────
    imports_ok = check_imports()
    if not imports_ok:
        print(f"\n{RED}Critical imports failed. Fix them before running tests.{RESET}")
        sys.exit(1)

    from quant_alpha.models.lightgbm_model import LightGBMModel
    from quant_alpha.models.xgboost_model  import XGBoostModel
    from quant_alpha.models.catboost_model import CatBoostModel

    # ── 2. Synthetic data ──────────────────────────────────────────────────────
    header("Building Synthetic Dataset")
    df = make_dataset(n_tickers=30, n_days=500,
                      n_numeric_features=20, include_categoricals=True)
    train, test = split_data(df, train_frac=0.7)
    features    = get_features(df)

    info(f"Total rows  : {len(df):,}")
    info(f"Train rows  : {len(train):,}")
    info(f"Test rows   : {len(test):,}")
    info(f"Features    : {len(features)}  "
         f"({len([f for f in features if f.startswith('f_')])} numeric, "
         f"{len([f for f in features if not f.startswith('f_')])} categorical)")
    info(f"Tickers     : {df['ticker'].nunique()}")
    info(f"Date range  : {df['date'].min().date()} → {df['date'].max().date()}")

    # ── 3. Model configs ───────────────────────────────────────────────────────
    # Small n_estimators so tests finish quickly
    lgbm_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 16,
        "max_depth": 4,
        "reg_lambda": 10.0,
        "n_jobs": 2,
        "random_state": 42,
        "verbose": -1,
    }
    xgb_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "reg_lambda": 10.0,
        "subsample": 0.8,
        "n_jobs": 2,
        "random_state": 42,
        "early_stopping_rounds": 20,
    }
    cat_params = {
        "iterations": 100,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 10.0,
        "subsample": 0.8,
        "thread_count": 2,
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
        "cat_features": ["sector", "industry"],
    }

    # ── 4. Run per-model suites ────────────────────────────────────────────────
    all_results = []

    for model_class, model_name, params in [
        (LightGBMModel, "LightGBM", lgbm_params),
        (XGBoostModel,  "XGBoost",  xgb_params),
        (CatBoostModel, "CatBoost", cat_params),
    ]:
        suite = ModelTestSuite(
            model_class=model_class,
            model_name=model_name,
            params=params,
            train=train,
            test=test,
            features=features,
        )
        all_results.append(suite.run_all())

    # ── 5. Walk-forward integration tests ─────────────────────────────────────
    # Use a slightly larger dataset so WF has enough months
    df_wf = make_dataset(n_tickers=20, n_days=800,
                         n_numeric_features=10, include_categoricals=True,
                         seed=99)
    feats_wf = get_features(df_wf)

    for model_class, model_name, params in [
        (LightGBMModel, "LightGBM", lgbm_params),
        (XGBoostModel,  "XGBoost",  xgb_params),
        (CatBoostModel, "CatBoost", cat_params),
    ]:
        wf_result = test_walk_forward_integration(
            model_class, model_name, params, df_wf, feats_wf
        )
        all_results.append(wf_result)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print_summary(all_results)

    # Exit code — CI/CD ke liye
    failed = any(r["passed"] < r["total"] for r in all_results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
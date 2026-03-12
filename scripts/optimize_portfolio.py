"""
Production Portfolio Construction Engine
========================================
Transforms raw alpha signals into optimal executable orders using modern portfolio theory (MPT).

Purpose
-------
This module serves as the **Portfolio Construction Layer**, bridging the gap between
predictive modeling (Alpha) and execution (Orders). It solves the convex optimization
problem:
.. math::
    \\max_{w} \\mu^T w - \\frac{\\lambda}{2} w^T \\Sigma w
Subject to constraints (leverage, cardinality, sector exposure).

Key capabilities:
1.  **Risk Modeling**: Estimates the covariance matrix $\\Sigma$ via Ledoit-Wolf shrinkage.
2.  **Objective Functions**: Supports Mean-Variance, Risk Parity, Black-Litterman, and Kelly Criterion.
3.  **Volatility Targeting**: Dynamically scales exposure to maintain constant portfolio risk $\\sigma_{target}$.
4.  **Order Generation**: Discretizes optimal weights into integer share counts, accounting for
    Close(T) vs Open(T+1) price drift.

Usage:
------
Executed via CLI for daily portfolio rebalancing.

.. code-block:: bash

    # Standard Mean-Variance Optimization ($1M Capital)
    python scripts/optimize_portfolio.py --capital 1000000 --method mean_variance

    # Risk Parity (Equal Risk Contribution)
    python scripts/optimize_portfolio.py --method risk_parity --top-n 30

    # Volatility Targeting ($\sigma = 15\%$)
    python scripts/optimize_portfolio.py --method mean_variance --target-vol 0.15

    python scripts/optimize_portfolio.py --method black_litterman

Importance
----------
-   **Risk Control**: Ensures the portfolio remains within defined volatility and drawdown limits.
-   **Diversification**: Mitigates idiosyncratic risk via HHI concentration checks.
-   **Alpha Conversion**: Translates rank-based signals into dollar-neutral or long-only allocations.

Tools & Frameworks
------------------
-   **Scikit-Learn**: Ledoit-Wolf covariance estimation for robust risk modeling.
-   **Pandas/NumPy**: Vectorized matrix operations for efficient frontier calculation.
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime, date

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging
from quant_alpha.utils import load_parquet, calculate_returns
from quant_alpha.optimization.allocator import PortfolioAllocator

setup_logging()
logger = logging.getLogger("Quant_Alpha")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.covariance")

# --- Configuration ---
PREDICTIONS_DIR = config.RESULTS_DIR / "predictions"
PRICES_DIR      = config.CACHE_DIR / "master_data_with_factors.parquet"
OUTPUT_DIR      = config.RESULTS_DIR / "orders"

DEFAULT_LOOKBACK_DAYS = getattr(config, "OPT_LOOKBACK_DAYS", 252)
RISK_FREE_RATE        = getattr(config, "RISK_FREE_RATE", 0.04)

# Data Buffer:
# Ensures sufficient calendar days are loaded to cover weekends + holidays
# (~45 calendar days ≈ 30 trading days buffer beyond lookback)
CALENDAR_BUFFER_DAYS = 45

# Risk-Based Methods: Pure risk diversification (Alpha scores ignored for weighting)
RISK_ONLY_METHODS = {"risk_parity", "inverse_vol"}

# Leverage Constraint: Maximum Gross Exposure ($|L| + |S|$).
# 1.0 = Fully funded (Long-Only or 130/30 Net).
# >1.0 requires margin facilities and incurs funding costs.
MAX_LEVERAGE = getattr(config, "MAX_LEVERAGE", 1.0)


class ProductionOptimizer:
    def __init__(self, capital: float, method: str = "mean_variance",
                 top_n: int = 25, target_vol: float = 0.15):
        self.capital    = capital
        self.method     = method
        self.top_n      = top_n
        self.target_vol = target_vol
        # Strategy Pattern: Use Mean-Variance as the base allocator for custom logic
        allocator_method = "mean_variance" if method == "top_n" else method
        self.allocator  = PortfolioAllocator(
            method=allocator_method,
            risk_aversion=getattr(config, "OPT_RISK_AVERSION", 2.5),
            fraction=getattr(config, "OPT_KELLY_FRACTION", 1.0),
            tau=0.05,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LOAD SIGNALS
    # ──────────────────────────────────────────────────────────────────────────
    def load_latest_predictions(self) -> tuple[pd.DataFrame, date]:
        """Retrieves the most recent alpha signal artifact from the Data Lake."""
        if not PREDICTIONS_DIR.exists():
            raise FileNotFoundError(f"Predictions directory missing: {PREDICTIONS_DIR}")

        files = sorted(PREDICTIONS_DIR.glob("alpha_signals_*.parquet"))
        if not files:
            raise FileNotFoundError(
                "No alpha signal files found. Run generate_predictions.py first."
            )

        latest_file = files[-1]
        logger.info(f"Loading latest signals: {latest_file.name}")

        df = load_parquet(latest_file)

        # Timezone Normalization:
        # Parquet files may store UTC-aware timestamps via PyArrow.
        # If signal_date is local date, tz mismatch → empty DataFrame → crash.
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        try:
            file_date_str = latest_file.stem.replace("alpha_signals_", "")
            signal_date   = datetime.strptime(file_date_str, "%Y-%m-%d").date()
        except ValueError:
            signal_date = df["date"].max().date()

        # Latency Check: Warn if signals are stale ($T_{lag} > 1$ trading day)
        trading_days_old = len(pd.bdate_range(signal_date, datetime.now().date())) - 1
        if trading_days_old > 1:
            logger.warning(
                f"Signals are {trading_days_old} trading day(s) old ({signal_date}). "
                "Market data may have changed since generation."
            )

        latest_df = df[df["date"].dt.date == signal_date].copy()
        if latest_df.empty:
            raise ValueError(f"No data found for date {signal_date} in {latest_file.name}.")

        return latest_df, signal_date

    # ──────────────────────────────────────────────────────────────────────────
    # 2. MARKET DATA
    # ──────────────────────────────────────────────────────────────────────────
    def load_market_data(self, tickers: list[str], end_date: date) -> tuple[pd.DataFrame, dict]:
        """
        Ingests historical pricing for Risk Modeling ($\Sigma$) and Market Caps for
        Black-Litterman Priors ($\Pi$).

        Returns:
            price_matrix : $T \times N$ Close price matrix
            market_caps  : {ticker: cap} — Real values if available,
                           else 1e9 dummy (signals equal market weight in BL prior)
        """
        logger.info(f"Loading market data for {len(tickers)} tickers...")

        if not PRICES_DIR.exists():
            raise FileNotFoundError(f"Master data cache missing: {PRICES_DIR}")

        master_df = load_parquet(PRICES_DIR)
        # Timezone Normalization
        master_df["date"] = pd.to_datetime(master_df["date"]).dt.tz_localize(None)

        # Date Range Slicing
        start_date = pd.Timestamp(end_date) - pd.Timedelta(
            days=DEFAULT_LOOKBACK_DAYS + CALENDAR_BUFFER_DAYS
        )
        mask = (master_df["date"] >= start_date) & (master_df["date"] <= pd.Timestamp(end_date))

        # Schema Validation: Check for 'market_cap' availability
        cols = ["date", "ticker", "close"]
        if "market_cap" in master_df.columns:
            cols.append("market_cap")
        df = master_df.loc[mask, cols].copy()

        price_matrix = df.pivot(index="date", columns="ticker", values="close")

        available_tickers = [t for t in tickers if t in price_matrix.columns]
        missing = set(tickers) - set(available_tickers)
        if missing:
            logger.warning(
                f"Missing price history for {len(missing)} ticker(s): "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. Dropping."
            )

        # Extract Market Caps (Vectorized Last Observation)
        market_caps: dict = {}
        if "market_cap" in df.columns:
            last_caps = df.sort_values("date").groupby("ticker")["market_cap"].last()
            market_caps = last_caps.dropna().to_dict()
            logger.info(f"Market caps loaded for {len(market_caps)} tickers.")
        else:
            logger.debug("market_cap column not in master data — BL will use equal-weight prior.")

        return price_matrix[available_tickers], market_caps

    # ──────────────────────────────────────────────────────────────────────────
    # 3. RISK MODEL
    # ──────────────────────────────────────────────────────────────────────────
    def estimate_risk_model(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """Estimates Annualized Covariance Matrix $\Sigma$ via Ledoit-Wolf Shrinkage."""
        returns   = calculate_returns(price_matrix).dropna(how="all")
        valid_cols = returns.columns[returns.isnull().mean() < 0.3]

        # Data Quality Check: Drop tickers with >30% missing history (IPO/Delisted)
        dropped = set(returns.columns) - set(valid_cols)
        if dropped:
            logger.warning(
                f"Dropped {len(dropped)} ticker(s) from risk model "
                f"(>30% missing history — likely newly listed): "
                f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}"
            )

        returns = returns[valid_cols].fillna(0.0)

        if len(returns) < 60:
            logger.warning("Short history (<60 days). Covariance estimation may be unstable.")

        lw = LedoitWolf()
        try:
            cov_np = lw.fit(returns).covariance_
            return pd.DataFrame(
                cov_np * 252,
                index=returns.columns,
                columns=returns.columns,
            )
        except Exception as exc:
            logger.error(f"Risk model estimation failed: {exc}. Falling back to diagonal.")
            vol = returns.std() * np.sqrt(252)
            return pd.DataFrame(
                np.diag(vol ** 2),
                index=returns.columns,
                columns=returns.columns,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. ORDER GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    def generate_orders(self, weights: dict[str, float],
                        prices: pd.Series) -> pd.DataFrame:
        """
        Discretizes target weights into integer share counts.

        Implementation Note:
            Sizing uses $Close_T$ as a proxy for execution price $Open_{T+1}$.
            This introduces a drift/slippage component of $\approx 0.3-0.8\%$.
            Precise execution requires an intraday execution algorithm.
        """
        logger.info(
            "Share counts based on Close(T). Orders execute at Open(T+1). "
            "Expect ~0.3-0.8% position sizing deviation on fill."
        )

        orders = []
        for ticker, weight in weights.items():
            if abs(weight) < 1e-4:
                continue

            # Validation: Ensure price existence and positivity
            if ticker not in prices.index:
                logger.warning(f"No price found for {ticker}. Skipping.")
                continue

            price = float(prices[ticker])
            if np.isnan(price) or price <= 0:
                logger.warning(f"Invalid price {price:.4f} for {ticker}. Skipping.")
                continue

            target_value = self.capital * weight
            shares = int(target_value / price)

            orders.append({
                "ticker": ticker,
                "weight": round(weight, 4),
                "price":  price,
                "shares": shares,
                "value":  shares * price,
                "side":   "LONG" if shares > 0 else "SHORT",
            })

        return pd.DataFrame(orders).sort_values("weight", ascending=False)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. RISK REPORT
    # ──────────────────────────────────────────────────────────────────────────
    def _print_risk_report(self, weights: dict, cov_matrix: pd.DataFrame,
                           orders_df: pd.DataFrame):
        """Calculates and reports Ex-Ante Portfolio Risk Metrics."""
        tickers = cov_matrix.index.tolist()
        w_vec   = np.array([weights.get(t, 0.0) for t in tickers])

        # Variance Calculation: $\sigma^2 = w^T \Sigma w$
        port_var = float(w_vec.T @ cov_matrix.values @ w_vec)
        port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

        # Exposure Metrics
        gross_exp = orders_df["value"].abs().sum() / self.capital
        net_exp   = orders_df["value"].sum() / self.capital
        
        # Concentration Risk (Herfindahl-Hirschman Index)
        hhi       = float((orders_df["weight"] ** 2).sum())

        top_buy  = orders_df.head(3)
        n_long   = (orders_df["side"] == "LONG").sum()
        n_short  = (orders_df["side"] == "SHORT").sum()

        print("\n" + "=" * 62)
        print("  PORTFOLIO RISK REPORT (Ex-Ante)")
        print("=" * 62)
        print(f"  Target Capital      : ${self.capital:,.0f}")
        print(f"  Gross Exposure      : {gross_exp:.1%}  "
              f"{'⚠️  Leveraged' if gross_exp > MAX_LEVERAGE else '✅ Within bounds'}")
        print(f"  Net Exposure        : {net_exp:.1%}")
        print(f"  Positions           : {len(orders_df)}  "
              f"({n_long} long / {n_short} short)")
        print("-" * 40)
        print(f"  Expected Vol (ann.) : {port_vol:.2%}")
        print(f"  Target Vol          : {self.target_vol:.2%}")
        print(f"  HHI Concentration   : {hhi:.4f}  "
              f"({'Concentrated ⚠️' if hhi > 0.10 else 'Diversified ✅'})")
        if not orders_df.empty:
            top_row = orders_df.iloc[0]
            print(f"  Largest Position    : {top_row['weight']:.2%}  ({top_row['ticker']})")
        print("-" * 40)
        print("  TOP 3 POSITIONS:")
        for _, row in top_buy.iterrows():
            print(f"    {row['ticker']:<6}  {row['weight']:>6.2%}  "
                  f"${row['value']:>9,.0f}  ({row['shares']} sh @ ${row['price']:.2f})")
        print("=" * 62 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        logger.info(f"Starting Portfolio Optimization | Capital: ${self.capital:,.0f} | "
                    f"Method: {self.method} | Top-N: {self.top_n}")

        # ── 1. Load signals ───────────────────────────────────────────────────
        preds_df, signal_date = self.load_latest_predictions()

        # Schema Validation: Ensure alpha score column exists
        if "ensemble_alpha" in preds_df.columns:
            score_col = "ensemble_alpha"
        elif "prediction" in preds_df.columns:
            score_col = "prediction"
        else:
            raise ValueError(
                f"No alpha score column found. Expected 'ensemble_alpha' or "
                f"'prediction'. Available: {preds_df.columns.tolist()}"
            )

        # Universe Selection:
        # For Risk-Only methods (Risk Parity), use Top-N candidates to define the investable
        # universe, even though weights are alpha-agnostic. This ensures we trade liquid,
        # high-quality names.
        if self.method in RISK_ONLY_METHODS:
            candidates = preds_df.sort_values(score_col, ascending=False).head(self.top_n * 2)
            logger.info(
                f"Method '{self.method}': using top-{self.top_n * 2} alpha tickers "
                f"as universe (weights will be risk-driven, not alpha-driven)."
            )
        else:
            candidates = preds_df.sort_values(score_col, ascending=False).head(self.top_n * 2)
            logger.info(f"Top-{self.top_n * 2} alpha candidates selected for optimizer.")

        tickers      = candidates["ticker"].tolist()
        alpha_scores = candidates.set_index("ticker")[score_col].to_dict()
        logger.info(f"Analysing {len(tickers)} candidates from {signal_date}")

        # ── 2. Market data + risk model ───────────────────────────────────────
        price_matrix, loaded_caps = self.load_market_data(tickers, signal_date)
        latest_prices = price_matrix.iloc[-1]  # Close(T) — used for share sizing

        valid_tickers = [t for t in tickers if t in price_matrix.columns]
        price_matrix  = price_matrix[valid_tickers]

        if len(valid_tickers) < 2:
            logger.error("Not enough valid tickers for optimization. Aborting.")
            return

        cov_matrix    = self.estimate_risk_model(price_matrix)
        valid_tickers = cov_matrix.index.tolist()  # may shrink after >30% NaN filter

        if len(valid_tickers) < 2:
            logger.error("Risk model returned fewer than 2 tickers. Aborting.")
            return

        # ── 3. Optimize ───────────────────────────────────────────────────────
        logger.info(f"Optimizing using {self.method}...")

        # Rescale alpha scores to interpretable expected return range for MVO/Kelly/BL.
        # ensemble_alpha is a rank percentile (0→1), not an annualized return.
        # MVO Sharpe maximization is scale-invariant, but having returns near 0
        # causes numerical degeneracy in some optimizers.
        # Rescale: centre at 0, scale to [-MAX_ALPHA_RET, +MAX_ALPHA_RET]
        # (e.g. top-ranked stock → +30% expected, bottom → -30%)
        MAX_ALPHA_RET = getattr(config, "MAX_ALPHA_RET", 0.30)  # annualized
        raw_scores = np.array([alpha_scores.get(t, 0.5) for t in valid_tickers])
        # Normalise rank to [-1, +1] then scale to return range
        normalised = (raw_scores - raw_scores.mean()) / (raw_scores.std() + 1e-8)
        scaled_rets = normalised * MAX_ALPHA_RET
        expected_returns = {t: float(scaled_rets[i]) for i, t in enumerate(valid_tickers)}
        logger.info(
            f"Alpha rescaled: [{scaled_rets.min():.1%}, {scaled_rets.max():.1%}] "
            f"(MAX_ALPHA_RET={MAX_ALPHA_RET:.0%}). Used for MVO/Kelly/BL.")

        # Market Cap Extraction (Required for Black-Litterman Equilibrium)
        market_caps = {}
        for t in valid_tickers:
            cap = loaded_caps.get(t, np.nan)
            market_caps[t] = float(cap) if pd.notna(cap) and cap > 0 else 1e9

        if self.method == "black_litterman":
            real_caps = sum(1 for v in market_caps.values() if v != 1e9)
            if real_caps == 0:
                logger.warning(
                    "Black-Litterman: no real market caps available. "
                    "Using equal-weight (1e9) prior — add 'market_cap' column "
                    "to master data for a proper equilibrium prior."
                )
            else:
                logger.info(f"Black-Litterman: {real_caps}/{len(market_caps)} tickers have real market caps.")

        # View Confidence (Black-Litterman):
        # $\tau = 1.0 \rightarrow$ Full Alpha trust. $\tau = 0.0 \rightarrow$ Market Prior only.
        bl_confidence = getattr(config, "BL_CONFIDENCE_LEVEL", 0.6)

        if self.method == "top_n":
            # Equal weight across top_n tickers — no optimizer needed
            top_tickers = sorted(expected_returns, key=expected_returns.get, reverse=True)[:self.top_n]
            weights = {t: 1.0 / len(top_tickers) for t in top_tickers}
            logger.info(f"top_n: equal weight across {len(top_tickers)} tickers ({1/len(top_tickers):.2%} each).")
        else:
            try:
                weights = self.allocator.allocate(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    market_caps=market_caps,
                    risk_free_rate=RISK_FREE_RATE,
                    confidence_level=bl_confidence,   # used by BL; ignored by other methods
                )
            except Exception as exc:
                logger.error(f"Optimization failed: {exc}. Falling back to equal weight.")
                weights = {t: 1.0 / len(valid_tickers) for t in valid_tickers}

        # ── 3b. Volatility targeting ──────────────────────────────────────────
        # Scale portfolio weights to match target annualized volatility.
        # Formula: $w_{scaled} = w \times \min(\frac{\sigma_{target}}{\sigma_{port}}, 3.0)$
        if self.target_vol > 0 and self.method != "top_n":
            w_vec    = np.array([weights.get(t, 0.0) for t in valid_tickers])
            port_var = float(w_vec.T @ cov_matrix.values @ w_vec)
            port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

            if port_vol > 1e-6:
                scaler = min(self.target_vol / port_vol, 3.0)

                # Constraint: For Risk Parity/Inverse Vol, prevent de-leveraging below 1.0x
                if scaler < 1.0 and self.method in RISK_ONLY_METHODS:
                    logger.info(
                        f"Vol targeting: raw vol={port_vol:.1%} > target={self.target_vol:.1%}. "
                        f"Skipping scale-down for '{self.method}' — "                        f"reduce universe size (--top-n) instead."
                    )
                else:
                    logger.info(
                        f"Vol targeting: scaling weights {scaler:.2f}x "
                        f"(raw vol={port_vol:.1%} → target={self.target_vol:.1%})"
                    )
                    scaled = {t: w * scaler for t, w in weights.items()}

                    # Leverage Constraint Check
                    gross_exp = sum(abs(w) for w in scaled.values())
                    if gross_exp > MAX_LEVERAGE:
                        logger.warning(
                            f"Vol scaling would create {gross_exp:.1%} gross exposure "
                            f"(scaler={scaler:.2f}x) — exceeds MAX_LEVERAGE={MAX_LEVERAGE:.1%}. "
                            f"Capping. To allow leverage, set config.MAX_LEVERAGE > 1.0."
                        )
                        scaled = {t: w / (gross_exp / MAX_LEVERAGE) for t, w in scaled.items()}

                    weights = scaled

        # ── 4. Generate orders ────────────────────────────────────────────────
        orders_df = self.generate_orders(weights, latest_prices)

        # Add signal date so create_report.py can check for staleness
        orders_df["signal_date"] = signal_date

        if orders_df.empty:
            logger.error("No valid orders generated. Check price data and weights.")
            return

        # ── 5. Risk report ────────────────────────────────────────────────────
        self._print_risk_report(weights, cov_matrix, orders_df)

        # ── 6. Save ───────────────────────────────────────────────────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Persistence:
        # 1. Timestamped log (Audit Trail)
        # 2. "latest" pointer (Downstream consumption)
        ts         = datetime.now().strftime("%H%M%S")
        filename   = f"orders_{signal_date}_{ts}.csv"
        latest_fn  = "orders_latest.csv"

        orders_df.to_csv(OUTPUT_DIR / filename,  index=False)
        orders_df.to_csv(OUTPUT_DIR / latest_fn, index=False)

        logger.info(f"Orders saved → {OUTPUT_DIR / filename}")
        logger.info(f"Latest copy  → {OUTPUT_DIR / latest_fn}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Production Portfolio Optimizer v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optimize_portfolio.py --capital 1000000
  python scripts/optimize_portfolio.py --method mean_variance --target-vol 0.15
  python scripts/optimize_portfolio.py --method risk_parity --top-n 30
  python scripts/optimize_portfolio.py --method top_n --top-n 20
        """,
    )
    parser.add_argument(
        "--capital", type=float,
        default=getattr(config, "INITIAL_CAPITAL", 100_000),
        help="Total capital to allocate (default: from config or 100k)",
    )
    parser.add_argument(
        "--method", type=str, default="mean_variance",
        choices=["mean_variance", "risk_parity", "inverse_vol", "top_n", "kelly", "black_litterman"],
        help="Optimization method (default: mean_variance)",
    )
    parser.add_argument(
        "--top-n", dest="top_n", type=int, default=25,
        help="Number of assets to consider/hold (default: 25)",
    )
    parser.add_argument(
        "--target-vol", dest="target_vol", type=float, default=0.15,
        help="Target annualized volatility (default: 0.15)",
    )
    args = parser.parse_args()

    optimizer = ProductionOptimizer(
        capital=args.capital,
        method=args.method,
        top_n=args.top_n,
        target_vol=args.target_vol,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
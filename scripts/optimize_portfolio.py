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

Role in Quantitative Workflow
-----------------------------
Executed downstream of inference (`generate_predictions.py`) to convert point-in-time
cross-sectional signals into discrete share counts. Enforces strict ex-ante risk
controls before capital allocation.

Usage:
------
Executed via CLI for daily portfolio rebalancing.

.. code-block:: bash

    # Standard Mean-Variance Optimization ($1M Capital)
    python scripts/optimize_portfolio.py --capital 1000000 --method mean_variance

    # Risk Parity (Equal Risk Contribution)
    python scripts/optimize_portfolio.py --method risk_parity --top-n 30

    # Volatility Targeting ($\\sigma = 15\\%$)
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

# Static execution constraints and directory bindings
PREDICTIONS_DIR = config.RESULTS_DIR / "predictions"
PRICES_DIR      = config.CACHE_DIR / "master_data_with_factors.parquet"
OUTPUT_DIR      = config.RESULTS_DIR / "orders"

DEFAULT_LOOKBACK_DAYS = getattr(config, "OPT_LOOKBACK_DAYS", 252)
RISK_FREE_RATE        = getattr(config, "RISK_FREE_RATE", 0.04)

# Ensures sufficient calendar days are loaded to cover weekends + holidays
# (~45 calendar days ≈ 30 trading days buffer beyond lookback)
CALENDAR_BUFFER_DAYS = 45

# Risk-Based Methods: Pure risk diversification where directional alpha scores are ignored
RISK_ONLY_METHODS = {"risk_parity", "inverse_vol"}

# Strict Gross Exposure Cap ($|L| + |S|$). 1.0 = Fully funded.
MAX_LEVERAGE = getattr(config, "MAX_LEVERAGE", 1.0)


class ProductionOptimizer:
    def __init__(self, capital: float, method: str = "mean_variance",
                 top_n: int = 25, target_vol: float = 0.15):
        """
        Initializes the ProductionOptimizer with targeted execution constraints.

        Args:
            capital (float): Total target allocation capital in base currency.
            method (str, optional): The optimization objective. Defaults to 'mean_variance'.
            top_n (int, optional): Maximum cardinality of the portfolio. Defaults to 25.
            target_vol (float, optional): The annualized volatility constraint. Defaults to 0.15.
        """
        self.capital    = capital
        self.method     = method
        self.top_n      = top_n
        self.target_vol = target_vol
        
        allocator_method = "mean_variance" if method == "top_n" else method
        self.allocator  = PortfolioAllocator(
            method=allocator_method,
            risk_aversion=getattr(config, "OPT_RISK_AVERSION", 2.5),
            fraction=getattr(config, "OPT_KELLY_FRACTION", 1.0),
            tau=0.05,
        )

    def load_latest_predictions(self) -> tuple[pd.DataFrame, date]:
        """
        Retrieves and normalizes the most recent alpha signal artifact from the Data Lake.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, date]: A tuple containing the localized signal 
                dataframe and its parsed generation date.

        Raises:
            FileNotFoundError: If the predictions directory or artifacts are missing.
            ValueError: If no data corresponds to the parsed signal date.
        """
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

        # Resolves UTC-aware timestamps stored via PyArrow to prevent temporal misalignment
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        try:
            file_date_str = latest_file.stem.replace("alpha_signals_", "")
            signal_date   = datetime.strptime(file_date_str, "%Y-%m-%d").date()
        except ValueError:
            signal_date = df["date"].max().date()

        # Observability Guard: Warns if signals are stale ($T_{lag} > 1$ trading day)
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

    def load_market_data(self, tickers: list[str], end_date: date) -> tuple[pd.DataFrame, dict]:
        r"""
        Ingests historical pricing and structural market capitalization data.

        Supplies the required longitudinal price matrix for Risk Modeling ($\Sigma$)
        and capitalization vectors for Black-Litterman Priors ($\Pi$).

        Args:
            tickers (list[str]): The universe of target candidate symbols.
            end_date (date): The temporal boundary for historical data extraction.

        Returns:
            tuple[pd.DataFrame, dict]: A tuple containing the $T \times N$ Close price matrix
                and a dictionary mapping tickers to their latest market capitalizations.
        """
        logger.info(f"Loading market data for {len(tickers)} tickers...")

        if not PRICES_DIR.exists():
            raise FileNotFoundError(f"Master data cache missing: {PRICES_DIR}")

        master_df = load_parquet(PRICES_DIR)
        master_df["date"] = pd.to_datetime(master_df["date"]).dt.tz_localize(None)

        # Binds required lookback horizon padded dynamically for non-trading calendar gaps
        start_date = pd.Timestamp(end_date) - pd.Timedelta(
            days=DEFAULT_LOOKBACK_DAYS + CALENDAR_BUFFER_DAYS
        )
        mask = (master_df["date"] >= start_date) & (master_df["date"] <= pd.Timestamp(end_date))

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

        # Isolates point-in-time structural variables required for equilibrium priors
        market_caps: dict = {}
        if "market_cap" in df.columns:
            last_caps = df.sort_values("date").groupby("ticker")["market_cap"].last()
            market_caps = last_caps.dropna().to_dict()
            logger.info(f"Market caps loaded for {len(market_caps)} tickers.")
        else:
            logger.debug("market_cap column not in master data — BL will use equal-weight prior.")

        return price_matrix[available_tickers], market_caps

    def estimate_risk_model(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        r"""
        Estimates the annualized Covariance Matrix ($\Sigma$) using Ledoit-Wolf shrinkage.

        Args:
            price_matrix (pd.DataFrame): The $T \times N$ historical price matrix.

        Returns:
            pd.DataFrame: The $N \times N$ annualized covariance matrix.

        Raises:
            ValueError: If insufficient clean return series remain after filtering sparse histories.
        """
        returns   = calculate_returns(price_matrix).dropna(how="all")
        valid_cols = returns.columns[returns.isnull().mean() < 0.3]

        # Liquidity Filter: Prunes structurally incomplete series (e.g., recent IPOs)
        dropped = set(returns.columns) - set(valid_cols)
        if dropped:
            logger.warning(
                f"Dropped {len(dropped)} ticker(s) from risk model "
                f"(>30% missing history — likely newly listed): "
                f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}"
            )

        # Stability Guard: Discards columns with unresolved structural NaNs to explicitly 
        # avoid zero-bias distortion in cross-sectional risk attribution.
        returns = returns[valid_cols].dropna(axis=1, how="any")
        if returns.empty or returns.shape[1] < 2:
            logger.error("Risk model: insufficient clean return columns after NaN drop.")
            raise ValueError("Not enough clean return series for covariance estimation.")

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

    def generate_orders(self, weights: dict[str, float],
                        prices: pd.Series) -> pd.DataFrame:
        r"""
        Discretizes target weights into integer share counts.

        Implementation Note:
            Sizing uses $Close_T$ as a proxy for execution price $Open_{T+1}$.
            This introduces a drift/slippage component of $\approx 0.3-0.8\%$.
            Precise execution requires an intraday execution algorithm.

        Args:
            weights (dict[str, float]): A dictionary mapping tickers to their optimized continuous weights.
            prices (pd.Series): The latest closing prices vector.

        Returns:
            pd.DataFrame: A structured ledger of generated trades detailing shares, side, and dollar value.
        """
        logger.info(
            "Share counts based on Close(T). Orders execute at Open(T+1). "
            "Expect ~0.3-0.8% position sizing deviation on fill."
        )

        orders = []
        for ticker, weight in weights.items():
            if abs(weight) < 1e-4:
                continue

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

    def _print_risk_report(self, weights: dict, cov_matrix: pd.DataFrame,
                           orders_df: pd.DataFrame):
        """
        Calculates and emits ex-ante portfolio risk telemetry to standard output.

        Args:
            weights (dict): The final optimized portfolio weights.
            cov_matrix (pd.DataFrame): The $N \times N$ annualized covariance matrix.
            orders_df (pd.DataFrame): The generated discrete order ledger.

        Returns:
            None
        """
        tickers = cov_matrix.index.tolist()
        w_vec   = np.array([weights.get(t, 0.0) for t in tickers])

        # Derives geometric variance structure via inner product resolution
        port_var = float(w_vec.T @ cov_matrix.values @ w_vec)
        port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

        gross_exp = orders_df["value"].abs().sum() / self.capital
        net_exp   = orders_df["value"].sum() / self.capital
        
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

    def run(self):
        """
        Orchestrates the portfolio construction Directed Acyclic Graph (DAG).

        Args:
            None

        Returns:
            None
        """
        logger.info(f"Starting Portfolio Optimization | Capital: ${self.capital:,.0f} | "
                    f"Method: {self.method} | Top-N: {self.top_n}")

        preds_df, signal_date = self.load_latest_predictions()

        if "ensemble_alpha" in preds_df.columns:
            score_col = "ensemble_alpha"
        elif "prediction" in preds_df.columns:
            score_col = "prediction"
        else:
            raise ValueError(
                f"No alpha score column found. Expected 'ensemble_alpha' or "
                f"'prediction'. Available: {preds_df.columns.tolist()}"
            )

        # Imposes universe boundaries using raw signals to define the active tradeable cohort
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

        price_matrix, loaded_caps = self.load_market_data(tickers, signal_date)
        latest_prices = price_matrix.iloc[-1]

        valid_tickers = [t for t in tickers if t in price_matrix.columns]
        price_matrix  = price_matrix[valid_tickers]

        if len(valid_tickers) < 2:
            logger.error("Not enough valid tickers for optimization. Aborting.")
            return

        cov_matrix    = self.estimate_risk_model(price_matrix)
        valid_tickers = cov_matrix.index.tolist()

        if len(valid_tickers) < 2:
            logger.error("Risk model returned fewer than 2 tickers. Aborting.")
            return

        logger.info(f"Optimizing using {self.method}...")

        MAX_ALPHA_RET = getattr(config, "MAX_ALPHA_RET", 0.30)
        raw_scores = np.array([alpha_scores.get(t, 0.5) for t in valid_tickers])
        
        # Remaps strict ordinal rank bounds [0, 1] into standardized expected return ranges [-1, 1]
        normalised = (raw_scores - 0.5) * 2.0
        scaled_rets = normalised * MAX_ALPHA_RET
        expected_returns = {t: float(scaled_rets[i]) for i, t in enumerate(valid_tickers)}
        logger.info(
            f"Alpha rescaled: [{scaled_rets.min():.1%}, {scaled_rets.max():.1%}] "
            f"(MAX_ALPHA_RET={MAX_ALPHA_RET:.0%}). Used for MVO/Kelly/BL.")

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

        # Defines the objective conviction boundary linking idiosyncratic alpha to systemic priors
        bl_confidence = getattr(config, "BL_CONFIDENCE_LEVEL", 0.6)
        
        _min_weight = getattr(config, "OPT_MIN_WEIGHT", 0.0)
        _POSITION_LIMIT = getattr(config, "BACKTEST_POSITION_LIMIT", 0.10)
        _max_weight = getattr(config, "OPT_MAX_WEIGHT", getattr(config, "MAX_POSITION_SIZE", _POSITION_LIMIT))

        if self.method == "top_n":
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
                    confidence_level=bl_confidence,
                    min_weight=_min_weight,
                    max_weight=_max_weight,
                )
            except Exception as exc:
                logger.error(f"Optimization failed: {exc}. Falling back to equal weight.")
                weights = {t: 1.0 / len(valid_tickers) for t in valid_tickers}

        if self.target_vol > 0 and self.method != "top_n":
            w_vec    = np.array([weights.get(t, 0.0) for t in valid_tickers])
            port_var = float(w_vec.T @ cov_matrix.values @ w_vec)
            port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

            if port_vol > 1e-6:
                scaler = min(self.target_vol / port_vol, 3.0)

                if scaler < 1.0 and self.method in RISK_ONLY_METHODS:
                    logger.info(
                        f"Vol targeting: raw vol={port_vol:.1%} > target={self.target_vol:.1%}. "
                        f"Skipping scale-down for '{self.method}' — "
                        f"reduce universe size (--top-n) instead."
                    )
                else:
                    logger.info(
                        f"Vol targeting: scaling weights {scaler:.2f}x "
                        f"(raw vol={port_vol:.1%} → target={self.target_vol:.1%})"
                    )
                    scaled = {t: w * scaler for t, w in weights.items()}

                    gross_exp = sum(abs(w) for w in scaled.values())
                    if gross_exp > MAX_LEVERAGE:
                        logger.warning(
                            f"Vol scaling would create {gross_exp:.1%} gross exposure "
                            f"(scaler={scaler:.2f}x) — exceeds MAX_LEVERAGE={MAX_LEVERAGE:.1%}. "
                            f"Capping. To allow leverage, set config.MAX_LEVERAGE > 1.0."
                        )
                        scaled = {t: w / (gross_exp / MAX_LEVERAGE) for t, w in scaled.items()}

                    for t, w in list(scaled.items()):
                        if w > _max_weight:
                            scaled[t] = _max_weight
                        elif w < -_max_weight:
                            scaled[t] = -_max_weight
                            
                    weights = scaled

        orders_df = self.generate_orders(weights, latest_prices)

        orders_df["signal_date"] = signal_date

        if orders_df.empty:
            logger.error("No valid orders generated. Check price data and weights.")
            return

        self._print_risk_report(weights, cov_matrix, orders_df)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        ts         = datetime.now().strftime("%H%M%S")
        filename   = f"orders_{signal_date}_{ts}.csv"
        latest_fn  = "orders_latest.csv"

        orders_df.to_csv(OUTPUT_DIR / filename,  index=False)
        orders_df.to_csv(OUTPUT_DIR / latest_fn, index=False)

        logger.info(f"Orders saved → {OUTPUT_DIR / filename}")
        logger.info(f"Latest copy  → {OUTPUT_DIR / latest_fn}")

def main():
    """
    Primary execution routine for CLI invocation.

    Args:
        None

    Returns:
        None
    """
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
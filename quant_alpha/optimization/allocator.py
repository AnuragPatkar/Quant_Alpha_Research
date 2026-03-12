"""
Portfolio Optimization Strategy Facade
======================================
Unified orchestration layer for portfolio construction and capital allocation algorithms.

Purpose
-------
The `PortfolioAllocator` implements the **Strategy Pattern** to abstract specific optimization
engines (Mean-Variance, Risk Parity, Kelly Criterion, Black-Litterman) behind a consistent
application programming interface (API). It acts as the switchboard for the portfolio
construction process, handling method selection, parameter injection, and defensive
fallback logic.

Usage
-----
Intended for use within the rebalancing loop of a backtesting engine or live trading system.

.. code-block:: python

    # Initialize with Mean-Variance preference
    allocator = PortfolioAllocator(method='mean_variance', risk_aversion=2.5)

    # Execute allocation
    target_weights = allocator.allocate(
        expected_returns=alpha_signals,
        covariance_matrix=risk_model
    )

Importance
----------
- **Architectural Decoupling**: Separates the *intention* (allocate capital) from the
  *implementation* (quadratic programming vs. numerical optimization).
- **Operational Resilience**: Enforces "Defensive Optimization" principles. If a complex
  solver fails (e.g., non-positive semi-definite covariance), the system gracefully
  degrades to a Maximum Entropy (Equal Weight) solution rather than halting execution.
- **Protocol Standardization**: Guarantees that all optimizers return normalized,
  fully-invested weight vectors ($\sum w_i = 1.0$), simplifying downstream execution logic.

Tools & Frameworks
------------------
- **Pandas**: Alignment of index-aware data structures (Tickers $\times$ Dates).
- **Optimization Engines**: Wraps internal modules (`mean_variance`, `risk_parity`, etc.)
  which utilize **CVXPY** and **SciPy**.
"""

import pandas as pd
import logging
from typing import Dict, Optional, List, Any

from .mean_variance import MeanVarianceOptimizer
from .risk_parity import RiskParityOptimizer
from .kelly_criterion import KellyCriterion
from .black_litterman import BlackLittermanModel

logger = logging.getLogger(__name__)


class PortfolioAllocator:
    """
    Facade class for dynamically instantiating and executing portfolio optimizers.
    
    Implements the Strategy design pattern, allowing the client to select algorithms
    at runtime via configuration strings.
    """

    def __init__(self, method: str = 'mean_variance', **kwargs):
        """
        Args:
            method (str): The optimization strategy identifier. Options:
                - ``'mean_variance'``: Classical Markowitz Mean-Variance.
                - ``'risk_parity'``: Equal Risk Contribution (ERC).
                - ``'kelly'``: Kelly Criterion (Geometric Growth Maximization).
                - ``'black_litterman'``: Bayesian view blending.
            **kwargs: Keyword arguments passed directly to the optimizer's constructor
                (e.g., `risk_aversion`, `target_risk`, `tau`).
        """
        self.method = method.lower()
        self.kwargs = kwargs
        self.optimizer = self._get_optimizer(self.method, kwargs)
        logger.info(f"PortfolioAllocator initialized with method: {self.method}")

    def _get_optimizer(self, method: str, kwargs: Dict[str, Any]):
        """Factory method: Instantiates the concrete optimization strategy class."""
        if method == 'mean_variance':
            return MeanVarianceOptimizer(
                risk_aversion=kwargs.get('risk_aversion', 1.0)
            )
        elif method == 'risk_parity':
            return RiskParityOptimizer(
                target_risk=kwargs.get('target_risk')
            )
        elif method == 'kelly':
            return KellyCriterion(
                fraction=kwargs.get('fraction', 0.5),
                max_leverage=kwargs.get('max_leverage', 1.0),
                use_solver=kwargs.get('use_solver', True)
            )
        elif method == 'black_litterman':
            return BlackLittermanModel(
                tau=kwargs.get('tau', 0.05),
                risk_aversion=kwargs.get('risk_aversion', 2.5)
            )
        else:
            raise ValueError(
                f"Unknown optimization method: {method}. "
                "Options: mean_variance, risk_parity, kelly, black_litterman"
            )

    def _equal_weight(self, tickers: List[str]) -> Dict[str, float]:
        """Maximum Entropy Fallback: Assigns $w_i = 1/N$ to all assets."""
        n = len(tickers)
        if n == 0:
            return {}
        return {t: 1.0 / n for t in tickers}

    def allocate(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Executes the optimization strategy to generate target portfolio weights.

        Args:
            expected_returns (Dict[str, float]): Expected returns vector ($\mu$).
            covariance_matrix (pd.DataFrame): Asset covariance matrix ($\Sigma$).
            constraints (Optional[Dict]): Optimization constraints (e.g., leverage limits).
            **kwargs: Runtime parameters (e.g., `market_caps` for BL, `risk_free_rate` for Kelly).

        Returns:
            Dict[str, float]: Normalized target weights where $\sum w_i = 1.0$.
            
        Note:
            Automatically falls back to Equal Weight allocation if the underlying
            optimizer fails or returns an empty set due to data misalignment.
        """
        tickers = list(expected_returns.keys())

        try:
            if self.method == 'mean_variance':
                result = self.optimizer.optimize(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    constraints=constraints,
                )

            elif self.method == 'risk_parity':
                result = self.optimizer.optimize(
                    covariance_matrix=covariance_matrix,
                    tickers=tickers,
                )

            elif self.method == 'kelly':
                rf_rate = kwargs.get('risk_free_rate', 0.04)
                result = self.optimizer.calculate_portfolio(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    risk_free_rate=rf_rate,
                )

            elif self.method == 'black_litterman':
                market_caps  = kwargs.get('market_caps', {})
                conf_level   = kwargs.get('confidence_level', 0.5)

                if not market_caps:
                    logger.warning(
                        "Black-Litterman requires 'market_caps'. "
                        "Using equal weights for prior as fallback."
                    )

                result = self.optimizer.optimize(
                    ml_predictions=expected_returns,
                    covariance_matrix=covariance_matrix,
                    market_caps=market_caps,
                    confidence_level=conf_level,
                    constraints=constraints,
                )

            else:
                return {}

            # Defensive Optimization: Check for solver failure / empty results.
            # Underlying optimizers may return {} on convergence failure or data mismatch
            # (e.g. zero intersection between returns and covariance).
            # We treat this as a failure mode and degrade to Equal Weight (Max Entropy).
            if not result:
                logger.warning(
                    f"{self.method} optimizer returned empty weights. "
                    "Falling back to Equal Weight allocation."
                )
                return self._equal_weight(tickers)

            return result

        except Exception as e:
            logger.error(f"Allocation failed using {self.method}: {e}")
            logger.warning("Falling back to Equal Weight allocation.")
            return self._equal_weight(tickers)
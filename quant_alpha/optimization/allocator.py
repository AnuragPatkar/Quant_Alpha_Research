"""
Portfolio Allocator - Strategy Facade
Unified interface for portfolio construction methods.

BUGS FIXED:
  BUG-AL-01 [HIGH]: When MeanVarianceOptimizer._prepare_data finds zero ticker
    intersection it raises ValueError internally, catches it, logs it, and
    returns {} — without raising. The allocator's outer except-block never
    fires so the empty dict propagates silently to the caller.

    Fix: After every optimizer call, check whether the result is empty and,
    if so, fall back to Equal Weight using the input expected_returns keys.
    This is the correct defensive posture: an empty optimizer result is always
    a failure mode (no ticker overlap, degenerate input, etc.) and the safest
    recovery is equal weight rather than crashing downstream portfolio logic.
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
    High-level interface for Portfolio Allocation.
    Selects and runs the appropriate optimization strategy based on config.

    Usage:
        allocator = PortfolioAllocator(method='mean_variance', risk_aversion=2.5)
        weights = allocator.allocate(expected_returns, covariance_matrix)
    """

    def __init__(self, method: str = 'mean_variance', **kwargs):
        """
        Args:
            method: Optimization method
                    ('mean_variance', 'risk_parity', 'kelly', 'black_litterman')
            **kwargs: Arguments passed to the specific optimizer constructor.
        """
        self.method = method.lower()
        self.kwargs = kwargs
        self.optimizer = self._get_optimizer(self.method, kwargs)
        logger.info(f"PortfolioAllocator initialized with method: {self.method}")

    def _get_optimizer(self, method: str, kwargs: Dict[str, Any]):
        """Factory method to instantiate the specific optimizer."""
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
        """Equal-weight fallback across all input tickers."""
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
        Generate target portfolio weights using the selected strategy.

        Args:
            expected_returns: Dictionary of expected returns (Alpha)
            covariance_matrix: DataFrame of asset covariance
            constraints: Dictionary of constraints (passed to optimizer where applicable)
            **kwargs: Additional runtime arguments
                      (e.g., market_caps for BL, risk_free_rate for Kelly)

        Returns:
            Dict[str, float]: Target weights {ticker: weight}.
            Always returns a fully-invested portfolio (sums to 1.0).
            Falls back to Equal Weight if optimizer returns empty or raises.
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

            # FIX BUG-AL-01: guard against empty optimizer result.
            # Optimizers catch their own errors and return {} — without raising.
            # An empty result here always indicates a failure mode (zero ticker
            # overlap, degenerate inputs, etc.) so we substitute Equal Weight.
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
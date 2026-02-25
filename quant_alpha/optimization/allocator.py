"""
Portfolio Allocator - Strategy Facade
Unified interface for portfolio construction methods.
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
            method: Optimization method ('mean_variance', 'risk_parity', 'kelly', 'black_litterman')
            **kwargs: Arguments passed to the specific optimizer constructor
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
            raise ValueError(f"Unknown optimization method: {method}. Options: mean_variance, risk_parity, kelly, black_litterman")

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
            **kwargs: Additional runtime arguments (e.g., market_caps for BL, risk_free_rate)
        
        Returns:
            Dict[str, float]: Target weights {ticker: weight}
        """
        try:
            if self.method == 'mean_variance':
                return self.optimizer.optimize(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    constraints=constraints
                )
            
            elif self.method == 'risk_parity':
                # Risk Parity calculates weights based on covariance, tickers derived from returns keys
                tickers = list(expected_returns.keys())
                return self.optimizer.optimize(
                    covariance_matrix=covariance_matrix,
                    tickers=tickers
                )
            
            elif self.method == 'kelly':
                rf_rate = kwargs.get('risk_free_rate', 0.04)
                return self.optimizer.calculate_portfolio(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    risk_free_rate=rf_rate
                )
            
            elif self.method == 'black_litterman':
                market_caps = kwargs.get('market_caps', {})
                conf_level = kwargs.get('confidence_level', 0.5)
                
                if not market_caps:
                    logger.warning("Black-Litterman requires 'market_caps'. Using equal weights for prior as fallback.")
                
                return self.optimizer.optimize(
                    ml_predictions=expected_returns,
                    covariance_matrix=covariance_matrix,
                    market_caps=market_caps,
                    confidence_level=conf_level,
                    constraints=constraints
                )
            
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Allocation failed using {self.method}: {e}")
            # Fallback: Equal Weight
            tickers = list(expected_returns.keys())
            n = len(tickers)
            if n > 0:
                logger.warning("Falling back to Equal Weight allocation.")
                return {t: 1.0/n for t in tickers}
            return {}
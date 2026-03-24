"""
Portfolio Optimization & Construction Subsystem
===============================================

Provides the public API for convex portfolio optimization and position sizing.

Purpose
-------
This module orchestrates capital allocation across generated alpha signals 
using advanced structural optimization techniques (Markowitz Mean-Variance, 
Risk Parity, Kelly Criterion, and Bayesian Black-Litterman inference).

Role in Quantitative Workflow
-----------------------------
Acts as the terminal mathematical transformation in the alpha generation pipeline, 
translating theoretical expected returns and empirical risk bounds into discrete, 
fully-invested portfolio weights capable of live order execution.

Mathematical Dependencies
-------------------------
- **CVXPY**: Solves linearly constrained Quadratic and Second-Order Cone Programs.
- **SciPy**: Utilizes L-BFGS-B gradient solvers for non-linear log-barrier constraints.
- **NumPy/Pandas**: Vectorized matrix manipulations and structural cross-sectional bounds.
"""

from .allocator        import PortfolioAllocator      # noqa: F401
from .mean_variance    import MeanVarianceOptimizer   # noqa: F401
from .risk_parity      import RiskParityOptimizer     # noqa: F401
from .kelly_criterion  import KellyCriterion          # noqa: F401
from .black_litterman  import BlackLittermanModel     # noqa: F401
from .constraints      import PortfolioConstraints    # noqa: F401

__all__ = [
    "PortfolioAllocator",
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "KellyCriterion",
    "BlackLittermanModel",
    "PortfolioConstraints",
]
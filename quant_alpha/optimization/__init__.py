"""
quant_alpha/optimization/__init__.py
======================================
Portfolio construction layer: weight optimisation and position sizing.

Confirmed public API (from run_backtest.py, optimize_portfolio.py,
test_optimization.py, run_hyperopt.py):
    from quant_alpha.optimization.allocator       import PortfolioAllocator
    from quant_alpha.optimization.mean_variance   import MeanVarianceOptimizer
    from quant_alpha.optimization.risk_parity     import RiskParityOptimizer
    from quant_alpha.optimization.kelly_criterion import KellyCriterion
    from quant_alpha.optimization.black_litterman import BlackLittermanModel
    from quant_alpha.optimization.constraints     import PortfolioConstraints

Supported allocation methods via PortfolioAllocator:
    mean_variance  — Markowitz MVO with Ledoit-Wolf covariance
    risk_parity    — Equal Risk Contribution (Spinu log-barrier)
    kelly          — Multi-asset fractional Kelly Criterion (QP)
    black_litterman — BL posterior with alpha-view blending
    inverse_vol    — Inverse-volatility heuristic weighting
    top_n          — Equal weight across top-N alpha tickers
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
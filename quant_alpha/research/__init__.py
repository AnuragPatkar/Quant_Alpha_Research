"""
quant_alpha/research/__init__.py
==================================
Quantitative research toolkit: factor analysis, statistical significance
testing, alpha decay, regime detection, and factor correlation analysis.

Confirmed public API (from validate_factors.py, test_validation_integration.py,
run_hyperopt.py, test_features.py):
    from quant_alpha.research import FactorAnalyzer
    from quant_alpha.research.factor_analysis      import FactorAnalyzer
    from quant_alpha.research.significance_testing import SignificanceTester
    from quant_alpha.research.alpha_decay          import AlphaDecayAnalyzer
    from quant_alpha.research.regime_detection     import RegimeDetector
    from quant_alpha.research.correlation_analysis import FactorCorrelator

NOTE: The correlation analysis class is FactorCorrelator (not CorrelationAnalyzer —
that was an incorrect name used in the first __init__ draft).
"""

from .factor_analysis      import FactorAnalyzer      # noqa: F401
from .significance_testing import SignificanceTester   # noqa: F401
from .alpha_decay          import AlphaDecayAnalyzer  # noqa: F401
from .regime_detection     import RegimeDetector      # noqa: F401
from .correlation_analysis import FactorCorrelator    # noqa: F401

__all__ = [
    "FactorAnalyzer",
    "SignificanceTester",
    "AlphaDecayAnalyzer",
    "RegimeDetector",
    "FactorCorrelator",
]
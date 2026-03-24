"""
Quantitative Research Subsystem
===============================

Provides an institutional-grade research toolkit for alpha factor discovery, 
statistical validation, and structural market analysis.

Purpose
-------
This module exposes a unified public API for discrete research engines evaluating 
alpha signal efficacy, persistence, correlation boundaries, and latent market regimes.

Role in Quantitative Workflow
-----------------------------
Serves as the primary analytical workbench for quantitative researchers, transforming 
raw algorithmic signals into statistically verified, production-ready alpha features 
prior to model ingestion.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Structural matrix manipulations and cross-sectional indices.
- **SciPy/Statsmodels**: Hypothesis testing, distributional analysis, and 
  time-series stationarity bounding.
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
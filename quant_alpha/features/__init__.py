"""
Feature Engineering Subsystem
=============================

Exposes the core foundational primitives for alpha factor creation, 
registration, and parallelized mathematical computation.

Purpose
-------
Provides a unified public API for constructing cross-sectional alpha models, 
standardizing the mathematical lifecycle from raw indicator extraction 
to strictly neutralized predictor variables.

Role in Quantitative Workflow
-----------------------------
Serves as the structural pipeline guaranteeing determinism across disparate 
alpha signal computations, strictly mitigating out-of-sample data leakage.
"""
from .base import BaseFactor
from .registry import FactorRegistry

# Import all factors from sub-packages
from .fundamental import *
from .technical import *
from .earnings import *
from .alternative import *
from .composite import *

# Define public API
__all__ = ['BaseFactor', 'FactorRegistry']

# Extend __all__ with fundamental factors
from .fundamental import __all__ as _fundamental_all
__all__.extend(_fundamental_all)

# Extend __all__ with technical factors
from .technical import __all__ as _technical_all
__all__.extend(_technical_all)

# Extend __all__ with earnings factors
from .earnings import __all__ as _earnings_all
__all__.extend(_earnings_all)

# Extend __all__ with alternative factors
from .alternative import __all__ as _alternative_all
__all__.extend(_alternative_all)

# Extend __all__ with composite factors
from .composite import __all__ as _composite_all
__all__.extend(_composite_all)
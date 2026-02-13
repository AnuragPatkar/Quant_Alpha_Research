"""
Technical Features Module
Exposes sub-modules to ensure FactorRegistry auto-discovery works.
"""

# Import sub-modules to trigger @FactorRegistry.register() decorators
from . import momentum
from . import volatility
from . import mean_reversion
from . import volume

__all__ = [
    'momentum',
    'volatility',
    'mean_reversion',
    'volume'
]
"""
Configuration package for Quant Alpha Research.
"""

from .settings import (
    # Main settings instance
    settings,
    
    # Utility functions
    get_universe,
    get_feature_names,
    print_welcome,
    
    # Dataclass configs (for type hints and custom instantiation)
    Settings,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    ValidationConfig,
    BacktestConfig,
    RiskConfig,
    InterpretabilityConfig,
    LogConfig,
    
    # Universe constants
    STOCKS_SP500_TOP50,
    SURVIVORSHIP_BIAS_WARNING,
)

__all__ = [
    # Main settings
    "settings",
    
    # Utility functions
    "get_universe",
    "get_feature_names",
    "print_welcome",
    
    # Config classes
    "Settings",
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "ValidationConfig",
    "BacktestConfig",
    "RiskConfig",
    "InterpretabilityConfig",
    "LogConfig",
    
    # Constants
    "STOCKS_SP500_TOP50",
    "SURVIVORSHIP_BIAS_WARNING",
]
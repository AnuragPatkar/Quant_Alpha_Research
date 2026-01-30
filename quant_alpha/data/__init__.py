"""
Data loading module for Quant Alpha Research.

This module provides utilities for loading, validating, and preprocessing
stock market data for quantitative analysis.
"""

from .loader import DataLoader, DataValidationError

__all__ = [
    'DataLoader',
    'DataValidationError',
]
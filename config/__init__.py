"""
Configuration Initialization Module
===================================
Initializes the configuration subsystem for the Quant Alpha platform.

Purpose
-------
This module serves as the central entry point for all configuration and logging
initialization across the research and production environments. It exposes
the core `Config` object and establishes the foundational root logger
(`QuantAlpha`) used for systemic observability.

Importance
----------
- **Centralized Configuration**: Ensures that any component importing from `config`
  accesses a unified setting state.
- **Standardized Observability**: Pre-configures the logger to guarantee that all
  subsequent modules inherit consistent formatting and log-level constraints.
"""

from .settings import Config
from .logging_config import setup_logging

# Initialize the global application logger for the Quant Alpha platform
logger = setup_logging("QuantAlpha")

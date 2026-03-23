"""
Centralized Observability and Logging Subsystem
===============================================
Initializes the dual-sink logging architecture for the Quant Alpha platform.

Purpose
-------
This module provisions standard, globally accessible loggers that stream
telemetry to both standard output (console) and persistent disk storage.
It guarantees deterministic log formatting and environment-aware severity
levels, forming the backbone of system auditing and production monitoring.

Importance
----------
- **Auditability**: Ensures all critical quantitative operations (data ingestion,
  model training, execution) leave an immutable, timestamped trail.
- **Idempotency**: Implements handler guards to prevent log duplication
  across complex, multi-module execution graphs.
"""

import logging
import sys
from pathlib import Path

from config.settings import config

def setup_logging(name='Quant_Alpha', default_level=None):
    """
    Configures and returns a dual-sink logger instance.

    Args:
        name (str): The namespace identifier for the logger instance. Defaults to 'Quant_Alpha'.
        default_level (int, optional): Explicit override for the logging severity level. 
            If None, inherits from the global Config object.

    Returns:
        logging.Logger: A fully configured logger instance bound to both File and Stream handlers.
    """
    log_file = config.LOG_FILE

    # Inject %(module)s into the formatter to establish granular file-level traceability
    log_format = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s', datefmt=config.LOG_DATE_FORMAT)

    # Environment-aware severity resolution with dynamic fallback
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    if default_level:
        log_level = default_level
    else:
        log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    # Persistent Sink: Append-only disk logging for post-mortem analysis and compliance
    file_handler = logging.FileHandler(log_file,mode='a',encoding='utf-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(log_level)

    # Ephemeral Sink: Standard output streaming for immediate execution monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(log_level)  

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Idempotency Guard: Prevents exponential handler stacking upon repeated module imports
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Provision the default global logger instance
logger = setup_logging()

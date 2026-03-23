"""
quant_alpha/monitoring/__init__.py
====================================
Production observability layer: signal drift detection, live performance
tracking, data quality checks, model drift alerts, and system dashboard.

Confirmed public API (from run_backtest.py, monitor_production.py,
test_production.py, test_memory.py):
    from quant_alpha.monitoring.performance_tracker import PerformanceTracker
    from quant_alpha.monitoring.data_quality        import DataQualityMonitor
    from quant_alpha.monitoring.model_drift         import ModelDriftDetector
    from quant_alpha.monitoring.alerts              import AlertSystem

Note: The dashboard module exposes standalone functions (load_data,
run_performance_tracker, run_drift_detector) rather than a class.
"""

from .performance_tracker import PerformanceTracker  # noqa: F401
from .data_quality        import DataQualityMonitor  # noqa: F401
from .model_drift         import ModelDriftDetector  # noqa: F401
from .alerts              import AlertSystem         # noqa: F401

# Dashboard convenience functions (not a class)
from .dashboard import (                             # noqa: F401
    load_data,
    run_performance_tracker,
    run_drift_detector,
)

__all__ = [
    "PerformanceTracker",
    "DataQualityMonitor",
    "ModelDriftDetector",
    "AlertSystem",
    "load_data",
    "run_performance_tracker",
    "run_drift_detector",
]
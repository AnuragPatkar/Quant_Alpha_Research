"""
Production Observability & Monitoring Subsystem
===============================================

Provides real-time signal drift detection, live performance tracking, 
data quality constraints, model drift alerts, and system dashboards.

Purpose
-------
This module exposes the unified public API for tracking the continuous health 
and statistical validity of models executing in live production environments.

Role in Quantitative Workflow
-----------------------------
Acts as the primary protective layer against structural market regime shifts, 
silent data ingestion failures, and out-of-sample alpha decay. Triggers 
systematic circuit breakers prior to execution if distributional boundaries 
or drawdown thresholds are breached.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Distributional shift metrics (PSI) and cross-sectional alignments.
- **SciPy**: Continuous statistical bound evaluations for signal stationarity.
"""

from .performance_tracker import PerformanceTracker  # noqa: F401
from .data_quality        import DataQualityMonitor  # noqa: F401
from .model_drift         import ModelDriftDetector  # noqa: F401
from .alerts              import AlertSystem         # noqa: F401

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
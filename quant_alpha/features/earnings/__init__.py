"""
Earnings Feature Engineering Subsystem
======================================

Provides extraction and normalization algorithms for corporate earnings data, 
mapping point-in-time analyst estimates and realized surprises into continuous 
alpha signals.

Purpose
-------
This module aggregates standardized unexpected earnings (SUE), analyst revision 
trajectories, and consensus momentum to systematically capture Post-Earnings 
Announcement Drift (PEAD) and structural fundamental shifts.

Role in Quantitative Workflow
-----------------------------
Converts sparse, low-frequency discrete event data into high-frequency, stationary 
time-series suitable for ingestion into gradient-boosted decision tree ensembles.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Forward-filling paradigms, event detection logic, and 
  rolling aggregation matrices.
"""

from .surprises import (ConsecutiveSurprise, EPSSurprise, EPSSurprisePercentage,
                        LastQuarterMagnitude, BeatMissMomentum)
from .revisions import (EarningsMomentum, EstimateAccuracyTrend, EPSAcceleration,
                       RecentPositiveRevisions)
from .estimates import (ConsensusStrength, EstimateGuidanceQuality,
                        EstimateSurpriseConsistency, PositiveEstimateConfidence)

__all__ = [
    'ConsecutiveSurprise', 'EPSSurprise', 'EPSSurprisePercentage',
    'LastQuarterMagnitude', 'BeatMissMomentum',
    'EarningsMomentum', 'EstimateAccuracyTrend', 'EPSAcceleration',
    'RecentPositiveRevisions',
    'ConsensusStrength', 'EstimateGuidanceQuality',
    'EstimateSurpriseConsistency', 'PositiveEstimateConfidence'
]
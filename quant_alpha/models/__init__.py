"""
Models Module
=============
Machine learning models for alpha prediction.
"""

from .boosting import LightGBMModel, create_model
from .trainer import WalkForwardTrainer, run_walk_forward_validation, create_trainer

__all__ = [
    # Model classes
    'LightGBMModel',
    'WalkForwardTrainer',
    
    # Factory functions
    'create_model',
    'create_trainer',
    'run_walk_forward_validation'
]
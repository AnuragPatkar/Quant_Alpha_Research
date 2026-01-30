"""
Models Module
=============
Machine learning models for alpha prediction.

This module provides:
- LightGBM model wrapper with proper validation
- Walk-forward trainer for time-series cross-validation
- Model evaluation metrics (IC, IR, hit rate, etc.)
- Model persistence utilities

Usage:
    >>> from quant_alpha.models import LightGBMModel, WalkForwardTrainer
    >>> model = LightGBMModel(feature_names)
    >>> trainer = WalkForwardTrainer(feature_names)
    >>> results = trainer.train_and_validate(features_df)

Author: [Your Name]
Last Updated: 2024
"""

from .boosting import (
    # Main model class
    LightGBMModel,
    ModelConfig,
    
    # Factory functions
    create_model,
    create_ensemble_model,
    
    # Metrics functions
    calculate_ic,
    calculate_rank_ic,
    calculate_hit_rate,
    calculate_cross_sectional_ic,
    calculate_information_ratio,
    calculate_all_metrics,
)

from .trainer import (
    # Main trainer class
    WalkForwardTrainer,
    
    # Data classes
    FoldResult,
    WalkForwardResults,
    
    # Factory functions
    create_trainer,
    run_walk_forward_validation,
)

__all__ = [
    # Model classes
    'LightGBMModel',
    'ModelConfig',
    
    # Trainer classes
    'WalkForwardTrainer',
    'FoldResult',
    'WalkForwardResults',
    
    # Factory functions
    'create_model',
    'create_ensemble_model',
    'create_trainer',
    'run_walk_forward_validation',
    
    # Metrics functions
    'calculate_ic',
    'calculate_rank_ic',
    'calculate_hit_rate',
    'calculate_cross_sectional_ic',
    'calculate_information_ratio',
    'calculate_all_metrics',
]
"""
Inference module - Prediction and cold-start handling.
"""

from .predictor import RecommenderPredictor
from .cold_start import ColdStartHandler

__all__ = ["RecommenderPredictor", "ColdStartHandler"]
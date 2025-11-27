"""
Models module - ML models for recommendation.

Available Models:
- PopularityRecommender: Simple popularity baseline
- PersonalFrequencyRecommender: Personal + global frequency blend
- LightGBMRanker: LightGBM LambdaRank
- XGBoostRanker: XGBoost pairwise ranking
- CatBoostRanker: CatBoost YetiRank

Usage:
    from src.models import LightGBMRanker, XGBoostRanker
    
    model = LightGBMRanker(num_leaves=31)
    model.fit(train_df, feature_names)
"""

from .base_model import BaseRecommender
from .baseline_models import PopularityRecommender, PersonalFrequencyRecommender

# Conditionally import ML models (may not be installed)
try:
    from .lightgbm_ranker import LightGBMRanker
except ImportError:
    LightGBMRanker = None

try:
    from .xgboost_ranker import XGBoostRanker
except ImportError:
    XGBoostRanker = None

try:
    from .catboost_ranker import CatBoostRanker
except ImportError:
    CatBoostRanker = None


def get_available_models():
    """Return list of available model classes."""
    models = [
        ("Popularity", PopularityRecommender),
        ("PersonalFrequency", PersonalFrequencyRecommender),
    ]
    
    if LightGBMRanker is not None:
        models.append(("LightGBM", LightGBMRanker))
    
    if XGBoostRanker is not None:
        models.append(("XGBoost", XGBoostRanker))
    
    if CatBoostRanker is not None:
        models.append(("CatBoost", CatBoostRanker))
    
    return models


__all__ = [
    "BaseRecommender",
    "PopularityRecommender", 
    "PersonalFrequencyRecommender",
    "LightGBMRanker",
    "XGBoostRanker",
    "CatBoostRanker",
    "get_available_models"
]

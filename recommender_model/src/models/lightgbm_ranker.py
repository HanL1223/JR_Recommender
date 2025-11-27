"""
LightGBM Ranker
===============
Learning-to-rank model using LightGBM's LambdaRank.

Refactored from: F_ml_models.py
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)

# Check if LightGBM is available
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed - pip install lightgbm")


class LightGBMRanker(BaseRecommender):
    """
    LightGBM-based learning-to-rank model.
    
    Uses LambdaRank objective to learn product rankings.
    
    Example:
        >>> model = LightGBMRanker(num_leaves=31, learning_rate=0.05)
        >>> model.fit(train_df, feature_names)
        >>> recs = model.recommend(customer_id=5, products, features, top_k=5)
    """
    
    @property
    def name(self) -> str:
        return "LightGBM"
    
    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_data_in_leaf: int = 20,
        lambda_l1: float = 0.1,
        lambda_l2: float = 0.1,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose: int = -1
    ):
        """
        Initialize LightGBM ranker.
        
        Args:
            num_leaves: Max leaves per tree
            learning_rate: Boosting learning rate
            feature_fraction: Fraction of features per tree
            bagging_fraction: Fraction of samples per tree
            bagging_freq: Bagging frequency
            min_data_in_leaf: Minimum samples per leaf
            lambda_l1: L1 regularization
            lambda_l2: L2 regularization
            num_boost_round: Max boosting iterations
            early_stopping_rounds: Early stopping patience
            verbose: Logging verbosity (-1 = silent)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'min_data_in_leaf': min_data_in_leaf,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'verbose': verbose,
            'seed': 42
        }
        
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        
        self.model = None
        self.feature_names = None
        self._is_fitted = False
        
        logger.info(f"LightGBMRanker initialized (num_leaves={num_leaves}, lr={learning_rate})")
    
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        valid_df: pd.DataFrame = None,
        **kwargs
    ) -> 'LightGBMRanker':
        """
        Train the ranking model.
        
        Args:
            train_df: Training data with features, labels, customer_id, order_idx
            feature_names: List of feature column names
            valid_df: Optional validation data
            
        Returns:
            self
        """
        logger.info(f"Fitting {self.name} model...")
        
        self.feature_names = feature_names
        
        # Prepare training data
        X_train = train_df[feature_names].values
        y_train = train_df['label'].values
        
        # Group by (customer_id, order_idx) for ranking
        train_groups = train_df.groupby(['customer_id', 'order_idx']).size().values
        
        train_dataset = lgb.Dataset(
            X_train, 
            label=y_train,
            group=train_groups,
            feature_name=feature_names
        )
        
        # Prepare validation if provided
        valid_sets = [train_dataset]
        valid_names = ['train']
        
        if valid_df is not None:
            X_valid = valid_df[feature_names].values
            y_valid = valid_df['label'].values
            valid_groups = valid_df.groupby(['customer_id', 'order_idx']).size().values
            
            valid_dataset = lgb.Dataset(
                X_valid,
                label=y_valid,
                group=valid_groups,
                feature_name=feature_names,
                reference=train_dataset
            )
            valid_sets.append(valid_dataset)
            valid_names.append('valid')
        
        # Train with callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]
        
        logger.info(f"Training with {len(X_train):,} samples, {len(train_groups):,} groups...")
        
        self.model = lgb.train(
            self.params,
            train_dataset,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self._is_fitted = True
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame
    ) -> np.ndarray:
        """Predict relevance scores for products."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = features[self.feature_names].values
        scores = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        return scores
    
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on DataFrame directly."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = df[self.feature_names].values
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_names, importance))
    
    def get_params(self) -> Dict[str, Any]:
        """Get all parameters for MLflow logging."""
        params = self.params.copy()
        params['num_boost_round'] = self.num_boost_round
        params['early_stopping_rounds'] = self.early_stopping_rounds
        
        if self._is_fitted:
            params['best_iteration'] = self.model.best_iteration
            params['n_trees'] = self.model.num_trees()
        
        return params
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'LightGBMRanker':
        """Create model from parameter dict (useful for hyperparameter tuning)."""
        return cls(
            num_leaves=params.get('num_leaves', 31),
            learning_rate=params.get('learning_rate', 0.05),
            feature_fraction=params.get('feature_fraction', 0.8),
            bagging_fraction=params.get('bagging_fraction', 0.8),
            min_data_in_leaf=params.get('min_data_in_leaf', 20),
            lambda_l1=params.get('lambda_l1', 0.1),
            lambda_l2=params.get('lambda_l2', 0.1),
            num_boost_round=params.get('num_boost_round', 500),
            early_stopping_rounds=params.get('early_stopping_rounds', 50)
        )
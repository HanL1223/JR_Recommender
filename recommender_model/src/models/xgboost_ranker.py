"""
XGBoost Ranker
==============
Learning-to-rank model using XGBoost's ranking objective.

Refactored from: F_ml_models.py
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)

# Check if XGBoost is available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed - pip install xgboost")


class XGBoostRanker(BaseRecommender):
    """
    XGBoost-based learning-to-rank model.
    
    Uses pairwise ranking objective (rank:pairwise) to learn product rankings.
    
    Example:
        >>> model = XGBoostRanker(max_depth=6, learning_rate=0.1)
        >>> model.fit(train_df, feature_names, valid_df)
        >>> recs = model.recommend(customer_id=5, products, features, top_k=5)
    """
    
    @property
    def name(self) -> str:
        return "XGBoost"
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 500,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        objective: str = 'rank:pairwise',  # or 'rank:ndcg'
        verbose: int = 0
    ):
        """
        Initialize XGBoost ranker.
        
        Args:
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            n_estimators: Number of boosting rounds
            subsample: Fraction of samples per tree
            colsample_bytree: Fraction of features per tree
            min_child_weight: Minimum sum of instance weight in a child
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            early_stopping_rounds: Early stopping patience
            objective: Ranking objective ('rank:pairwise' or 'rank:ndcg')
            verbose: Logging verbosity
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self.params = {
            'objective': objective,
            'eval_metric': 'ndcg',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'verbosity': verbose,
            'seed': 42
        }
        
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        
        self.model = None
        self.feature_names = None
        self._is_fitted = False
        
        logger.info(f"XGBoostRanker initialized (max_depth={max_depth}, lr={learning_rate})")
    
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        valid_df: pd.DataFrame = None,
        **kwargs
    ) -> 'XGBoostRanker':
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
        
        # Group sizes for ranking (group by customer_id, order_idx)
        train_groups = train_df.groupby(['customer_id', 'order_idx']).size().values
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtrain.set_group(train_groups)
        
        # Prepare validation if provided
        evals = [(dtrain, 'train')]
        
        if valid_df is not None:
            X_valid = valid_df[feature_names].values
            y_valid = valid_df['label'].values
            valid_groups = valid_df.groupby(['customer_id', 'order_idx']).size().values
            
            dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
            dvalid.set_group(valid_groups)
            evals.append((dvalid, 'valid'))
        
        logger.info(f"Training with {len(X_train):,} samples, {len(train_groups):,} groups...")
        
        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100 if self.params['verbosity'] > 0 else False
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
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        scores = self.model.predict(dmatrix)
        
        return scores
    
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on DataFrame directly."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = df[self.feature_names].values
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmatrix)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            return {}
        
        importance = self.model.get_score(importance_type='gain')
        return importance
    
    def get_params(self) -> Dict[str, Any]:
        """Get all parameters for MLflow logging."""
        params = self.params.copy()
        params['n_estimators'] = self.n_estimators
        params['early_stopping_rounds'] = self.early_stopping_rounds
        
        if self._is_fitted:
            params['best_iteration'] = self.model.best_iteration
            params['best_score'] = self.model.best_score
        
        return params
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'XGBoostRanker':
        """Create model from parameter dict (useful for hyperparameter tuning)."""
        return cls(
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            n_estimators=params.get('n_estimators', 500),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            min_child_weight=params.get('min_child_weight', 1),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 1.0),
            early_stopping_rounds=params.get('early_stopping_rounds', 50)
        )
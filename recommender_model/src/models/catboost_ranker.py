"""
CatBoost Ranker
===============
Learning-to-rank model using CatBoost's YetiRank.

CatBoost is known for:
- Handling categorical features natively
- Robust default parameters
- Good performance out-of-the-box
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)

# Check if CatBoost is available
try:
    from catboost import CatBoost, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not installed - pip install catboost")


class CatBoostRanker(BaseRecommender):
    """
    CatBoost-based learning-to-rank model.
    
    Uses YetiRank or YetiRankPairwise for ranking.
    
    Example:
        >>> model = CatBoostRanker(depth=6, learning_rate=0.1)
        >>> model.fit(train_df, feature_names, valid_df)
        >>> recs = model.recommend(customer_id=5, products, features, top_k=5)
    """
    
    @property
    def name(self) -> str:
        return "CatBoost"
    
    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.1,
        iterations: int = 500,
        l2_leaf_reg: float = 3.0,
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
        early_stopping_rounds: int = 50,
        loss_function: str = 'YetiRank',  # or 'YetiRankPairwise'
        verbose: int = 0
    ):
        """
        Initialize CatBoost ranker.
        
        Args:
            depth: Maximum tree depth
            learning_rate: Boosting learning rate
            iterations: Number of boosting iterations
            l2_leaf_reg: L2 regularization coefficient
            bagging_temperature: Bayesian bootstrap parameter
            random_strength: Random strength for scoring splits
            early_stopping_rounds: Early stopping patience
            loss_function: Ranking loss ('YetiRank' or 'YetiRankPairwise')
            verbose: Logging verbosity
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        self.params = {
            'loss_function': loss_function,
            'custom_metric': ['NDCG:top=3', 'NDCG:top=5'],
            'depth': depth,
            'learning_rate': learning_rate,
            'iterations': iterations,
            'l2_leaf_reg': l2_leaf_reg,
            'bagging_temperature': bagging_temperature,
            'random_strength': random_strength,
            'random_seed': 42,
            'verbose': verbose
        }
        
        self.early_stopping_rounds = early_stopping_rounds
        
        self.model = None
        self.feature_names = None
        self._is_fitted = False
        
        logger.info(f"CatBoostRanker initialized (depth={depth}, lr={learning_rate})")
    
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        valid_df: pd.DataFrame = None,
        **kwargs
    ) -> 'CatBoostRanker':
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
        
        # Group IDs for ranking
        train_df = train_df.copy()
        train_df['group_id'] = train_df.groupby(['customer_id', 'order_idx']).ngroup()
        group_id_train = train_df['group_id'].values
        
        train_pool = Pool(
            data=X_train,
            label=y_train,
            group_id=group_id_train,
            feature_names=feature_names
        )
        
        # Prepare validation if provided
        eval_set = None
        if valid_df is not None:
            X_valid = valid_df[feature_names].values
            y_valid = valid_df['label'].values
            
            valid_df = valid_df.copy()
            valid_df['group_id'] = valid_df.groupby(['customer_id', 'order_idx']).ngroup()
            group_id_valid = valid_df['group_id'].values
            
            eval_set = Pool(
                data=X_valid,
                label=y_valid,
                group_id=group_id_valid,
                feature_names=feature_names
            )
        
        logger.info(f"Training with {len(X_train):,} samples...")
        
        # Train
        self.model = CatBoost(self.params)
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.params['verbose']
        )
        
        self._is_fitted = True
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration_}")
        
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
        scores = self.model.predict(X)
        
        return scores
    
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on DataFrame directly."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = df[self.feature_names].values
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            return {}
        
        importance = self.model.get_feature_importance()
        return dict(zip(self.feature_names, importance))
    
    def get_params(self) -> Dict[str, Any]:
        """Get all parameters for MLflow logging."""
        params = self.params.copy()
        params['early_stopping_rounds'] = self.early_stopping_rounds
        
        if self._is_fitted:
            params['best_iteration'] = self.model.best_iteration_
        
        return params
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'CatBoostRanker':
        """Create model from parameter dict."""
        return cls(
            depth=params.get('depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            iterations=params.get('iterations', 500),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            bagging_temperature=params.get('bagging_temperature', 1.0),
            early_stopping_rounds=params.get('early_stopping_rounds', 50)
        )
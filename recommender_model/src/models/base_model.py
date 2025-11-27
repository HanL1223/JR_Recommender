"""
Base Model
==========
Abstract base class for all recommendation models.

This enforces a consistent interface across models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """
    Abstract base class for recommendation models.
    
    All recommender models must implement:
    - fit(): Train the model
    - predict(): Score products for a customer
    - recommend(): Get top-k recommendations
    
    Example:
        >>> class MyModel(BaseRecommender):
        ...     def fit(self, train_df, ...): ...
        ...     def predict(self, customer_id, products): ...
        ...     def recommend(self, customer_id, top_k): ...
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging and tracking."""
        pass
    
    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> 'BaseRecommender':
        """
        Train the model.
        
        Args:
            train_df: Training DataFrame with features and labels
            feature_names: List of feature column names
            **kwargs: Additional training arguments
            
        Returns:
            self (for chaining)
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        Score products for a customer.
        
        Args:
            customer_id: Customer ID
            products: List of products to score
            features: Feature matrix for each product
            
        Returns:
            Array of scores (higher = more likely to purchase)
        """
        pass
    
    def recommend(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top-k product recommendations.
        
        Args:
            customer_id: Customer ID
            products: Candidate products
            features: Feature matrix
            top_k: Number of recommendations
            
        Returns:
            List of dicts with 'product', 'score', 'rank'
        """
        scores = self.predict(customer_id, products, features)
        
        # Sort by score descending
        sorted_idx = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for rank, idx in enumerate(sorted_idx, 1):
            recommendations.append({
                'product': products[idx],
                'score': float(scores[idx]),
                'rank': rank
            })
        
        return recommendations
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for MLflow logging."""
        return {}
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseRecommender':
        """Load model from disk."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
"""
Baseline Models
===============
Simple baseline recommenders for comparison.

Refactored from: E_baseline_models.py
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import Counter

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """
    Popularity-based recommender.
    
    Simply recommends the most globally popular products.
    Ignores customer history - useful as a baseline.
    
    Example:
        >>> model = PopularityRecommender()
        >>> model.fit(train_df, feature_names)
        >>> recs = model.recommend(customer_id=5, products, features, top_k=5)
    """
    
    @property
    def name(self) -> str:
        return "Popularity"
    
    def __init__(self):
        self.popularity_scores = {}
        self._is_fitted = False
        logger.info("PopularityRecommender initialized")
    
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> 'PopularityRecommender':
        """Learn product popularity from training data."""
        logger.info(f"Fitting {self.name} model...")
        
        # Count product occurrences in positive samples
        positive_df = train_df[train_df['label'] == 1]
        product_counts = positive_df['product'].value_counts()
        
        # Normalize to probabilities
        total = product_counts.sum()
        self.popularity_scores = (product_counts / total).to_dict()
        
        self._is_fitted = True
        logger.info(f"Learned popularity for {len(self.popularity_scores)} products")
        
        return self
    
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame
    ) -> np.ndarray:
        """Return popularity scores (ignores customer)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        scores = np.array([
            self.popularity_scores.get(p, 0.0) 
            for p in products
        ])
        
        return scores
    
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch prediction on DataFrame.
        
        Args:
            df: DataFrame with 'product' column
            
        Returns:
            Array of scores for each row
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return df['product'].map(lambda p: self.popularity_scores.get(p, 0.0)).values
    
    def get_params(self) -> Dict[str, Any]:
        return {"model_type": "popularity", "n_products": len(self.popularity_scores)}


class PersonalFrequencyRecommender(BaseRecommender):
    """
    Personal frequency-based recommender.
    
    Blends personal purchase frequency with global popularity:
    score = (1 - smoothing) * personal_freq + smoothing * global_freq
    
    This is a strong baseline for repeat-purchase scenarios.
    
    Example:
        >>> model = PersonalFrequencyRecommender(smoothing=0.3)
        >>> model.fit(train_df, feature_names)
        >>> recs = model.recommend(customer_id=5, products, features, top_k=5)
    """
    
    @property
    def name(self) -> str:
        return "PersonalFrequency"
    
    def __init__(self, smoothing: float = 0.3):
        """
        Initialize model.
        
        Args:
            smoothing: Weight for global popularity (0-1)
                      0 = pure personal, 1 = pure popularity
        """
        self.smoothing = smoothing
        self.customer_frequencies = {}
        self.global_popularity = {}
        self._is_fitted = False
        logger.info(f"PersonalFrequencyRecommender initialized (smoothing={smoothing})")
    
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> 'PersonalFrequencyRecommender':
        """Learn customer frequencies and global popularity."""
        logger.info(f"Fitting {self.name} model...")
        
        positive_df = train_df[train_df['label'] == 1]
        
        # Global popularity
        product_counts = positive_df['product'].value_counts()
        total = product_counts.sum()
        self.global_popularity = (product_counts / total).to_dict()
        
        # Per-customer frequencies
        for customer_id, group in positive_df.groupby('customer_id'):
            customer_counts = group['product'].value_counts()
            customer_total = customer_counts.sum()
            self.customer_frequencies[customer_id] = (customer_counts / customer_total).to_dict()
        
        self._is_fitted = True
        logger.info(f"Learned frequencies for {len(self.customer_frequencies)} customers")
        
        return self
    
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame
    ) -> np.ndarray:
        """Blend personal and global frequencies."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get customer's personal frequencies (empty dict if new customer)
        personal_freq = self.customer_frequencies.get(customer_id, {})
        
        scores = []
        for product in products:
            personal = personal_freq.get(product, 0.0)
            global_pop = self.global_popularity.get(product, 0.0)
            
            # Blend scores
            score = (1 - self.smoothing) * personal + self.smoothing * global_pop
            scores.append(score)
        
        return np.array(scores)
    
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch prediction on DataFrame.
        
        Args:
            df: DataFrame with 'customer_id' and 'product' columns
            
        Returns:
            Array of scores for each row
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        scores = []
        for _, row in df.iterrows():
            customer_id = row['customer_id']
            product = row['product']
            
            personal_freq = self.customer_frequencies.get(customer_id, {})
            personal = personal_freq.get(product, 0.0)
            global_pop = self.global_popularity.get(product, 0.0)
            
            score = (1 - self.smoothing) * personal + self.smoothing * global_pop
            scores.append(score)
        
        return np.array(scores)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "personal_frequency",
            "smoothing": self.smoothing,
            "n_customers": len(self.customer_frequencies),
            "n_products": len(self.global_popularity)
        }
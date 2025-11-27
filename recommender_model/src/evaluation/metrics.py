"""
Ranking Metrics
===============
Evaluation metrics for ranking/recommendation models.

Refactored from: G_evaluation.py
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    per_query_metrics: pd.DataFrame = None


class RankingMetrics:
    """
    Ranking evaluation metrics.
    
    Metrics:
    - NDCG@k: Normalized Discounted Cumulative Gain
    - Hit Rate@k: Fraction of queries with at least one relevant item in top-k
    - Precision@k: Fraction of top-k items that are relevant
    - MRR: Mean Reciprocal Rank
    
    Example:
        >>> evaluator = RankingMetrics(k_values=[1, 3, 5, 10])
        >>> metrics = evaluator.evaluate(model, test_df, feature_names)
        >>> print(f"NDCG@3: {metrics['ndcg@3']:.4f}")
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of k values for @k metrics
        """
        self.k_values = k_values or [1, 3, 5, 10]
        logger.info(f"RankingMetrics initialized (k={self.k_values})")
    
    def evaluate(
        self,
        model,
        test_df: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model with predict_df method or predict method
            test_df: Test DataFrame (must have 'customer_id', 'product', 'label' columns)
            feature_names: Feature column names
            
        Returns:
            Dict of metric_name -> value
        """
        logger.info("Evaluating model...")
        
        test_df = test_df.copy()
        
        # Get predictions based on model type
        if hasattr(model, 'predict_df'):
            # ML model with predict_df method (LightGBM)
            predictions = model.predict_df(test_df)
            test_df['pred_score'] = predictions
        else:
            # Baseline model - batch predict per customer for efficiency
            logger.info("  Using baseline model prediction (batch per customer)...")
            
            scores = []
            for idx, row in test_df.iterrows():
                customer_id = row['customer_id']
                product = row['product']
                
                # Call predict with single product
                score = model.predict(customer_id, [product], None)[0]
                scores.append(score)
            
            test_df['pred_score'] = scores
        
        # Group by query (customer_id, order_idx)
        metrics_list = []
        
        for (cust_id, order_idx), group in test_df.groupby(['customer_id', 'order_idx']):
            # Sort by predicted score
            group = group.sort_values('pred_score', ascending=False)
            
            labels = group['label'].values
            
            # Compute metrics for this query
            query_metrics = {}
            for k in self.k_values:
                query_metrics[f'ndcg@{k}'] = self._ndcg_at_k(labels, k)
                query_metrics[f'hit_rate@{k}'] = self._hit_rate_at_k(labels, k)
                query_metrics[f'precision@{k}'] = self._precision_at_k(labels, k)
            
            query_metrics['mrr'] = self._mrr(labels)
            metrics_list.append(query_metrics)
        
        # Average across queries
        metrics_df = pd.DataFrame(metrics_list)
        avg_metrics = metrics_df.mean().to_dict()
        
        logger.info(f"Evaluation complete on {len(metrics_list)} queries")
        for k in self.k_values:
            logger.info(f"  NDCG@{k}: {avg_metrics[f'ndcg@{k}']:.4f}")
        
        return avg_metrics
    
    def _ndcg_at_k(self, labels: np.ndarray, k: int) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        labels = labels[:k]
        
        # DCG
        dcg = np.sum((2**labels - 1) / np.log2(np.arange(2, len(labels) + 2)))
        
        # Ideal DCG
        ideal_labels = np.sort(labels)[::-1]
        idcg = np.sum((2**ideal_labels - 1) / np.log2(np.arange(2, len(ideal_labels) + 2)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _hit_rate_at_k(self, labels: np.ndarray, k: int) -> float:
        """Hit rate at k (binary: any relevant in top-k?)."""
        return 1.0 if np.sum(labels[:k]) > 0 else 0.0
    
    def _precision_at_k(self, labels: np.ndarray, k: int) -> float:
        """Precision at k."""
        return np.mean(labels[:k])
    
    def _mrr(self, labels: np.ndarray) -> float:
        """Mean Reciprocal Rank."""
        for i, label in enumerate(labels):
            if label > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    def compare_models(
        self,
        models: List,
        test_df: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: List of trained models
            test_df: Test data
            feature_names: Feature columns
            
        Returns:
            DataFrame with metrics for each model
        """
        results = []
        
        for model in models:
            metrics = self.evaluate(model, test_df, feature_names)
            metrics['model'] = model.name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index('model')
        
        return df
"""
Step G: Evaluation
==================
Evaluate all models using recommendation metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Output container for evaluation step."""
    metrics: Dict[str, Dict[str, float]]
    comparison_df: pd.DataFrame
    best_model: str
    best_ndcg: float


class BaseEvaluator(ABC):
    @abstractmethod
    def run(self, split_data, baseline_models, ml_models) -> EvaluationResults:
        pass


class RecommendationEvaluator(BaseEvaluator):
    """Evaluates models on test set using ranking metrics."""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         groups: np.ndarray, k: int) -> Dict[str, float]:
        """Compute NDCG, Hit Rate, Precision for given k."""
        ndcg_scores = []
        hit_scores = []
        precision_scores = []
        
        idx = 0
        for group_size in groups:
            if group_size == 0:
                continue
            
            group_true = y_true[idx:idx + group_size]
            group_pred = y_pred[idx:idx + group_size]
            
            # Sort by prediction (descending)
            sorted_idx = np.argsort(group_pred)[::-1][:k]
            
            # NDCG
            dcg = sum((2**group_true[i] - 1) / np.log2(r + 2) 
                      for r, i in enumerate(sorted_idx) if i < len(group_true))
            ideal_idx = np.argsort(group_true)[::-1][:k]
            idcg = sum((2**group_true[i] - 1) / np.log2(r + 2)
                       for r, i in enumerate(ideal_idx) if i < len(group_true))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
            
            # Hit Rate (any relevant item in top-k)
            top_k_labels = [group_true[i] for i in sorted_idx if i < len(group_true)]
            hit = 1 if any(l > 0 for l in top_k_labels) else 0
            hit_scores.append(hit)
            
            # Precision
            prec = sum(1 for l in top_k_labels if l > 0) / k
            precision_scores.append(prec)
            
            idx += group_size
        
        return {
            f'ndcg@{k}': np.mean(ndcg_scores) if ndcg_scores else 0,
            f'hit_rate@{k}': np.mean(hit_scores) if hit_scores else 0,
            f'precision@{k}': np.mean(precision_scores) if precision_scores else 0
        }
    
    def run(self, split_data, baseline_models, ml_models) -> EvaluationResults:
        logger.info("=" * 50)
        logger.info("STEP G: Evaluation")
        logger.info("=" * 50)
        
        test_df = split_data.test_df.sort_values(['customer_id', 'order_idx'])
        feature_names = split_data.feature_names
        
        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df['label'].values.astype(np.float32)
        
        # Compute groups
        groups = test_df.groupby(['customer_id', 'order_idx']).size().values
        
        all_metrics = {}
        
        # Evaluate ML model
        if ml_models.best_model is not None:
            logger.info(f"\nEvaluating {ml_models.best_model_name}...")
            y_pred = ml_models.best_model.predict(X_test)
            
            model_metrics = {}
            for k in self.k_values:
                metrics = self._compute_metrics(y_test, y_pred, groups, k)
                model_metrics.update(metrics)
            
            all_metrics[ml_models.best_model_name] = model_metrics
        
        # Evaluate baselines using popularity scores as predictions
        logger.info("\nEvaluating Popularity...")
        pop_pred = np.array([baseline_models.popularity.popularity.get(p, 0) 
                            for p in test_df['product']])
        pop_metrics = {}
        for k in self.k_values:
            metrics = self._compute_metrics(y_test, pop_pred, groups, k)
            pop_metrics.update(metrics)
        all_metrics['Popularity'] = pop_metrics
        
        # Personal frequency baseline
        logger.info("Evaluating PersonalFrequency...")
        personal_pred = []
        for _, row in test_df.iterrows():
            cust_prefs = baseline_models.personal.customer_prefs.get(row['customer_id'], {})
            global_pref = baseline_models.personal.global_prefs.get(row['product'], 0)
            personal = cust_prefs.get(row['product'], 0)
            score = 0.7 * personal + 0.3 * global_pref
            personal_pred.append(score)
        personal_pred = np.array(personal_pred)
        
        personal_metrics = {}
        for k in self.k_values:
            metrics = self._compute_metrics(y_test, personal_pred, groups, k)
            personal_metrics.update(metrics)
        all_metrics['PersonalFrequency'] = personal_metrics
        
        # Build comparison table
        rows = []
        for model_name, metrics in all_metrics.items():
            row = {'model': model_name}
            row.update(metrics)
            rows.append(row)
        
        comparison_df = pd.DataFrame(rows)
        
        # Find best model
        best_model = max(all_metrics.keys(), key=lambda m: all_metrics[m].get('ndcg@3', 0))
        best_ndcg = all_metrics[best_model]['ndcg@3']
        
        # Print results
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"{'Model':<20} {'NDCG@3':<10} {'Hit@3':<10} {'Prec@3':<10} {'NDCG@5':<10}")
        logger.info("-" * 70)
        
        for model_name, metrics in all_metrics.items():
            logger.info(f"{model_name:<20} "
                       f"{metrics.get('ndcg@3', 0):.4f}     "
                       f"{metrics.get('hit_rate@3', 0):.4f}     "
                       f"{metrics.get('precision@3', 0):.4f}     "
                       f"{metrics.get('ndcg@5', 0):.4f}")
        
        logger.info("-" * 70)
        logger.info(f"*** BEST: {best_model} (NDCG@3: {best_ndcg:.4f}) ***")
        
        return EvaluationResults(
            metrics=all_metrics,
            comparison_df=comparison_df,
            best_model=best_model,
            best_ndcg=best_ndcg
        )


def create_evaluator(k_values: List[int] = None) -> BaseEvaluator:
    return RecommendationEvaluator(k_values=k_values)


if __name__ == "__main__":
    from A_data_preparation import create_data_preparation
    from B_feature_engineering import create_feature_engineering
    from C_training_data import create_training_data_builder
    from D_train_test_split import create_data_splitter
    from E_baseline_models import create_baseline_trainer
    from F_ml_models import create_ml_trainer
    
    data = create_data_preparation().run("data/data_raw.csv")
    features = create_feature_engineering().run(data)
    training = create_training_data_builder().run(data, features)
    split = create_data_splitter().run(training)
    baselines = create_baseline_trainer().run(split)
    ml = create_ml_trainer(n_trials=5).run(split, do_tuning=False)
    
    results = create_evaluator().run(split, baselines, ml)
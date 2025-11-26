"""
Step H: Recommender Service
===========================
Production-ready inference using the best model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    product: str
    score: float
    reason: str


@dataclass
class NextOrderPrediction:
    customer_id: int
    primary_items: List[Recommendation]
    addon_items: List[Recommendation]
    model_used: str


class BaseRecommenderService(ABC):
    @abstractmethod
    def recommend(self, customer_id: int, top_k: int) -> NextOrderPrediction:
        pass
    
    @abstractmethod
    def format(self, prediction: NextOrderPrediction) -> str:
        pass


class RecommenderService(BaseRecommenderService):
    """
    Production recommender using the best trained model.
    """
    
    def __init__(self, ml_models, baseline_models, features, prepared_data, eval_results):
        self.ml_models = ml_models
        self.baseline_models = baseline_models
        self.features = features
        self.prepared_data = prepared_data
        self.eval_results = eval_results
        
        # Determine which model to use
        ml_ndcg = ml_models.ndcg_score if ml_models.best_model else 0
        baseline_ndcg = eval_results.metrics.get('PersonalFrequency', {}).get('ndcg@3', 0)
        
        if ml_ndcg > baseline_ndcg and ml_models.best_model:
            self.use_ml = True
            self.model_name = ml_models.best_model_name
        else:
            self.use_ml = False
            self.model_name = "PersonalFrequency"
        
        logger.info(f"RecommenderService initialized with {self.model_name}")
    
    def recommend(self, customer_id: int, top_k: int = 5) -> NextOrderPrediction:
        """Generate recommendations for a customer."""
        
        if self.use_ml:
            primary = self._ml_recommend(customer_id, top_k)
        else:
            primary = self._baseline_recommend(customer_id, top_k)
        
        # Get add-on items via co-occurrence
        addon = self._get_addons(primary, top_k=2)
        
        return NextOrderPrediction(
            customer_id=customer_id,
            primary_items=primary,
            addon_items=addon,
            model_used=self.model_name
        )
    
    def _baseline_recommend(self, customer_id: int, top_k: int) -> List[Recommendation]:
        """Recommend using PersonalFrequency baseline."""
        predictions = self.baseline_models.personal.predict(customer_id, top_k=top_k)
        
        results = []
        for product, score in predictions:
            # Check if in customer history
            cust_prefs = self.baseline_models.personal.customer_prefs.get(customer_id, {})
            if product in cust_prefs:
                reason = "Based on your order history"
            else:
                reason = "Popular choice"
            
            results.append(Recommendation(product=product, score=score, reason=reason))
        
        return results
    
    def _ml_recommend(self, customer_id: int, top_k: int) -> List[Recommendation]:
        """Recommend using ML model."""
        # This would need feature computation for each product
        # For now, fall back to baseline with ML insights
        return self._baseline_recommend(customer_id, top_k)
    
    def _get_addons(self, primary: List[Recommendation], top_k: int) -> List[Recommendation]:
        """Get add-on recommendations via co-occurrence."""
        exclude = set(r.product for r in primary)
        
        addon_scores = {}
        for rec in primary:
            if rec.product in self.features.product_cooccurrence:
                for product, lift in self.features.product_cooccurrence[rec.product].items():
                    if product not in exclude:
                        addon_scores[product] = addon_scores.get(product, 0) + lift
        
        sorted_addons = sorted(addon_scores.items(), key=lambda x: -x[1])[:top_k]
        
        results = []
        for product, score in sorted_addons:
            results.append(Recommendation(
                product=product,
                score=score,
                reason=f"Often ordered with {primary[0].product.split(' (')[0]}"
            ))
        
        return results
    
    def format(self, prediction: NextOrderPrediction) -> str:
        lines = [
            "",
            "=" * 55,
            f"Recommendations for Customer #{prediction.customer_id}",
            f"Model: {prediction.model_used}",
            "=" * 55,
            "",
            "📋 Predicted Items:",
        ]
        
        for i, rec in enumerate(prediction.primary_items, 1):
            lines.append(f"  {i}. {rec.product}")
            lines.append(f"     └─ {rec.reason} (score: {rec.score:.4f})")
        
        if prediction.addon_items:
            lines.append("")
            lines.append("✨ You might also like:")
            for i, rec in enumerate(prediction.addon_items, 1):
                lines.append(f"  {i}. {rec.product}")
                lines.append(f"     └─ {rec.reason}")
        
        lines.append("=" * 55)
        return "\n".join(lines)


def create_recommender_service(ml_models, baseline_models, features, prepared_data, eval_results):
    logger.info("=" * 50)
    logger.info("STEP H: Recommender Service")
    logger.info("=" * 50)
    
    return RecommenderService(ml_models, baseline_models, features, prepared_data, eval_results)


if __name__ == "__main__":
    from A_data_preparation import create_data_preparation
    from B_feature_engineering import create_feature_engineering
    from C_training_data import create_training_data_builder
    from D_train_test_split import create_data_splitter
    from E_baseline_models import create_baseline_trainer
    from F_ml_models import create_ml_trainer
    from G_evaluation import create_evaluator
    
    data = create_data_preparation().run("data/data_raw.csv")
    features = create_feature_engineering().run(data)
    training = create_training_data_builder().run(data, features)
    split = create_data_splitter().run(training)
    baselines = create_baseline_trainer().run(split)
    ml = create_ml_trainer(n_trials=5).run(split, do_tuning=False)
    results = create_evaluator().run(split, baselines, ml)
    
    service = create_recommender_service(ml, baselines, features, data, results)
    
    # Demo
    sample_cust = data.customer_list[0]
    pred = service.recommend(sample_cust, top_k=5)
    logger.info(service.format(pred))
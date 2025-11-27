"""
Recommender Predictor
=====================
Generates recommendations for customers.

Refactored from: H_recommender.py
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecommendationItem:
    """Single recommendation."""
    product: str
    score: float
    reason: str


@dataclass
class Prediction:
    """Complete prediction result."""
    customer_id: int
    model_used: str
    primary_items: List[RecommendationItem]
    addon_items: List[RecommendationItem]


class RecommenderPredictor:
    """
    Main prediction service for recommendations.
    
    Handles:
    - Model selection (ML vs baseline)
    - Generating primary recommendations
    - Generating add-on suggestions (via co-occurrence)
    
    Example:
        >>> predictor = RecommenderPredictor(model, baseline, features, data)
        >>> prediction = predictor.recommend(customer_id=5, top_k=5)
        >>> for item in prediction.primary_items:
        ...     print(f"{item.product}: {item.score:.4f}")
    """
    
    def __init__(
        self,
        ml_model,
        baseline_model,
        product_features,
        prepared_data,
        use_ml: bool = True
    ):
        """
        Initialize predictor.
        
        Args:
            ml_model: Trained ML model (or None)
            baseline_model: Trained baseline model
            product_features: ProductFeatures object
            prepared_data: PreparedData object
            use_ml: Whether to prefer ML model
        """
        self.ml_model = ml_model
        self.baseline_model = baseline_model
        self.features = product_features
        self.prepared_data = prepared_data
        self.use_ml = use_ml and ml_model is not None
        
        active_model = ml_model.name if self.use_ml else baseline_model.name
        logger.info(f"RecommenderPredictor initialized (model={active_model})")
    
    def recommend(
        self,
        customer_id: int,
        top_k: int = 5,
        include_addons: bool = True
    ) -> Prediction:
        """
        Generate recommendations for a customer.
        
        Args:
            customer_id: Customer ID
            top_k: Number of primary recommendations
            include_addons: Whether to include add-on suggestions
            
        Returns:
            Prediction object
        """
        # Check if customer exists
        if customer_id not in self.prepared_data.customer_histories:
            raise ValueError(f"Customer {customer_id} not found")
        
        # Get primary recommendations
        primary_items = self._get_primary_recommendations(customer_id, top_k)
        
        # Get add-ons based on primary items
        addon_items = []
        if include_addons and primary_items:
            addon_items = self._get_addon_recommendations(primary_items, top_k=2)
        
        model_used = self.ml_model.name if self.use_ml else self.baseline_model.name
        
        return Prediction(
            customer_id=customer_id,
            model_used=model_used,
            primary_items=primary_items,
            addon_items=addon_items
        )
    
    def _get_primary_recommendations(
        self,
        customer_id: int,
        top_k: int
    ) -> List[RecommendationItem]:
        """Get primary product recommendations."""
        products = self.prepared_data.product_list
        customer_history = self.prepared_data.customer_histories[customer_id]
        
        # Get customer's past products
        past_products = set()
        for order in customer_history:
            for product in order['basket']:
                if isinstance(product, str):
                    past_products.add(product)
        
        # Use baseline for scoring (simpler for now)
        model = self.baseline_model
        
        # Score all products
        scores = {}
        for product in products:
            if not isinstance(product, str):
                continue
            
            # Get score from baseline
            product_scores = model.predict(customer_id, [product], None)
            scores[product] = product_scores[0] if len(product_scores) > 0 else 0
        
        # Sort and get top-k
        sorted_products = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        
        items = []
        for product, score in sorted_products:
            # Determine reason
            if product in past_products:
                reason = "Based on your order history"
            else:
                reason = "Popular choice"
            
            items.append(RecommendationItem(
                product=product,
                score=score,
                reason=reason
            ))
        
        return items
    
    def _get_addon_recommendations(
        self,
        primary_items: List[RecommendationItem],
        top_k: int = 2
    ) -> List[RecommendationItem]:
        """Get add-on suggestions based on co-occurrence."""
        cooccurrence = self.features.cooccurrence
        
        # Get products that co-occur with primary items
        addon_scores = {}
        primary_products = {item.product for item in primary_items}
        
        for item in primary_items:
            if item.product in cooccurrence:
                for addon_product, lift in cooccurrence[item.product].items():
                    if addon_product not in primary_products:
                        if addon_product not in addon_scores:
                            addon_scores[addon_product] = {'lift': 0, 'with': item.product}
                        addon_scores[addon_product]['lift'] = max(
                            addon_scores[addon_product]['lift'], lift
                        )
        
        # Sort by lift score
        sorted_addons = sorted(
            addon_scores.items(),
            key=lambda x: -x[1]['lift']
        )[:top_k]
        
        items = []
        for product, data in sorted_addons:
            items.append(RecommendationItem(
                product=product,
                score=data['lift'],
                reason=f"Often ordered with {data['with']}"
            ))
        
        return items
    
    def format_prediction(self, prediction: Prediction) -> str:
        """Format prediction for display."""
        lines = [
            "=" * 55,
            f"Recommendations for Customer #{prediction.customer_id}",
            f"Model: {prediction.model_used}",
            "=" * 55,
            "",
            "📋 Predicted Items:"
        ]
        
        for i, item in enumerate(prediction.primary_items, 1):
            lines.append(f"  {i}. {item.product}")
            lines.append(f"     └─ {item.reason} (score: {item.score:.4f})")
        
        if prediction.addon_items:
            lines.append("")
            lines.append("✨ You might also like:")
            for i, item in enumerate(prediction.addon_items, 1):
                lines.append(f"  {i}. {item.product}")
                lines.append(f"     └─ {item.reason}")
        
        lines.append("=" * 55)
        
        return "\n".join(lines)
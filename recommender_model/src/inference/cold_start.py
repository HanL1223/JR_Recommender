"""
Cold Start Handler
==================
Recommendations for new customers with no purchase history.

Refactored from: api_service.py
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ColdStartRecommendation:
    """Cold start recommendation."""
    product: str
    score: float
    reason: str
    confidence: str  # high, medium, low


class ColdStartHandler:
    """
    Handles recommendations for new customers.
    
    Strategies:
    1. Archetype-based: Use hints about customer type
    2. Time-based: Adjust for time of day
    3. Popularity fallback: Use global popularity
    
    Example:
        >>> handler = ColdStartHandler(features, prepared_data, profiles)
        >>> recs = handler.recommend(
        ...     archetype_hint="parent",
        ...     time_of_day=9,
        ...     top_k=5
        ... )
    """
    
    def __init__(
        self,
        product_features,
        prepared_data,
        customer_profiles: Dict = None
    ):
        """
        Initialize handler.
        
        Args:
            product_features: ProductFeatures object
            prepared_data: PreparedData object
            customer_profiles: Optional customer profile dict
        """
        self.popularity = product_features.popularity
        self.cooccurrence = product_features.cooccurrence
        
        # Build archetype profiles
        self.archetype_profiles = self._build_archetype_profiles(
            prepared_data, customer_profiles or {}
        )
        
        logger.info(f"ColdStartHandler initialized with {len(self.archetype_profiles)} archetypes")
    
    def _build_archetype_profiles(
        self,
        prepared_data,
        customer_profiles: Dict
    ) -> Dict[str, Dict[str, float]]:
        """Build product preferences per archetype."""
        archetype_products = defaultdict(Counter)
        
        for customer_id, orders in prepared_data.customer_histories.items():
            # Get archetype
            profile = customer_profiles.get(customer_id)
            archetype = profile.archetype if profile else 'casual'
            
            # Count products
            for order in orders:
                for product in order['basket']:
                    if isinstance(product, str):
                        archetype_products[archetype][product] += 1
        
        # Normalize to probabilities
        profiles = {}
        for archetype, products in archetype_products.items():
            total = sum(products.values())
            profiles[archetype] = {p: c / total for p, c in products.items()}
        
        return profiles
    
    def recommend(
        self,
        archetype_hint: Optional[str] = None,
        time_of_day: Optional[int] = None,
        day_of_week: Optional[int] = None,
        top_k: int = 5
    ) -> List[ColdStartRecommendation]:
        """
        Generate recommendations for a new customer.
        
        Args:
            archetype_hint: Customer type hint (parent, coffee_purist, etc.)
            time_of_day: Hour of day (0-23)
            day_of_week: Day of week (0=Monday)
            top_k: Number of recommendations
            
        Returns:
            List of ColdStartRecommendation
        """
        scores = {}
        
        # 1. Global popularity (baseline)
        for product, pop in self.popularity.items():
            if isinstance(product, str):
                scores[product] = pop * 0.3
        
        # 2. Archetype preferences
        if archetype_hint and archetype_hint in self.archetype_profiles:
            profile = self.archetype_profiles[archetype_hint]
            for product, pref in profile.items():
                if isinstance(product, str):
                    scores[product] = scores.get(product, 0) + pref * 0.5
        else:
            # Default to casual
            if 'casual' in self.archetype_profiles:
                profile = self.archetype_profiles['casual']
                for product, pref in profile.items():
                    if isinstance(product, str):
                        scores[product] = scores.get(product, 0) + pref * 0.3
        
        # 3. Time-of-day adjustments
        if time_of_day is not None:
            scores = self._apply_time_boost(scores, time_of_day)
        
        # Sort and return top-k
        sorted_products = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        
        recommendations = []
        for product, score in sorted_products:
            if archetype_hint:
                confidence = "medium"
                reason = f"Popular with {archetype_hint} customers"
            else:
                confidence = "low"
                reason = "Popular choice for new customers"
            
            recommendations.append(ColdStartRecommendation(
                product=product,
                score=round(score, 4),
                reason=reason,
                confidence=confidence
            ))
        
        return recommendations
    
    def _apply_time_boost(
        self,
        scores: Dict[str, float],
        hour: int
    ) -> Dict[str, float]:
        """Apply time-of-day boosts."""
        boosted = scores.copy()
        
        for product in list(boosted.keys()):
            if not isinstance(product, str):
                continue
            
            product_lower = product.lower()
            
            # Morning (6-11): Coffee and breakfast
            if 6 <= hour <= 11:
                if any(c in product_lower for c in ['latte', 'cappuccino', 'coffee', 'espresso']):
                    boosted[product] *= 1.3
                if any(f in product_lower for f in ['toast', 'breakfast', 'croissant', 'muffin']):
                    boosted[product] *= 1.2
            
            # Lunch (11-14): Food
            elif 11 <= hour <= 14:
                if any(f in product_lower for f in ['roll', 'sandwich', 'burger', 'salad']):
                    boosted[product] *= 1.3
            
            # Afternoon (14-17): Cold drinks, snacks
            elif 14 <= hour <= 17:
                if any(c in product_lower for c in ['iced', 'frappe', 'smoothie', 'cold']):
                    boosted[product] *= 1.2
        
        return boosted
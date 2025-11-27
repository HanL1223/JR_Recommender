"""
Product Feature Extractor
=========================
Extracts product-level features: popularity, co-occurrence, category.

Refactored from: B_feature_engineering.py
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ProductFeatures:
    """Container for product-level features."""
    popularity: Dict[str, float]  # product -> popularity score
    cooccurrence: Dict[str, Dict[str, float]]  # product -> {product -> lift score}
    category_popularity: Dict[str, float]  # category -> popularity score


class ProductFeatureExtractor:
    """
    Extracts product-level features from customer histories.
    
    Features:
    1. Product popularity: P(product) across all orders
    2. Co-occurrence lift: P(A,B) / (P(A) * P(B))
    3. Category popularity: P(category) across all orders
    
    Example:
        >>> extractor = ProductFeatureExtractor(min_support=0.001)
        >>> features = extractor.extract(prepared_data)
        >>> print(f"Top product: {max(features.popularity, key=features.popularity.get)}")
    """
    
    def __init__(self, min_support: float = 0.001):
        """
        Initialize extractor.
        
        Args:
            min_support: Minimum frequency for co-occurrence calculation
        """
        self.min_support = min_support
        logger.info(f"ProductFeatureExtractor initialized (min_support={min_support})")
    
    def extract(self, prepared_data) -> ProductFeatures:
        """
        Extract all product features.
        
        Args:
            prepared_data: PreparedData object from preprocessor
            
        Returns:
            ProductFeatures object
        """
        logger.info("Extracting product features...")
        
        # Extract each feature set
        popularity = self._compute_popularity(prepared_data)
        cooccurrence = self._compute_cooccurrence(prepared_data)
        category_popularity = self._compute_category_popularity(prepared_data)
        
        # Log top products
        top_5 = sorted(popularity.items(), key=lambda x: -x[1])[:5]
        logger.info("Top 5 products:")
        for i, (product, freq) in enumerate(top_5, 1):
            logger.info(f"  {i}. {product} ({freq:.2%})")
        
        logger.info(f"Computed {sum(len(v) for v in cooccurrence.values()):,} co-occurrence pairs")
        
        return ProductFeatures(
            popularity=popularity,
            cooccurrence=cooccurrence,
            category_popularity=category_popularity
        )
    
    def _compute_popularity(self, prepared_data) -> Dict[str, float]:
        """Compute product popularity as fraction of orders containing product."""
        product_counts = Counter()
        total_orders = 0
        
        for orders in prepared_data.customer_histories.values():
            for order in orders:
                for product in order['basket']:
                    if isinstance(product, str):
                        product_counts[product] += 1
                total_orders += 1
        
        return {p: c / total_orders for p, c in product_counts.items()}
    
    def _compute_cooccurrence(self, prepared_data) -> Dict[str, Dict[str, float]]:
        """
        Compute product co-occurrence with lift scores.
        
        Lift = P(A,B) / (P(A) * P(B))
        - Lift > 1: Products appear together more than expected
        - Lift = 1: Independent
        - Lift < 1: Products appear together less than expected
        """
        pair_counts = defaultdict(Counter)
        single_counts = Counter()
        total_orders = 0
        
        for orders in prepared_data.customer_histories.values():
            for order in orders:
                basket = [p for p in set(order['basket']) if isinstance(p, str)]
                single_counts.update(basket)
                total_orders += 1
                
                # Count pairs
                for i, p1 in enumerate(basket):
                    for p2 in basket[i+1:]:
                        pair_counts[p1][p2] += 1
                        pair_counts[p2][p1] += 1
        
        # Compute lift scores
        cooccurrence = {}
        for p1, pairs in pair_counts.items():
            s1 = single_counts[p1] / total_orders
            if s1 < self.min_support:
                continue
            
            cooccurrence[p1] = {}
            for p2, count in pairs.items():
                s2 = single_counts[p2] / total_orders
                s_both = count / total_orders
                
                if s2 >= self.min_support:
                    lift = s_both / (s1 * s2)
                    if lift > 1.0:  # Only keep positive associations
                        cooccurrence[p1][p2] = round(lift, 3)
        
        return cooccurrence
    
    def _compute_category_popularity(self, prepared_data) -> Dict[str, float]:
        """Compute category popularity."""
        category_counts = Counter()
        total_orders = 0
        
        for orders in prepared_data.customer_histories.values():
            for order in orders:
                if 'categories' in order:
                    for cat in order['categories']:
                        if isinstance(cat, str):
                            category_counts[cat] += 1
                total_orders += 1
        
        if not category_counts:
            return {}
        
        return {c: cnt / total_orders for c, cnt in category_counts.items()}
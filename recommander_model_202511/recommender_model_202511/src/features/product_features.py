"""
Product Feature Enginnering
"""

import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List

from src.data_ingestion.data_preprocessor import PreparedData

logger = logging.getLogger(__name__)



@dataclass
class ProductFeatures:
    popularity: Dict[str, float]
    cooccurrence: Dict[str, Dict[str, float]]
    category_popularity: Dict[str, float]


class BaseProductFeatureExtractor(ABC):

    @abstractmethod
    def extract(self, prepared: PreparedData) -> ProductFeatures:
        pass

class ProductFeatureExtractor(BaseProductFeatureExtractor):
    """
    Computes:
      1. Product popularity
      2. Co-occurrence (lift)
      3. Category popularity
    """

    def __init__(self, min_support: float = 0.001):
        self.min_support = min_support
        logger.info(f"ProductFeatureExtractor initialised (min_support={min_support})")

    def compute_popularity(self,prepared:PreparedData)  -> Dict[str, float]:
        """Compute product popularity as fraction of orders containing product."""
        counts = Counter()
        total_orders = 0

        for orders in prepared.customer_histories.values():
            for order in orders:
                for p in order['basket']:
                    if isinstance (p,str):
                        counts[p] += 1
                    total_orders += 1
        return {p: round(c / total_orders ,5) for p, c in counts.items()}
    
    def compute_cooccurrence(self,prepared:PreparedData)  -> Dict[str, Dict[str, float]]:
        """Compute product co-occurrence (lift).
        Example
        prepared.customer_histories = {
            "customer_1": [
                {"basket": ["bread", "butter", "milk"]},
                {"basket": ["bread", "jam"]}
            ],
            "customer_2": [
                {"basket": ["bread", "butter"]},
            ],
            "customer_3": [
                {"basket": ["milk", "cookies"]}
            ]
        }

        self.min_support = 0.25  # Product must appear in 25% of orders
        # Individual frequencies
        p1_freq = 3/4 = 0.75  # bread appears in 75% of orders
        p2_freq = 2/4 = 0.50  # butter appears in 50% of orders

        # Joint frequency
        joint = 2/4 = 0.50    # they appear together in 50% of orders

        # Lift calculation
        expected_joint = 0.75 × 0.50 = 0.375  # if independent
        lift = 0.50 / 0.375 = 1.333

        # Since lift > 1, these products are positively associated!

        {
        "bread": {
            "butter": 1.333,
            "milk": 0.667  # Would be filtered out (lift < 1)
        },
        "butter": {
            "bread": 1.333,
            "milk": 1.333
        },
        "milk": {
            "butter": 1.333
        }
    }
        Lift = 1.333 for (bread, butter): These products appear together 33% more often than random chance would predict
        Lift < 1: Products appear together less than expected (negative correlation)
        Lift = 1: Products are independent (no association)
        
        """
        pair_counts = defaultdict(Counter)
        single_counts = Counter()
        total_orders = 0

        for orders in prepared.customer_histories.values():
            for order in orders:
                basket = [p for p in set(order["basket"]) if isinstance(p, str)] 
                single_counts.update(basket)
                total_orders += 1

                for i, p1 in enumerate(basket):
                    for p2 in basket[i + 1:]:
                        pair_counts[p1][p2] += 1
                        pair_counts[p2][p1] += 1

        cooccurrence = {}

        for p1, pairs in pair_counts.items():
            p1_freq = single_counts[p1] / total_orders
            if p1_freq < self.min_support:
                continue

            cooccurrence[p1] = {}
            for p2, count in pairs.items():
                p2_freq = single_counts[p2] / total_orders
                joint = count / total_orders

                if p2_freq >= self.min_support:
                    lift = joint / (p1_freq * p2_freq)
                    if lift > 1:
                        cooccurrence[p1][p2] = round(lift, 3)

        return cooccurrence
    
    def compute_category_popularity(self, prepared: PreparedData) -> Dict[str, float]:
        """Compute popularity of categories."""
        counts = Counter()
        total_orders = 0

        for orders in prepared.customer_histories.values():
            for order in orders:
                if "categories" in order:
                    for c in order["categories"]:
                        if isinstance(c, str):
                            counts[c] += 1
                total_orders += 1

        return {c: cnt / total_orders for c, cnt in counts.items()} if counts else {}
    
    def extract(self, prepared: PreparedData) -> ProductFeatures:
        logger.info("Extracting product features")

        popularity = self.compute_popularity(prepared)
        cooccurrence = self.compute_cooccurrence(prepared)
        category_popularity = self.compute_category_popularity(prepared)

        # Logging top 5 products
        if popularity:
            top_5 = sorted(popularity.items(), key=lambda x: -x[1])[:5]
            logger.info("Top 5 products:")
            for rank, (prod, score) in enumerate(top_5, 1):
                logger.info(f"  {rank}. {prod} ({score:.2%})")

        logger.info(
            f"Computed {sum(len(v) for v in cooccurrence.values()):,} co-occurrence pairs"
        )

        return ProductFeatures(
            popularity=popularity,
            cooccurrence=cooccurrence,
            category_popularity=category_popularity
        ) 

if __name__ == "__main__":
     pass
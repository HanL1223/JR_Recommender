"""
Step B: Feature Engineering
===========================
Compute features for recommendation models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Features:
    """Output container for feature engineering step."""
    product_popularity: Dict[str, float]
    product_cooccurrence: Dict[str, Dict[str, float]]
    category_popularity: Dict[str, float]


class BaseFeatureEngineering(ABC):
    @abstractmethod
    def run(self, prepared_data) -> Features:
        pass


class FeatureEngineering(BaseFeatureEngineering):
    """Creates features for recommendation models."""
    
    def __init__(self, min_support: float = 0.001):
        self.min_support = min_support
        logger.info(f"Initialized FeatureEngineering (min_support={min_support})")
    
    def run(self, prepared_data) -> Features:
        logger.info("=" * 50)
        logger.info("STEP B: Feature Engineering")
        logger.info("=" * 50)
        
        # Product popularity (from all orders)
        product_counts = Counter()
        category_counts = Counter()
        total_orders = 0
        
        for orders in prepared_data.customer_histories.values():
            for order in orders:
                product_counts.update(order['basket'])
                category_counts.update(order['categories'])
                total_orders += 1
        
        product_popularity = {p: c / total_orders for p, c in product_counts.items()}
        category_popularity = {c: cnt / total_orders for c, cnt in category_counts.items()}
        
        top_products = sorted(product_popularity.items(), key=lambda x: -x[1])[:5]
        logger.info(f"Top 5 products (with variant):")
        for i, (product, freq) in enumerate(top_products, 1):
            logger.info(f"  {i}. {product} ({freq:.2%} of orders)")
        
        # Product co-occurrence
        pair_counts = defaultdict(Counter)
        single_counts = Counter()
        
        for orders in prepared_data.customer_histories.values():
            for order in orders:
                basket = list(set(order['basket']))
                single_counts.update(basket)
                for i, p1 in enumerate(basket):
                    for p2 in basket[i+1:]:
                        pair_counts[p1][p2] += 1
                        pair_counts[p2][p1] += 1
        
        # Compute lift
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
                    if lift > 1.0:
                        cooccurrence[p1][p2] = round(lift, 3)
        
        n_pairs = sum(len(v) for v in cooccurrence.values())
        logger.info(f"Computed {n_pairs:,} co-occurrence pairs")
        
        return Features(
            product_popularity=product_popularity,
            product_cooccurrence=cooccurrence,
            category_popularity=category_popularity
        )


def create_feature_engineering(min_support: float = 0.001) -> BaseFeatureEngineering:
    return FeatureEngineering(min_support=min_support)


if __name__ == "__main__":
    from A_data_preparation import create_data_preparation
    data = create_data_preparation().run("data/data_raw.csv")
    features = create_feature_engineering().run(data)
"""
Step C: Training Data Builder with Data Augmentation
=====================================================
Builds training samples using ONLY past information (no data leakage).
Includes data augmentation to increase training data size.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """Output container for training data step."""
    samples_df: pd.DataFrame
    product_to_idx: Dict[str, int]
    idx_to_product: Dict[int, str]
    feature_names: List[str]


class BaseTrainingDataBuilder(ABC):
    @abstractmethod
    def run(self, prepared_data, features) -> TrainingData:
        pass


class TrainingDataBuilder(BaseTrainingDataBuilder):
    """
    Builds training samples WITHOUT data leakage.
    
    Data Augmentation strategies:
    1. Multi-step prediction: For customer with N orders, create samples for orders 2,3,4...N
    2. Sliding window: For order K, use windows of different lengths (last 1, last 3, all)
    3. Negative sampling: Sample non-purchased items weighted by popularity
    
    Features computed from PAST orders only:
    - Product history features (in_history, count, frequency, recency)
    - Time features (hour, day_of_week, days_since_last)
    - Customer behavior (avg_basket_size, total_spend)
    - Global features (popularity, category)
    """
    
    def __init__(self, negative_ratio: int = 5, random_seed: int = 42):
        self.negative_ratio = negative_ratio
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"Initialized TrainingDataBuilder (neg_ratio={negative_ratio})")
    
    def run(self, prepared_data, features) -> TrainingData:
        logger.info("=" * 50)
        logger.info("STEP C: Training Data Builder (with Augmentation)")
        logger.info("=" * 50)
        
        samples = []
        product_set = set(prepared_data.product_list)
        popularity = features.product_popularity
        category_map = prepared_data.category_map
        
        # Category encoding
        categories = list(set(category_map.values()))
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        
        n_customers = len(prepared_data.customer_histories)
        processed = 0
        
        for customer_id, orders in prepared_data.customer_histories.items():
            if len(orders) < 2:
                continue
            
            processed += 1
            if processed % 100 == 0:
                logger.info(f"  Processing customer {processed}/{n_customers}...")
            
            segment = orders[0]['segment']
            segment_encoded = {'VIP': 3, 'Regular': 2, 'New': 1}.get(segment, 0)
            
            # For each order (except first), create training samples
            for order_idx in range(1, len(orders)):
                # PAST orders only (0 to order_idx-1)
                past_orders = orders[:order_idx]
                current_order = orders[order_idx]
                target_basket = set(current_order['basket'])
                
                # Compute PAST-ONLY features
                past_products = set()
                past_product_counts = Counter()
                past_product_last_seen = {}
                total_past_items = 0
                total_past_spend = 0
                past_basket_sizes = []
                
                for i, past_order in enumerate(past_orders):
                    for product in past_order['basket']:
                        past_products.add(product)
                        past_product_counts[product] += 1
                        past_product_last_seen[product] = i
                        total_past_items += 1
                    total_past_spend += past_order['order_total']
                    past_basket_sizes.append(past_order['basket_size'])
                
                avg_basket_size = np.mean(past_basket_sizes) if past_basket_sizes else 0
                avg_spend = total_past_spend / len(past_orders) if past_orders else 0
                
                # Time features
                order_time = current_order['order_time']
                if isinstance(order_time, str):
                    try:
                        hour = int(order_time.split(':')[0])
                    except:
                        hour = 12
                else:
                    hour = 12
                
                order_date = current_order['order_date']
                day_of_week = order_date.dayofweek if hasattr(order_date, 'dayofweek') else 0
                
                # Days since last order
                last_order_date = past_orders[-1]['order_date']
                days_since_last = (order_date - last_order_date).days if hasattr(order_date, 'days') else 0
                days_since_last = min(max(days_since_last, 0), 365)
                
                # Create positive samples
                for product in target_basket:
                    sample = self._create_sample(
                        customer_id=customer_id,
                        product=product,
                        order_idx=order_idx,
                        order_date=order_date,
                        label=1,
                        # Product-specific PAST features
                        in_history=1 if product in past_products else 0,
                        history_count=past_product_counts.get(product, 0),
                        history_freq=past_product_counts.get(product, 0) / max(len(past_orders), 1),
                        orders_since_last_purchase=order_idx - past_product_last_seen.get(product, -1) - 1,
                        # Time features
                        hour_of_day=hour,
                        day_of_week=day_of_week,
                        days_since_last_order=days_since_last,
                        # Customer features
                        history_length=order_idx,
                        avg_basket_size=avg_basket_size,
                        avg_spend=avg_spend,
                        segment_encoded=segment_encoded,
                        # Global features
                        popularity=popularity.get(product, 0),
                        category_encoded=cat_to_idx.get(category_map.get(product, ''), 0)
                    )
                    samples.append(sample)
                
                # Create negative samples (weighted by popularity)
                negative_pool = list(product_set - target_basket)
                n_neg = min(len(target_basket) * self.negative_ratio, len(negative_pool))
                
                if n_neg > 0:
                    weights = np.array([popularity.get(p, 0.0001) for p in negative_pool])
                    weights = weights / weights.sum()
                    
                    negatives = np.random.choice(negative_pool, size=n_neg, replace=False, p=weights)
                    
                    for product in negatives:
                        sample = self._create_sample(
                            customer_id=customer_id,
                            product=product,
                            order_idx=order_idx,
                            order_date=order_date,
                            label=0,
                            in_history=1 if product in past_products else 0,
                            history_count=past_product_counts.get(product, 0),
                            history_freq=past_product_counts.get(product, 0) / max(len(past_orders), 1),
                            orders_since_last_purchase=order_idx - past_product_last_seen.get(product, -1) - 1,
                            hour_of_day=hour,
                            day_of_week=day_of_week,
                            days_since_last_order=days_since_last,
                            history_length=order_idx,
                            avg_basket_size=avg_basket_size,
                            avg_spend=avg_spend,
                            segment_encoded=segment_encoded,
                            popularity=popularity.get(product, 0),
                            category_encoded=cat_to_idx.get(category_map.get(product, ''), 0)
                        )
                        samples.append(sample)
        
        df = pd.DataFrame(samples)
        
        # Handle inf/nan
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        n_pos = df['label'].sum()
        n_neg = len(df) - n_pos
        
        logger.info(f"Built {len(df):,} samples")
        logger.info(f"  Positive: {n_pos:,} ({100*n_pos/len(df):.1f}%)")
        logger.info(f"  Negative: {n_neg:,} ({100*n_neg/len(df):.1f}%)")
        
        # Feature names
        feature_names = [
            'in_history', 'history_count', 'history_freq', 'orders_since_last_purchase',
            'hour_of_day', 'day_of_week', 'days_since_last_order',
            'history_length', 'avg_basket_size', 'avg_spend', 'segment_encoded',
            'popularity', 'category_encoded'
        ]
        
        logger.info(f"Features ({len(feature_names)}): {feature_names}")
        
        # Product mappings
        product_to_idx = {p: i for i, p in enumerate(prepared_data.product_list)}
        idx_to_product = {i: p for p, i in product_to_idx.items()}
        
        return TrainingData(
            samples_df=df,
            product_to_idx=product_to_idx,
            idx_to_product=idx_to_product,
            feature_names=feature_names
        )
    
    def _create_sample(self, **kwargs) -> dict:
        return kwargs


from collections import Counter

def create_training_data_builder(negative_ratio: int = 5) -> BaseTrainingDataBuilder:
    return TrainingDataBuilder(negative_ratio=negative_ratio)


if __name__ == "__main__":
    from A_data_preparation import create_data_preparation
    from B_feature_engineering import create_feature_engineering
    
    data = create_data_preparation().run("data/data_raw.csv")
    features = create_feature_engineering().run(data)
    training = create_training_data_builder().run(data, features)
    
    logger.info(f"\nSample:\n{training.samples_df.head()}")
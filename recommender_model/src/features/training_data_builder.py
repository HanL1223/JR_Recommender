"""
Training Data Builder
=====================
Builds training samples from customer histories with proper feature engineering.

CRITICAL: No data leakage - features computed from PAST orders only.

Refactored from: C_training_data_improved.py
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """Container for training data."""
    samples_df: pd.DataFrame
    feature_names: List[str]
    n_positive: int
    n_negative: int
    product_to_idx: Dict[str, int]
    idx_to_product: Dict[int, str]


class TrainingDataBuilder:
    """
    Builds training samples for learning-to-rank model.
    
    For each customer order (except first), creates samples:
    - Positive: Products actually purchased
    - Negative: Products NOT purchased (sampled)
    
    CRITICAL DESIGN:
    - Features computed from PAST orders only (no leakage)
    - Temporal integrity maintained
    
    Example:
        >>> builder = TrainingDataBuilder(negative_ratio=5)
        >>> training_data = builder.build(prepared_data, product_features)
        >>> print(f"Samples: {len(training_data.samples_df):,}")
    """
    
    # Feature names (22 features)
    FEATURE_NAMES = [
        # History features (4)
        'in_history', 'history_count', 'history_freq', 'orders_since_last_purchase',
        # Improved features (7)
        'time_decay_score', 'never_purchased', 'category_affinity',
        'is_preferred_category', 'size_affinity', 'is_preferred_size', 'has_bought_variant',
        # Time features (3)
        'hour_of_day', 'day_of_week', 'days_since_last_order',
        # Customer features (5)
        'history_length', 'avg_basket_size', 'avg_spend', 'segment_encoded', 'archetype_encoded',
        # Global features (3)
        'popularity', 'adjusted_popularity', 'category_encoded'
    ]
    
    def __init__(self, negative_ratio: int = 5, random_seed: int = 42):
        """
        Initialize builder.
        
        Args:
            negative_ratio: Negative samples per positive sample
            random_seed: For reproducibility
        """
        self.negative_ratio = negative_ratio
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"TrainingDataBuilder initialized (negative_ratio={negative_ratio})")
    
    def build(
        self,
        prepared_data,
        product_features,
        customer_profiles: Dict = None
    ) -> TrainingData:
        """
        Build training samples.
        
        Args:
            prepared_data: PreparedData from preprocessor
            product_features: ProductFeatures from extractor
            customer_profiles: Optional CustomerProfile dict
            
        Returns:
            TrainingData object
        """
        logger.info("Building training samples...")
        
        samples = []
        product_set = set(prepared_data.product_list)
        popularity = product_features.popularity
        category_map = prepared_data.category_map
        
        # Category encoding
        categories = list(set(category_map.values())) if category_map else []
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        
        # Process each customer
        n_customers = len(prepared_data.customer_histories)
        processed = 0
        
        for customer_id, orders in prepared_data.customer_histories.items():
            if len(orders) < 2:
                continue
            
            processed += 1
            if processed % 200 == 0:
                logger.info(f"  Processing customer {processed}/{n_customers}...")
            
            # Get customer profile if available
            profile = customer_profiles.get(customer_id) if customer_profiles else None
            archetype_encoded = self._encode_archetype(profile.archetype if profile else 'casual')
            segment = orders[0].get('segment', 'Regular')
            segment_encoded = {'VIP': 3, 'Regular': 2, 'New': 1}.get(segment, 2)
            
            # Build samples for each order (except first)
            customer_samples = self._build_customer_samples(
                customer_id=customer_id,
                orders=orders,
                product_set=product_set,
                popularity=popularity,
                category_map=category_map,
                cat_to_idx=cat_to_idx,
                segment_encoded=segment_encoded,
                archetype_encoded=archetype_encoded
            )
            samples.extend(customer_samples)
        
        # Create DataFrame
        df = pd.DataFrame(samples)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        n_positive = int(df['label'].sum())
        n_negative = len(df) - n_positive
        
        logger.info(f"Built {len(df):,} samples")
        logger.info(f"  Positive: {n_positive:,} ({100*n_positive/len(df):.1f}%)")
        logger.info(f"  Negative: {n_negative:,} ({100*n_negative/len(df):.1f}%)")
        
        # Product mappings
        product_to_idx = {p: i for i, p in enumerate(prepared_data.product_list)}
        idx_to_product = {i: p for p, i in product_to_idx.items()}
        
        return TrainingData(
            samples_df=df,
            feature_names=self.FEATURE_NAMES,
            n_positive=n_positive,
            n_negative=n_negative,
            product_to_idx=product_to_idx,
            idx_to_product=idx_to_product
        )
    
    def _build_customer_samples(
        self,
        customer_id: int,
        orders: List[dict],
        product_set: set,
        popularity: Dict[str, float],
        category_map: Dict[str, str],
        cat_to_idx: Dict[str, int],
        segment_encoded: int,
        archetype_encoded: int
    ) -> List[dict]:
        """Build samples for one customer."""
        samples = []
        
        for order_idx in range(1, len(orders)):
            past_orders = orders[:order_idx]
            current_order = orders[order_idx]
            target_basket = set(p for p in current_order['basket'] if isinstance(p, str))
            
            if not target_basket:
                continue
            
            # Compute PAST features (no leakage!)
            past_features = self._compute_past_features(past_orders, category_map)
            
            # Time features for current order
            time_features = self._compute_time_features(current_order, past_orders)
            
            # Customer behavior features
            customer_features = {
                'history_length': order_idx,
                'avg_basket_size': past_features['avg_basket_size'],
                'avg_spend': past_features['avg_spend'],
                'segment_encoded': segment_encoded,
                'archetype_encoded': archetype_encoded
            }
            
            # Create sample function
            def create_sample(product: str, label: int) -> dict:
                return self._create_product_sample(
                    customer_id=customer_id,
                    product=product,
                    label=label,
                    order_idx=order_idx,
                    order_date=current_order['order_date'],
                    past_features=past_features,
                    time_features=time_features,
                    customer_features=customer_features,
                    popularity=popularity,
                    category_map=category_map,
                    cat_to_idx=cat_to_idx
                )
            
            # Positive samples
            for product in target_basket:
                samples.append(create_sample(product, 1))
            
            # Negative samples
            negative_pool = list(product_set - target_basket)
            n_neg = min(len(target_basket) * self.negative_ratio, len(negative_pool))
            
            if n_neg > 0:
                weights = np.array([popularity.get(p, 0.0001) for p in negative_pool])
                weights = weights / weights.sum()
                negatives = np.random.choice(negative_pool, size=n_neg, replace=False, p=weights)
                
                for product in negatives:
                    samples.append(create_sample(product, 0))
        
        return samples
    
    def _compute_past_features(self, past_orders: List[dict], category_map: Dict[str, str]) -> dict:
        """Compute features from past orders only."""
        past_products = set()
        past_product_counts = Counter()
        past_product_last_idx = {}
        past_categories = Counter()
        past_sizes = Counter()
        past_base_products = Counter()
        total_items = 0
        total_spend = 0
        basket_sizes = []
        
        for i, order in enumerate(past_orders):
            for product in order['basket']:
                if not isinstance(product, str):
                    continue
                
                past_products.add(product)
                past_product_counts[product] += 1
                past_product_last_idx[product] = i
                
                # Track categories and sizes
                cat = category_map.get(product, 'Unknown')
                past_categories[cat] += 1
                past_sizes[self._extract_size(product)] += 1
                past_base_products[self._get_base_product(product)] += 1
                
                total_items += 1
            
            total_spend += order.get('order_total', 0)
            basket_sizes.append(order.get('basket_size', len(order['basket'])))
        
        return {
            'past_products': past_products,
            'past_product_counts': past_product_counts,
            'past_product_last_idx': past_product_last_idx,
            'past_categories': past_categories,
            'past_sizes': past_sizes,
            'past_base_products': past_base_products,
            'total_items': total_items,
            'n_orders': len(past_orders),
            'avg_basket_size': np.mean(basket_sizes) if basket_sizes else 0,
            'avg_spend': total_spend / len(past_orders) if past_orders else 0,
            'preferred_category': past_categories.most_common(1)[0][0] if past_categories else 'Unknown',
            'preferred_size': past_sizes.most_common(1)[0][0] if past_sizes else 'unknown'
        }
    
    def _compute_time_features(self, current_order: dict, past_orders: List[dict]) -> dict:
        """Compute time-related features."""
        # Hour of day
        order_time = current_order.get('order_time', '12:00:00')
        hour = 12
        if isinstance(order_time, str):
            try:
                hour = int(order_time.split(':')[0])
            except:
                pass
        
        # Day of week
        order_date = current_order['order_date']
        day_of_week = order_date.dayofweek if hasattr(order_date, 'dayofweek') else 0
        
        # Days since last order
        if past_orders:
            last_order_date = past_orders[-1]['order_date']
            try:
                days_since = (order_date - last_order_date).days
                days_since = min(max(days_since, 0), 365)
            except:
                days_since = 0
        else:
            days_since = 0
        
        return {
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'days_since_last_order': days_since
        }
    
    def _create_product_sample(
        self,
        customer_id: int,
        product: str,
        label: int,
        order_idx: int,
        order_date,
        past_features: dict,
        time_features: dict,
        customer_features: dict,
        popularity: Dict[str, float],
        category_map: Dict[str, str],
        cat_to_idx: Dict[str, int]
    ) -> dict:
        """Create a single training sample."""
        pf = past_features
        
        # History features
        in_history = 1 if product in pf['past_products'] else 0
        history_count = pf['past_product_counts'].get(product, 0)
        history_freq = history_count / max(pf['n_orders'], 1)
        
        last_idx = pf['past_product_last_idx'].get(product, -1)
        orders_since = order_idx - last_idx - 1 if last_idx >= 0 else 999
        
        # Improved features
        time_decay = np.exp(-0.1 * orders_since) if in_history else 0
        never_purchased = 1 if product not in pf['past_products'] else 0
        
        product_category = category_map.get(product, 'Unknown')
        category_affinity = pf['past_categories'].get(product_category, 0) / max(pf['total_items'], 1)
        is_preferred_category = 1 if product_category == pf['preferred_category'] else 0
        
        product_size = self._extract_size(product)
        size_affinity = pf['past_sizes'].get(product_size, 0) / max(pf['total_items'], 1)
        is_preferred_size = 1 if product_size == pf['preferred_size'] else 0
        
        base_product = self._get_base_product(product)
        has_bought_variant = 1 if pf['past_base_products'].get(base_product, 0) > 0 else 0
        
        # Global features
        global_pop = popularity.get(product, 0)
        adjusted_pop = global_pop * (0.3 if never_purchased and order_idx > 3 else 1.0)
        
        return {
            'customer_id': customer_id,
            'product': product,
            'order_idx': order_idx,
            'order_date': order_date,
            'label': label,
            
            # History features
            'in_history': in_history,
            'history_count': history_count,
            'history_freq': history_freq,
            'orders_since_last_purchase': min(orders_since, 100),
            
            # Improved features
            'time_decay_score': time_decay,
            'never_purchased': never_purchased,
            'category_affinity': category_affinity,
            'is_preferred_category': is_preferred_category,
            'size_affinity': size_affinity,
            'is_preferred_size': is_preferred_size,
            'has_bought_variant': has_bought_variant,
            
            # Time features
            'hour_of_day': time_features['hour_of_day'],
            'day_of_week': time_features['day_of_week'],
            'days_since_last_order': time_features['days_since_last_order'],
            
            # Customer features
            'history_length': customer_features['history_length'],
            'avg_basket_size': customer_features['avg_basket_size'],
            'avg_spend': customer_features['avg_spend'],
            'segment_encoded': customer_features['segment_encoded'],
            'archetype_encoded': customer_features['archetype_encoded'],
            
            # Global features
            'popularity': global_pop,
            'adjusted_popularity': adjusted_pop,
            'category_encoded': cat_to_idx.get(product_category, 0)
        }
    
    def _extract_size(self, product: str) -> str:
        """Extract size from product name."""
        product_lower = product.lower()
        if 'extra large' in product_lower:
            return 'XL'
        elif 'large' in product_lower:
            return 'L'
        elif 'regular' in product_lower:
            return 'R'
        elif 'small' in product_lower:
            return 'S'
        return 'unknown'
    
    def _get_base_product(self, product: str) -> str:
        """Extract base product name without variant."""
        if ' (' in product:
            return product.split(' (')[0]
        return product
    
    def _encode_archetype(self, archetype: str) -> int:
        """Encode archetype to integer."""
        mapping = {
            'parent': 1,
            'coffee_purist': 2,
            'latte_lover': 3,
            'health_conscious': 4,
            'food_focused': 5,
            'casual': 0
        }
        return mapping.get(archetype, 0)
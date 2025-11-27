"""
Customer Feature Extractor
==========================
Extracts customer-level features: archetypes, preferences, behavior patterns.

Refactored from: C_training_data_improved.py
"""

import logging
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CustomerArchetypes:
    """
    Customer archetype definitions.
    
    Archetypes help with:
    1. Cold-start recommendations
    2. Customer segmentation
    3. Personalization strategies
    """
    
    PARENT = "parent"
    COFFEE_PURIST = "coffee_purist"
    LATTE_LOVER = "latte_lover"
    HEALTH_CONSCIOUS = "health_conscious"
    FOOD_FOCUSED = "food_focused"
    CASUAL = "casual"
    
    ALL = [PARENT, COFFEE_PURIST, LATTE_LOVER, HEALTH_CONSCIOUS, FOOD_FOCUSED, CASUAL]


@dataclass
class CustomerProfile:
    """Profile for a single customer."""
    customer_id: int
    archetype: str
    total_orders: int
    avg_basket_size: float
    avg_spend: float
    preferred_category: Optional[str]
    preferred_size: Optional[str]
    segment: str  # VIP, Regular, New


class CustomerFeatureExtractor:
    """
    Extracts customer-level features and archetypes.
    
    Features:
    1. Archetype classification (parent, coffee_purist, etc.)
    2. Purchase behavior (avg basket size, avg spend)
    3. Preferences (category, size, time)
    
    Example:
        >>> extractor = CustomerFeatureExtractor()
        >>> profiles = extractor.extract(prepared_data)
        >>> print(f"Parents: {sum(1 for p in profiles.values() if p.archetype == 'parent')}")
    """
    
    def __init__(self):
        logger.info("CustomerFeatureExtractor initialized")
    
    def extract(self, prepared_data) -> Dict[int, CustomerProfile]:
        """
        Extract profiles for all customers.
        
        Args:
            prepared_data: PreparedData from preprocessor
            
        Returns:
            Dict mapping customer_id -> CustomerProfile
        """
        logger.info("Extracting customer profiles...")
        
        profiles = {}
        archetype_counts = Counter()
        
        for customer_id, orders in prepared_data.customer_histories.items():
            profile = self._build_profile(customer_id, orders)
            profiles[customer_id] = profile
            archetype_counts[profile.archetype] += 1
        
        logger.info(f"Extracted profiles for {len(profiles):,} customers")
        logger.info(f"Archetype distribution: {dict(archetype_counts)}")
        
        return profiles
    
    def _build_profile(self, customer_id: int, orders: List[dict]) -> CustomerProfile:
        """Build profile for single customer."""
        # Collect all products
        all_products = []
        all_categories = []
        all_sizes = []
        total_spend = 0
        
        for order in orders:
            for product in order['basket']:
                if isinstance(product, str):
                    all_products.append(product)
                    all_sizes.append(self._extract_size(product))
            
            if 'categories' in order:
                for cat in order['categories']:
                    if isinstance(cat, str):
                        all_categories.append(cat)
            
            total_spend += order.get('order_total', 0)
        
        # Compute metrics
        total_orders = len(orders)
        avg_basket_size = len(all_products) / total_orders if total_orders > 0 else 0
        avg_spend = total_spend / total_orders if total_orders > 0 else 0
        
        # Most common category and size
        preferred_category = Counter(all_categories).most_common(1)[0][0] if all_categories else None
        preferred_size = Counter(all_sizes).most_common(1)[0][0] if all_sizes else None
        
        # Determine archetype
        archetype = self._classify_archetype(all_products)
        
        # Get segment from first order
        segment = orders[0].get('segment', 'Regular') if orders else 'Regular'
        
        return CustomerProfile(
            customer_id=customer_id,
            archetype=archetype,
            total_orders=total_orders,
            avg_basket_size=avg_basket_size,
            avg_spend=avg_spend,
            preferred_category=preferred_category,
            preferred_size=preferred_size,
            segment=segment
        )
    
    def _classify_archetype(self, products: List[str]) -> str:
        """
        Classify customer archetype based on purchase patterns.
        
        Rules (in priority order):
        1. Parent: Orders babychino or kids items
        2. Coffee Purist: Orders espresso, long black
        3. Latte Lover: Primarily orders lattes
        4. Health Conscious: Orders smoothies, juices
        5. Food Focused: Primarily orders food items
        6. Casual: Default
        """
        if not products:
            return CustomerArchetypes.CASUAL
        
        product_str = ' '.join(products).lower()
        
        # Priority-based classification
        if 'babychino' in product_str or 'kids' in product_str:
            return CustomerArchetypes.PARENT
        
        if 'espresso' in product_str or 'long black' in product_str or 'macchiato' in product_str:
            return CustomerArchetypes.COFFEE_PURIST
        
        if product_str.count('latte') > len(products) * 0.3:  # >30% lattes
            return CustomerArchetypes.LATTE_LOVER
        
        if 'smoothie' in product_str or 'juice' in product_str or 'acai' in product_str:
            return CustomerArchetypes.HEALTH_CONSCIOUS
        
        food_keywords = ['breakfast', 'eggs', 'toast', 'burger', 'sandwich', 'salad', 'roll']
        if any(kw in product_str for kw in food_keywords):
            return CustomerArchetypes.FOOD_FOCUSED
        
        return CustomerArchetypes.CASUAL
    
    def _extract_size(self, product: str) -> str:
        """Extract size from product name."""
        product_lower = product.lower()
        
        if 'extra large' in product_lower:
            return 'XL'
        elif 'large' in product_lower:
            return 'L'
        elif 'regular' in product_lower:
            return 'R'
        elif 'small' in product_lower or 'mini' in product_lower:
            return 'S'
        
        return 'unknown'
    
    def get_archetype_summary(self, profiles: Dict[int, CustomerProfile]) -> Dict[str, int]:
        """Get count of customers per archetype."""
        return Counter(p.archetype for p in profiles.values())
"""
Customer Feature Enginnering

"""
from abc import ABC, abstractmethod
import pandas as pd
import logging 
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional
from src.data_ingestion.data_preprocessor import PreparedData   # <-- pipeline dependency

logger = logging.getLogger(__name__)


class CustomerArchetypes:
    """Customer Segmentation to be use for cold start"""
    PARENT = "parent"
    COFFEE_PURIST = "coffee_purist"
    LATTE_LOVER = "latte_lover"
    HEALTH_CONSCIOUS = "health_conscious"
    FOOD_FOCUSED = "food_focused"
    CASUAL = "casual"
    ALL = [PARENT, COFFEE_PURIST, LATTE_LOVER,
           HEALTH_CONSCIOUS, FOOD_FOCUSED, CASUAL]

@dataclass
class CustomerProfile:
    customer_id: int
    archetype: str
    total_orders: int
    avg_basket_size: float
    avg_spend: float
    preferred_category: Optional[str]
    preferred_size: Optional[str]
    segment: str  # derived from first order
    

class BaseFeatureExtractor(ABC):

    @abstractmethod
    def extract(self,prepared: PreparedData ) -> Dict[int, CustomerProfile]:
        pass

class CustomerFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        logger.info("CustomerFeatureExtractor initialised")

    def extract_size(self, product: str) -> str:
        p = product.lower()
        if "extra large" in p:
            return "XL"
        if "large" in p:
            return "L"
        if "regular" in p:
            return "R"
        if "small" in p or "mini" in p:
            return "S"
        return "unknown"
    
    def classify_archetype(self, products: List[str]) -> str:
        """
        Rule-based labeling.
        """
        if not products:
            return CustomerArchetypes.CASUAL

        s = " ".join(products).lower()

        if "babychino" in s or "kids" in s:
            return CustomerArchetypes.PARENT

        if "espresso" in s or "long black" in s or "macchiato" in s:
            return CustomerArchetypes.COFFEE_PURIST

        if s.count("latte") > len(products) * 0.3:
            return CustomerArchetypes.LATTE_LOVER

        if "smoothie" in s or "juice" in s or "acai" in s:
            return CustomerArchetypes.HEALTH_CONSCIOUS

        food_kw = ["breakfast", "eggs", "toast", "burger", "sandwich", "salad", "roll"]
        if any(k in s for k in food_kw):
            return CustomerArchetypes.FOOD_FOCUSED

        return CustomerArchetypes.CASUAL

    def build_profile(self, customer_id: int, orders: List[dict]) -> CustomerProfile:
        all_products = []
        all_categories = []
        all_sizes = []
        total_spend = 0
        #Item static
        for order in orders:
            for product in order["basket"]:
                if isinstance(product,str):
                    all_products.append(product)
                    all_sizes.append(self.extract_size(product))

            if "categories" in order:
                for c in order["categories"]:
                    if isinstance(c, str):
                        all_categories.append(c)
            total_spend +=order.get("order_total",0)

        #Order Static
        total_orders = len(orders)
        avg_basket_size = len(all_products) / total_orders if total_orders else 0
        avg_spend = total_spend / total_orders if total_orders else 0

        preferred_category = Counter(all_categories).most_common(1)[0][0] if all_categories else None
        preferred_size = Counter(all_sizes).most_common(1)[0][0] if all_sizes else None
        archetype = self.classify_archetype(all_products)

        segment = orders[0].get("segment", "Regular") if orders else "Regular"

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

    def extract(self, prepared: PreparedData) -> Dict[int, CustomerProfile]:
        logger.info("Extracting customer profiles")

        profiles = {}
        archetype_counts = Counter()
        for id,orders in prepared.customer_histories.items():
            profile = self.build_profile(id, orders)
            profiles[id] = profile
            archetype_counts[profile.archetype] += 1
            
        logger.info(f"Extracted {len(profiles):,} customer profiles")
        logger.info(f"Archetype distribution: {dict(archetype_counts)}")
        return profiles


if __name__  == "__main__":
    pass
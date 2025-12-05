
import logging
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List
# from data_ingestion.data_preprocessor import PreparedData

logger = logging.getLogger(__name__)

@dataclass
class TrainingData:
    """
Container for training data
"""
    samples_df: pd.DataFrame
    feature_names: List[str]
    n_positive: int
    n_negative: int
    product_to_idx: Dict[str, int]
    idx_to_product: Dict[int, str]

#Building Training Data
class TrainingDataBuilder:

    FEATURE_NAMES = [
        # History features
        "in_history", "history_count", "history_freq", "orders_since_last_purchase",

        # Improved affinity features
        "time_decay_score", "never_purchased", "category_affinity",
        "is_preferred_category", "size_affinity", "is_preferred_size",
        "has_bought_variant",

        # Time features
        "hour_of_day", "day_of_week", "days_since_last_order",

        # Customer features
        "history_length", "avg_basket_size", "avg_spend",
        "segment_encoded", "archetype_encoded",

        # Global features
        "popularity", "adjusted_popularity", "category_encoded",
    ]

    def __init__(self, negative_ratio: int = 5, random_seed: int = 42):
        self.negative_ratio = negative_ratio
        np.random.seed(random_seed)
        logger.info(f"TrainingDataBuilder init (negative_ratio={negative_ratio})")


    def build(self, prepared_data, product_features, customer_profiles: Dict = None) -> TrainingData:
        logger.info("Building training samples...")

        samples = []
        product_set = set(prepared_data.product_list)
        category_map = prepared_data.category_map or {}
        popularity = product_features.popularity or {}

        # Build category index
        categories = sorted(set(category_map.values()))
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        # Iterate through customers
        total_customers = len(prepared_data.customer_histories)
        for idx, (customer_id, orders) in enumerate(prepared_data.customer_histories.items()):
            if len(orders) < 2:
                continue

            if idx % 200 == 0:
                #Print out status for every 200 customer processed
                logger.info(f"Processing customer {idx}/{total_customers}")

            profile = customer_profiles.get(customer_id) if customer_profiles else None

            archetype_encoded = self.encode_archetype(profile.archetype if profile else "casual")

            #If not segment not found then default regular
            segment = orders[0].get("segment", "Regular")

            
            segment_encoded = {"VIP": 3, "Regular": 2, "New": 1}.get(segment, 2)


            #Use extend to create 1 list (if Append then list of lists)
            samples.extend(
                    self.build_customer_samples(
                        customer_id=customer_id,
                        orders=orders,
                        product_set=product_set,
                        popularity=popularity,
                        category_map=category_map,
                        cat_to_idx=cat_to_idx,
                        segment_encoded=segment_encoded,
                        archetype_encoded=archetype_encoded,
                    )
                )
        # Convert to DF
        df = pd.DataFrame(samples).replace([np.inf, -np.inf], 0).fillna(0)

        n_positive = int(df["label"].sum())
        n_negative = len(df) - n_positive

        logger.info(f"Samples built: {len(df):,}")
        logger.info(f"Positive: {n_positive:,}  Negative: {n_negative:,}")

        # Product index mapping
        product_to_idx = {p: i for i, p in enumerate(prepared_data.product_list)}
        idx_to_product = {i: p for p, i in product_to_idx.items()}

        return TrainingData(
            samples_df=df,
            feature_names=self.FEATURE_NAMES,
            n_positive=n_positive,
            n_negative=n_negative,
            product_to_idx=product_to_idx,
            idx_to_product=idx_to_product,
        )
    
    #Helper function
    def encode_archetype(self, type: str):
        mapping = {
            "parent": 1,
            "coffee_purist": 2,
            "latte_lover": 3,
            "health_conscious": 4,
            "food_focused": 5,
            "casual": 0,
        }
        return mapping.get(type, 0)
    
    def build_customer_samples(
        self, customer_id, orders, product_set, popularity,
        category_map, cat_to_idx, segment_encoded, archetype_encoded
    ) -> List[dict]:

        samples = []

        for order_idx in range(1, len(orders)):
            past_orders = orders[:order_idx]
            current_order = orders[order_idx]

            target_basket = {p for p in current_order["basket"] if isinstance(p, str)}
            if not target_basket:
                continue

            # Past-only features
            past_features = self.compute_past_features(past_orders, category_map)

            # Time features
            time_features = self.compute_time_features(current_order, past_orders)

            # Customer profile features
            customer_features = {
                "history_length": order_idx,
                "avg_basket_size": past_features["avg_basket_size"],
                "avg_spend": past_features["avg_spend"],
                "segment_encoded": segment_encoded,
                "archetype_encoded": archetype_encoded,
            }

            # Helper
            def make_sample(product, label):
                return self.create_product_sample(
                    customer_id, product, label, order_idx,
                    current_order["order_date"], past_features,
                    time_features, customer_features, popularity,
                    category_map, cat_to_idx
                )

            # Positive samples
            for product in target_basket:
                samples.append(make_sample(product, 1))

            # Negative samples
            negative_pool = list(product_set - target_basket)
            n_neg = min(len(target_basket) * self.negative_ratio, len(negative_pool))

            if n_neg > 0:
                weights = np.array([popularity.get(p, 0.0001) for p in negative_pool])
                negatives = np.random.choice(negative_pool, size=n_neg, replace=False, p=weights / weights.sum())
                for product in negatives:
                    samples.append(make_sample(product, 0))

        return samples
    
    def create_product_sample(
        self, customer_id, product, label, order_idx, order_date,
        past_features, time_features, customer_features,
        popularity, category_map, cat_to_idx
    ):
        pf = past_features

        in_history = int(product in pf["past_products"])
        history_count = pf["past_product_counts"].get(product, 0)
        history_freq = history_count / max(pf["n_orders"], 1)

        last_idx = pf["past_product_last_idx"].get(product, -1)
        orders_since = order_idx - last_idx - 1 if last_idx >= 0 else 999
        time_decay = np.exp(-0.1 * orders_since) if in_history else 0

        product_category = category_map.get(product, "Unknown")
        category_affinity = pf["past_categories"].get(product_category, 0) / max(pf["total_items"], 1)
        is_preferred_category = int(product_category == pf["preferred_category"])

        product_size = self.extract_size(product)
        size_affinity = pf["past_sizes"].get(product_size, 0) / max(pf["total_items"], 1)
        is_preferred_size = int(product_size == pf["preferred_size"])

        base_product = self.get_base_product(product)
        has_bought_variant = int(pf["past_base_products"].get(base_product, 0) > 0)

        global_pop = popularity.get(product, 0)
        adjusted_pop = global_pop * (0.3 if orders_since > 3 and not in_history else 1)

        return {
            "customer_id": customer_id,
            "product": product,
            "order_date": order_date,
            "order_idx": order_idx,
            "label": label,

            # History features
            "in_history": in_history,
            "history_count": history_count,
            "history_freq": history_freq,
            "orders_since_last_purchase": min(orders_since, 100),

            # Improved features
            "time_decay_score": time_decay,
            "never_purchased": int(not in_history),
            "category_affinity": category_affinity,
            "is_preferred_category": is_preferred_category,
            "size_affinity": size_affinity,
            "is_preferred_size": is_preferred_size,
            "has_bought_variant": has_bought_variant,

            # Time features
            "hour_of_day": time_features["hour_of_day"],
            "day_of_week": time_features["day_of_week"],
            "days_since_last_order": time_features["days_since_last_order"],

            # Customer features
            **customer_features,

            # Global features
            "popularity": global_pop,
            "adjusted_popularity": adjusted_pop,
            "category_encoded": cat_to_idx.get(product_category, 0),
        }


    def extract_size(self, product):
        p = product.lower()
        if "extra large" in p: return "XL"
        if "large" in p: return "L"
        if "regular" in p: return "R"
        if "small" in p or "mini" in p: return "S"
        return "unknown"
    
    def get_base_product(self, product):
        return product.split(" (")[0] if " (" in product else product
    
    def compute_time_features(self, current_order, past_orders):
        order_time = current_order.get("order_time", "12:00:00")
        hour = int(order_time.split(":")[0]) if isinstance(order_time, str) else 12

        order_date = current_order["order_date"]
        day_of_week = getattr(order_date, "dayofweek", 0)

        if past_orders:
            last_date = past_orders[-1]["order_date"]
            days_since = max(0, min((order_date - last_date).days, 365))
        else:
            days_since = 0

        return {
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "days_since_last_order": days_since,
        }
    
    def compute_past_features(self, past_orders, category_map):
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
            for p in order["basket"]:
                if not isinstance(p, str):
                    continue
                past_products.add(p)
                past_product_counts[p] += 1
                past_product_last_idx[p] = i

                cat = category_map.get(p, "Unknown")
                past_categories[cat] += 1

                size = self.extract_size(p)
                past_sizes[size] += 1

                base = self.get_base_product(p)
                past_base_products[base] += 1

                total_items += 1

            total_spend += order.get("order_total", 0)
            basket_sizes.append(order.get("basket_size", len(order["basket"])))

        return {
            "past_products": past_products,
            "past_product_counts": past_product_counts,
            "past_product_last_idx": past_product_last_idx,
            "past_categories": past_categories,
            "past_sizes": past_sizes,
            "past_base_products": past_base_products,
            "total_items": total_items,
            "n_orders": len(past_orders),
            "avg_basket_size": np.mean(basket_sizes) if basket_sizes else 0,
            "avg_spend": total_spend / len(past_orders) if past_orders else 0,
            "preferred_category": past_categories.most_common(1)[0][0] if past_categories else "Unknown",
            "preferred_size": past_sizes.most_common(1)[0][0] if past_sizes else "unknown",
        }
    
if __name__ == '__main__':
    pass
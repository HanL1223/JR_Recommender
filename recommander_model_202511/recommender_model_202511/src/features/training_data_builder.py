import logging
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# TrainingData Container
# ----------------------------------------------------------------------
@dataclass
class TrainingData:
    samples_df: pd.DataFrame
    feature_names: List[str]
    n_positive: int
    n_negative: int
    product_to_idx: Dict[str, int]
    idx_to_product: Dict[int, str]
    encoders: Dict[str, Dict[str, int]]  # segment, archetype, category


# ----------------------------------------------------------------------
# Main TrainingDataBuilder
# ----------------------------------------------------------------------
class TrainingDataBuilder:

    FEATURE_NAMES = [
        # History features
        "in_history", "history_count", "history_freq", "orders_since_last_purchase",

        # Affinity and variant features
        "time_decay_score", "never_purchased",
        "category_affinity", "is_preferred_category",
        "size_affinity", "is_preferred_size",
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
        logger.info(f"TrainingDataBuilder initialised (negative_ratio={negative_ratio})")

    # ------------------------------------------------------------------
    # MASTER ENTRYPOINT
    # ------------------------------------------------------------------
    def build(self, prepared_data, product_features, customer_profiles):

        logger.info("Building training samples...")

        samples = []
        product_set = set(prepared_data.product_list)
        category_map = prepared_data.category_map or {}
        popularity = product_features.popularity or {}

        # --------------------------
        # BUILD ENCODERS (returned for inference)
        # --------------------------
        categories = sorted(set(category_map.values()))
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        segment_to_idx = {"New": 1, "Regular": 2, "VIP": 3}
        archetype_to_idx = {
            "casual": 0, "parent": 1, "coffee_purist": 2,
            "latte_lover": 3, "health_conscious": 4, "food_focused": 5
        }

        encoders = {
            "category": cat_to_idx,
            "segment": segment_to_idx,
            "archetype": archetype_to_idx,
        }

        # --------------------------
        # ITERATE THROUGH CUSTOMERS
        # --------------------------
        for idx, (customer_id, orders) in enumerate(prepared_data.customer_histories.items()):
            if len(orders) < 2:
                continue

            if idx % 200 == 0:
                logger.info(f"Processing customer {idx}/{len(prepared_data.customer_histories)}")

            profile = customer_profiles.get(customer_id)
            archetype = profile.archetype if profile else "casual"
            archetype_encoded = archetype_to_idx.get(archetype, 0)

            segment = orders[0].get("segment", "Regular")
            segment_encoded = segment_to_idx.get(segment, 2)

            samples.extend(
                self.build_customer_samples(
                    customer_id, orders, product_set,
                    popularity, category_map, cat_to_idx,
                    segment_encoded, archetype_encoded
                )
            )

        # --------------------------
        # Convert samples to DataFrame
        # --------------------------
        df = pd.DataFrame(samples).replace([np.inf, -np.inf], 0).fillna(0)
        # --- Ensure order_date exists AND is datetime ---
        if "order_date" not in df.columns:
            raise ValueError("TrainingDataBuilder error: order_date missing from samples_df")

        df["order_date"] = pd.to_datetime(df["order_date"])
        n_positive = int(df["label"].sum())
        n_negative = len(df) - n_positive

        logger.info(f"Training samples built: {len(df):,}")
        logger.info(f"Positive={n_positive:,}  Negative={n_negative:,}")

        # Product index tables
        product_to_idx = {p: i for i, p in enumerate(prepared_data.product_list)}
        idx_to_product = {i: p for p, i in product_to_idx.items()}

        return TrainingData(
            samples_df=df,
            feature_names=self.FEATURE_NAMES,
            n_positive=n_positive,
            n_negative=n_negative,
            product_to_idx=product_to_idx,
            idx_to_product=idx_to_product,
            encoders=encoders,   # <-- IMPORTANT
        )

    # ------------------------------------------------------------------
    # Build samples for a single customer's history
    # ------------------------------------------------------------------
    def build_customer_samples(
        self, customer_id, orders, product_set, popularity,
        category_map, cat_to_idx, segment_encoded, archetype_encoded
    ):
        samples = []

        for order_idx in range(1, len(orders)):
            past_orders = orders[:order_idx]
            current_order = orders[order_idx]

            target_basket = {p for p in current_order["basket"] if isinstance(p, str)}
            if not target_basket:
                continue

            past_feats = self.compute_past_features(past_orders, category_map)
            time_feats = self.compute_time_features(current_order, past_orders)

            customer_feats = {
                "history_length": order_idx,
                "avg_basket_size": past_feats["avg_basket_size"],
                "avg_spend": past_feats["avg_spend"],
                "segment_encoded": segment_encoded,
                "archetype_encoded": archetype_encoded,
            }

            def make(product, label):
                return self.create_product_sample(
                    customer_id, product, label, order_idx,
                    current_order["order_date"], past_feats,
                    time_feats, customer_feats, popularity,
                    category_map, cat_to_idx
                )

            # Positive samples
            for p in target_basket:
                samples.append(make(p, 1))

            # Negative samples
            negative_pool = list(product_set - target_basket)
            n_neg = min(len(target_basket) * self.negative_ratio, len(negative_pool))

            if n_neg > 0:
                weights = np.array([popularity.get(p, 0.0001) for p in negative_pool])
                negative_choices = np.random.choice(
                    negative_pool, size=n_neg, replace=False, p=weights / weights.sum()
                )
                for p in negative_choices:
                    samples.append(make(p, 0))

        return samples

    # ------------------------------------------------------------------
    # Create training sample row
    # ------------------------------------------------------------------
    def create_product_sample(
        self, customer_id, product, label, order_idx, order_date,
        past_feats, time_feats, customer_feats, popularity, category_map, cat_to_idx
    ):
        pf = past_feats

        in_hist = int(product in pf["past_products"])
        hist_count = pf["past_product_counts"].get(product, 0)
        hist_freq = hist_count / max(pf["n_orders"], 1)

        last_idx = pf["past_product_last_idx"].get(product, -1)
        orders_since = (order_idx - last_idx - 1) if last_idx >= 0 else 999
        time_decay = np.exp(-0.1 * orders_since) if in_hist else 0

        category = category_map.get(product, "Unknown")
        cat_aff = pf["past_categories"].get(category, 0) / max(pf["total_items"], 1)
        is_pref_cat = int(category == pf["preferred_category"])

        size = self.extract_size(product)
        size_aff = pf["past_sizes"].get(size, 0) / max(pf["total_items"], 1)
        is_pref_size = int(size == pf["preferred_size"])

        base = self.get_base_product(product)
        has_variant = int(pf["past_base_products"].get(base, 0) > 0)

        pop = popularity.get(product, 0)
        adj_pop = pop * (0.3 if orders_since > 3 and not in_hist else 1)

        return {
            "customer_id": customer_id,
            "product": product,
            "order_date": order_date,     # <-- REQUIRED FOR TEMPORAL SPLIT
            "label": label,

            # History
            "in_history": in_hist,
            "history_count": hist_count,
            "history_freq": hist_freq,
            "orders_since_last_purchase": min(orders_since, 100),

            # Affinity/variant
            "time_decay_score": time_decay,
            "never_purchased": int(not in_hist),
            "category_affinity": cat_aff,
            "is_preferred_category": is_pref_cat,
            "size_affinity": size_aff,
            "is_preferred_size": is_pref_size,
            "has_bought_variant": has_variant,

            # Time features
            **time_feats,

            # Customer features
            **customer_feats,

            # Global product stats
            "popularity": pop,
            "adjusted_popularity": adj_pop,
            "category_encoded": cat_to_idx.get(category, 0),
        }

    # ------------------------------------------------------------------
    # Utility functions (kept consistent with inference builder)
    # ------------------------------------------------------------------
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
        hour = int(current_order.get("order_time", "12:00:00").split(":")[0])

        day_of_week = current_order["order_date"].dayofweek

        if past_orders:
            last_date = past_orders[-1]["order_date"]
            days_since = min((current_order["order_date"] - last_date).days, 365)
        else:
            days_since = 0

        return {
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "days_since_last_order": max(days_since, 0)
        }

    def compute_past_features(self, past_orders, category_map):
        past_products = set()
        product_counts = Counter()
        last_idx = {}
        past_categories = Counter()
        past_sizes = Counter()
        past_base = Counter()

        total_items = 0
        total_spend = 0
        basket_sizes = []

        for i, order in enumerate(past_orders):
            for p in order["basket"]:
                if not isinstance(p, str):
                    continue

                past_products.add(p)
                product_counts[p] += 1
                last_idx[p] = i

                cat = category_map.get(p, "Unknown")
                past_categories[cat] += 1

                size = self.extract_size(p)
                past_sizes[size] += 1

                base = self.get_base_product(p)
                past_base[base] += 1

                total_items += 1

            total_spend += order.get("order_total", 0)
            basket_sizes.append(order.get("basket_size", len(order["basket"])))

        return {
            "past_products": past_products,
            "past_product_counts": product_counts,
            "past_product_last_idx": last_idx,
            "past_categories": past_categories,
            "past_sizes": past_sizes,
            "past_base_products": past_base,
            "total_items": total_items,
            "n_orders": len(past_orders),
            "avg_basket_size": np.mean(basket_sizes) if basket_sizes else 0,
            "avg_spend": total_spend / len(past_orders) if past_orders else 0,
            "preferred_category": past_categories.most_common(1)[0][0] if past_categories else "Unknown",
            "preferred_size": past_sizes.most_common(1)[0][0] if past_sizes else "unknown",
        }

"""
RecommenderPredictor
====================

Unified, production-ready recommendation predictor.

Key guarantees:
- Inference uses FeatureMatrixBuilder (1:1 match with TrainingDataBuilder).
- No ad-hoc feature engineering at prediction time.
- Ensures all required ML feature columns exist.
- Supports ML rankers and baseline models.
- Supports cold-start and add-on recommendations.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd

from src.features.feature_matrix_builder import FeatureMatrixBuilder

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# OUTPUT DATA STRUCTURES
# ----------------------------------------------------------------------
@dataclass
class RecommendationItem:
    product: str
    score: float
    reason: str


@dataclass
class Prediction:
    customer_id: Optional[int]
    model_used: str
    primary_items: List[RecommendationItem]
    addon_items: List[RecommendationItem]


# ----------------------------------------------------------------------
# MAIN PREDICTOR
# ----------------------------------------------------------------------
class RecommenderPredictor:
    """
    Final unified predictor used in production and demo.

    Steps:
    1. Build feature matrix using FeatureMatrixBuilder.
    2. Score using ML or baseline model.
    3. Select top-K items.
    4. Generate addon recommendations via co-occurrence.
    5. Provide cold-start fallback when needed.
    """

    def __init__(
        self,
        ml_model,
        baseline_model,
        product_features,
        prepared_data,
        customer_profiles,
        feature_names,
        cold_start_handler,
        encoders: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.ml_model = ml_model
        self.baseline_model = baseline_model
        self.product_features = product_features
        self.prepared_data = prepared_data
        self.customer_profiles = customer_profiles
        self.feature_names = list(feature_names) if feature_names else []
        self.cold_start_handler = cold_start_handler

        # Build encoders if not provided
        self.encoders = encoders or self._build_encoders_from_prepared(prepared_data)

        # FeatureMatrixBuilder for ML-aligned inference
        self.feature_builder = FeatureMatrixBuilder(
            prepared_data=prepared_data,
            product_features=product_features,
            customer_profiles=customer_profiles,
            category_map=getattr(prepared_data, "category_map", {}),
            encoders=self.encoders,
        )

        self.use_ml = ml_model is not None

        logger.info("RecommenderPredictor initialized with ML=%s", bool(self.use_ml))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def recommend(self, customer_id: int, top_k: int = 5) -> Prediction:
        """
        Return top-K recommendations for a customer. Automatically handles:
        - ML prediction
        - Baseline fallback
        - Cold-start fallback
        - Add-on suggestions
        """

        # Cold-start detection
        if customer_id not in self.prepared_data.customer_histories:
            logger.info("Customer %s not found; using cold-start fallback.", customer_id)
            return self._cold_start(top_k)

        # Build feature matrix from aligned inference builder
        feature_df = self._build_feature_matrix(customer_id)

        # Score rows
        if self.use_ml:
            scores = self.ml_model.predict_df(feature_df)
            model_used = getattr(self.ml_model, "name", "MLModel")
        else:
            scores = self.baseline_model.predict_df(feature_df)
            model_used = getattr(self.baseline_model, "name", "BaselineModel")

        # Apply scores
        feature_df = feature_df.copy()
        feature_df["score"] = scores

        # Select top-K recommendations
        top_df = feature_df.sort_values("score", ascending=False).head(top_k)

        primary_items = [
            RecommendationItem(
                product=row["product"],
                score=float(row["score"]),
                reason="ML ranked" if self.use_ml else "Baseline popularity",
            )
            for _, row in top_df.iterrows()
        ]

        addon_items = self._get_addon_recommendations(primary_items)

        return Prediction(
            customer_id=customer_id,
            model_used=model_used,
            primary_items=primary_items,
            addon_items=addon_items,
        )

    # ------------------------------------------------------------------
    # FEATURE MATRIX BUILDER
    # ------------------------------------------------------------------
    def _build_feature_matrix(self, customer_id: int) -> pd.DataFrame:
        """
        Ensure inference feature matrix matches EXACT training feature set.
        """
        df = self.feature_builder.build(customer_id)

        # Enforce training-time feature ordering
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0

        # Reorder columns to match model's training format (important)
        df = df[self.feature_names + ["product", "customer_id"]]

        return df

    # ------------------------------------------------------------------
    # ENCODERS
    # ------------------------------------------------------------------
    def _build_encoders_from_prepared(self, prepared_data) -> Dict[str, Dict[str, int]]:
        category_map = prepared_data.category_map or {}
        categories = sorted(set(category_map.values()))
        return {
            "category": {cat: i for i, cat in enumerate(categories)},
            "segment": {"New": 1, "Regular": 2, "VIP": 3},
            "archetype": {
                "casual": 0,
                "parent": 1,
                "coffee_purist": 2,
                "latte_lover": 3,
                "health_conscious": 4,
                "food_focused": 5,
            },
        }

    # ------------------------------------------------------------------
    # ADD-ON RECOMMENDATIONS
    # ------------------------------------------------------------------
    def _get_addon_recommendations(self, primary_items, top_k: int = 2):
        co = self.product_features.cooccurrence
        scores = {}

        primary_set = {p.product for p in primary_items}

        for item in primary_items:
            if item.product not in co:
                continue
            for other, lift in co[item.product].items():
                if other not in primary_set:
                    scores[other] = max(scores.get(other, 0.0), float(lift))

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        return [
            RecommendationItem(
                product=prod,
                score=lift,
                reason="Frequently purchased together",
            )
            for prod, lift in sorted_items
        ]

    # ------------------------------------------------------------------
    # COLD START
    # ------------------------------------------------------------------
    def _cold_start(self, top_k: int) -> Prediction:
        items = self.cold_start_handler.recommend(top_k=top_k)
        primary_items = [
            RecommendationItem(product=i.product, score=i.score, reason=i.reason)
            for i in items
        ]
        return Prediction(
            customer_id=None,
            model_used="ColdStart",
            primary_items=primary_items,
            addon_items=[],
        )

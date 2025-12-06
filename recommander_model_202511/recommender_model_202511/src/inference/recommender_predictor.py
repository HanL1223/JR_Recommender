"""
RecommenderPredictor
====================

Production-ready unified recommendation predictor.

Key principles:
- Uses ONLY features present during model training.
- No generation of new behavioural or temporal features at inference.
- Product features taken from ProductFeatures.to_row(product)
- Customer features taken from CustomerProfile.to_ml_dict()
- Missing features (vs training feature list) are automatically added as zeros.
- Supports ML model or fallback baseline model.
- Supports addon recommendations using product co-occurrence.
- Supports cold-start fallback.
"""

import logging
from dataclasses import dataclass
from typing import List
import pandas as pd

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
    customer_id: int | None
    model_used: str
    primary_items: List[RecommendationItem]
    addon_items: List[RecommendationItem]


# ----------------------------------------------------------------------
# MAIN PREDICTOR
# ----------------------------------------------------------------------
class RecommenderPredictor:
    """
    Final unified predictor.

    Responsibilities:
    - Construct ML feature matrix
    - Score with ML or baseline model
    - Add-on recommendations using product co-occurrence
    - Cold-start fallback
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
    ):
        self.ml_model = ml_model
        self.baseline_model = baseline_model
        self.product_features = product_features
        self.prepared_data = prepared_data
        self.customer_profiles = customer_profiles
        self.feature_names = list(feature_names) if feature_names else []
        self.cold_start_handler = cold_start_handler

        self.use_ml = ml_model is not None

        logger.info("RecommenderPredictor initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def recommend(self, customer_id: int, top_k: int = 5) -> Prediction:
        """Generate top-K recommendations."""

        # -------------------------------
        # Cold-start detection
        # -------------------------------
        if customer_id not in self.prepared_data.customer_histories:
            logger.info(
                "Customer %s not found in histories. Using cold-start strategy.",
                customer_id,
            )
            return self._cold_start(top_k)

        # -------------------------------
        # Build feature matrix
        # -------------------------------
        feature_df = self._build_feature_matrix(customer_id)

        # -------------------------------
        # Score with appropriate model
        # -------------------------------
        if self.use_ml:
            scores = self.ml_model.predict_df(feature_df)
            model_used = getattr(self.ml_model, "name", "MLModel")
        else:
            scores = self.baseline_model.predict_df(feature_df)
            model_used = getattr(self.baseline_model, "name", "BaselineModel")

        feature_df = feature_df.copy()
        feature_df["score"] = scores

        feature_df = feature_df.sort_values("score", ascending=False).head(top_k)

        # -------------------------------
        # Format outputs
        # -------------------------------
        primary_items = [
            RecommendationItem(
                product=row["product"],
                score=float(row["score"]),
                reason="ML ranked" if self.use_ml else "Baseline popularity",
            )
            for _, row in feature_df.iterrows()
        ]

        addon_items = self._get_addon_recommendations(primary_items)

        return Prediction(
            customer_id=customer_id,
            model_used=model_used,
            primary_items=primary_items,
            addon_items=addon_items,
        )

    # ------------------------------------------------------------------
    # FEATURE MATRIX
    # ------------------------------------------------------------------
    def _build_feature_matrix(self, customer_id: int) -> pd.DataFrame:
        """
        Build ML-ready feature matrix.

        VERY IMPORTANT:
        - Only uses features available during training.
        - Does NOT generate new temporal/behavioural inference features.
        """
        rows: list[dict] = []

        # -------------------------------
        # Customer-level features
        # -------------------------------
        cust_profile = self.customer_profiles.get(customer_id)

        if cust_profile is not None and hasattr(cust_profile, "to_ml_dict"):
            cust_features = cust_profile.to_ml_dict()
        elif cust_profile is not None:
            cust_features = cust_profile.to_dict()
        else:
            cust_features = {}

        # -------------------------------
        # Build a row per product
        # -------------------------------
        for product in self.product_features.popularity.keys():
            prod_features = self.product_features.to_row(product)

            row = {
                "customer_id": customer_id,
                "product": product,
            }

            row.update(cust_features)
            row.update(prod_features)

            rows.append(row)

        df = pd.DataFrame(rows)

        # -------------------------------
        # Enforce training feature space
        # Missing features must be added as 0
        # -------------------------------
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0

        # Keep column order consistent (model.predict_df will select correctly)
        return df

    # ------------------------------------------------------------------
    # ADD-ON RECOMMENDATIONS
    # ------------------------------------------------------------------
    def _get_addon_recommendations(
        self, primary_items: List[RecommendationItem], top_k: int = 2
    ) -> List[RecommendationItem]:
        co = self.product_features.cooccurrence
        scores: dict[str, float] = {}

        primary_set = {item.product for item in primary_items}

        for item in primary_items:
            p = item.product
            if p not in co:
                continue

            for other, lift in co[p].items():
                if other in primary_set:
                    continue

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
        cold_items = self.cold_start_handler.recommend(top_k=top_k)

        primary_items = [
            RecommendationItem(
                product=i.product,
                score=float(i.score),
                reason=i.reason,
            )
            for i in cold_items
        ]

        return Prediction(
            customer_id=None,
            model_used="ColdStart",
            primary_items=primary_items,
            addon_items=[],
        )

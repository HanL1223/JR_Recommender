"""
RecommenderPredictor
====================
Generates recommendations using ML model or baseline,
and falls back to cold-start handler when required.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecommendationItem:
    product: str
    score: float
    reason: str


@dataclass
class Prediction:
    customer_id: int
    model_used: str
    primary_items: List[RecommendationItem]
    addon_items: List[RecommendationItem]


class RecommenderPredictor:
    """
    Final unified predictor.

    Responsibilities:
    - Build ML feature matrix for a customer
    - Score using LightGBM or baseline model
    - Add-on (co-occurrence) recommendations
    """

    def __init__(
        self,
        ml_model,
        baseline_model,
        product_features,
        prepared_data,
        customer_profiles,
        feature_names,
        cold_start_handler
    ):
        self.ml_model = ml_model
        self.baseline_model = baseline_model
        self.product_features = product_features
        self.prepared_data = prepared_data
        self.customer_profiles = customer_profiles
        self.feature_names = feature_names
        self.cold_start_handler = cold_start_handler

        self.use_ml = ml_model is not None

        logger.info("RecommenderPredictor initialized")

    # ----------------------------------------------------------------------
    # MAIN METHOD
    # ----------------------------------------------------------------------
    def recommend(self, customer_id: int, top_k: int = 5):

        # Cold-start if customer does not appear in dataset
        if customer_id not in self.prepared_data.customer_histories:
            return self._cold_start(top_k)

        feature_df = self._build_feature_matrix(customer_id)

        # ML MODEL
        if self.use_ml:
            scores = self.ml_model.predict_df(feature_df)
            model_used = self.ml_model.name
        else:
            model_used = self.baseline_model.name
            scores = self.baseline_model.predict_df(feature_df)

        feature_df["score"] = scores
        feature_df = feature_df.sort_values("score", ascending=False).head(top_k)

        primary_items = [
            RecommendationItem(
                product=row["product"],
                score=row["score"],
                reason="ML ranked" if self.use_ml else "Baseline popularity"
            )
            for _, row in feature_df.iterrows()
        ]

        addon_items = self._get_addon_recommendations(primary_items)

        return Prediction(
            customer_id=customer_id,
            model_used=model_used,
            primary_items=primary_items,
            addon_items=addon_items
        )

    # ----------------------------------------------------------------------
    # FEATURE MATRIX BUILDER FOR ML MODEL
    # ----------------------------------------------------------------------
    def _build_feature_matrix(self, customer_id: int) -> pd.DataFrame:
        rows = []

        cust_profile = self.customer_profiles.get(customer_id)
        cust_dict = cust_profile.to_dict() if cust_profile else {}

        for product in self.product_features.popularity.keys():
            prod_dict = self.product_features.to_row(product)

            row = {"customer_id": customer_id, "product": product}
            row.update(cust_dict)
            row.update(prod_dict)

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("product")

        return df

    # ----------------------------------------------------------------------
    # ADD-ON PRODUCTS BASED ON COOCCURRENCE
    # ----------------------------------------------------------------------
    def _get_addon_recommendations(
        self, primary_items: List[RecommendationItem], top_k: int = 2
    ) -> List[RecommendationItem]:

        co = self.product_features.cooccurrence
        scores = {}

        primary_set = {i.product for i in primary_items}

        for item in primary_items:
            product = item.product

            if product not in co:
                continue

            for other, lift in co[product].items():
                if other not in primary_set:
                    scores[other] = max(scores.get(other, 0), lift)

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        return [
            RecommendationItem(
                product=p,
                score=v,
                reason="Frequently purchased together"
            )
            for p, v in sorted_items
        ]

    # ----------------------------------------------------------------------
    # COLD START HANDLER
    # ----------------------------------------------------------------------
    def _cold_start(self, top_k: int):
        cold_items = self.cold_start_handler.recommend(top_k=top_k)

        primary = [RecommendationItem(
            product=i.product,
            score=i.score,
            reason=i.reason
        ) for i in cold_items]

        return Prediction(
            customer_id=None,
            model_used="ColdStart",
            primary_items=primary,
            addon_items=[]
        )

"""
Baseline Models
===============
Simple baseline recommenders for comparison.

- PopularityRecommender: global product popularity
- PersonalFrequencyRecommender: blends personal frequency with global popularity
"""

import logging
from typing import List,Dict,Any,Self

import numpy as np
import pandas as pd

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)



class PopularityRecommender(BaseRecommender):
    """
    Docstring for PopularityRecommender

    """
    @property
    def name(self) -> str:
        return "Popularity"
    def __init__(self):
        self.popularity_scores:Dict[str,float] = {}
        self._is_fitted:bool = False

    def fit(self,train_df,feature_names:List[str],**kwarg):
        logger.info(f"Fitting {self.name} model ")
        #Error Handler for missing crtical trainning column
        if "label" not in train_df.columns or "product" not in train_df.columns:
            raise ValueError ("train_df must contain label and product column")
        positive_df = train_df[train_df["label"] == 1]

        if positive_df.empty:
            logger.warning('No positive samples in training data.')
            self.popularity_scores = {}
        else:
            product_counts = positive_df['product'].value_counts()
            total = float(product_counts.sum())
            self.popularity_scores = (product_counts/total).to_dict()
        self._is_fitted = True
        logger.info(f"Learned popularity for {len(self.popularity_scores)} product")

        return self
    
    def predict (
            self,
            customer_id:int,
            products: List[str],
            features = pd.DataFrame
    ) ->np.ndarray:
        """
        Score product by global popularity
        i.e. the more a item is ordered the higher chance it will be recommended
        predicting a single column/vale
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted, call fit() prior to predict()")
        
        scores = np.array([
            self.popularity_scores.get(p,0.0) for p in products
        ],dtype=float)
        return scores
    
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch prediction on a DataFrame with a 'product' column.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() before predict_df().")

        if "product" not in df.columns:
            raise ValueError("df must contain a 'product' column for predict_df().")

        return df["product"].map(lambda p: self.popularity_scores.get(p, 0.0)).to_numpy(dtype=float)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "popularity",
            "n_products": len(self.popularity_scores),
        }
class PersonalFrequencyRecommender(BaseRecommender):
    """
    Personal frequency-based recommender.

    Combines:
      - how often THIS customer bought a product (personal frequency)
      - how often the product is bought globally (global popularity)

    score = (1 - smoothing) * personal_freq + smoothing * global_freq

    This is baseline for repeat-purchase scenarios.
    """

    @property
    def name(self) -> str:
        return "PersonalFrequency"

    def __init__(self, smoothing: float = 0.3) -> None:
        """
        Args:
            smoothing: Weight for global popularity (0–1).
                       0.0 = only personal frequency
                       1.0 = only global popularity
        """
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError("smoothing must be between 0.0 and 1.0")

        self.smoothing: float = smoothing
        self.customer_frequencies: Dict[int, Dict[str, float]] = {}
        self.global_popularity: Dict[str, float] = {}
        self._is_fitted: bool = False

        logger.info(f"PersonalFrequencyRecommender initialised (smoothing={smoothing})")

    # -------------------- fit --------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> Self:
        """
        Learn:
          - global popularity of products
          - per-customer product frequencies (from positive samples only)
        """
        logger.info(f"Fitting {self.name} model...")

        if "label" not in train_df.columns or "product" not in train_df.columns or "customer_id" not in train_df.columns:
            raise ValueError("train_df must contain 'label', 'product', and 'customer_id' columns.")

        positive_df = train_df[train_df["label"] == 1]

        if positive_df.empty:
            logger.warning("No positive samples in training data. Frequencies will be empty.")
            self.global_popularity = {}
            self.customer_frequencies = {}
            self._is_fitted = True
            return self

        # Global popularity P(product)
        product_counts = positive_df["product"].value_counts()
        total = float(product_counts.sum())
        self.global_popularity = (product_counts / total).to_dict()

        # Personal frequencies P(product | customer)
        self.customer_frequencies = {}
        for customer_id, group in positive_df.groupby("customer_id"):
            counts = group["product"].value_counts()
            c_total = float(counts.sum())
            self.customer_frequencies[customer_id] = (counts / c_total).to_dict()

        self._is_fitted = True
        logger.info(f"Learned frequencies for {len(self.customer_frequencies)} customers")
        logger.info(f"Global popularity for {len(self.global_popularity)} products")

        return self

    # -------------------- predict --------------------
    def predict(
        self,
        customer_id: int,
        products: List[str],
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        Blend personal frequency and global popularity for each product.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() before predict().")

        personal_freq = self.customer_frequencies.get(customer_id, {})

        scores: List[float] = []
        for p in products:
            p_personal = personal_freq.get(p, 0.0)
            p_global = self.global_popularity.get(p, 0.0)
            score = (1.0 - self.smoothing) * p_personal + self.smoothing * p_global
            scores.append(score)

        return np.array(scores, dtype=float)

    # -------------------- batch prediction helper --------------------
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch prediction on a DataFrame with 'customer_id' and 'product' columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() before predict_df().")

        if "customer_id" not in df.columns or "product" not in df.columns:
            raise ValueError("df must contain 'customer_id' and 'product' columns for predict_df().")

        scores: List[float] = []
        for _, row in df.iterrows():
            cid = row["customer_id"]
            prod = row["product"]

            personal_freq = self.customer_frequencies.get(cid, {})
            p_personal = personal_freq.get(prod, 0.0)
            p_global = self.global_popularity.get(prod, 0.0)
            score = (1.0 - self.smoothing) * p_personal + self.smoothing * p_global
            scores.append(score)

        return np.array(scores, dtype=float)

    # -------------------- params for logging --------------------
    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "personal_frequency",
            "smoothing": self.smoothing,
            "n_customers": len(self.customer_frequencies),
            "n_products": len(self.global_popularity),
        }
    
if __name__ == "__main__":
    pass
"""
Fit & evaluate baselines
pop_model = PopularityRecommender().fit(train_df, feature_names)
pf_model = PersonalFrequencyRecommender(smoothing=0.3).fit(train_df, feature_names)

Example scoring on test_df
test_scores_pop = pop_model.predict_df(test_df)
test_scores_pf = pf_model.predict_df(test_df)
"""
"""
Step E: Baseline Models
=======================
Train simple baseline models for comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> 'BaseRecommender':
        pass
    
    @abstractmethod
    def predict(self, customer_id: int, top_k: int) -> List[Tuple[str, float]]:
        pass


class PopularityRecommender(BaseRecommender):
    """Recommends globally popular products."""
    
    def __init__(self):
        self.popularity = {}
        self.top_products = []
    
    def fit(self, train_df: pd.DataFrame) -> 'PopularityRecommender':
        # Get positive samples only
        pos_df = train_df[train_df['label'] == 1]
        counts = pos_df['product'].value_counts()
        total = len(pos_df)
        self.popularity = (counts / total).to_dict()
        self.top_products = list(self.popularity.keys())
        logger.info(f"PopularityRecommender fitted on {len(self.popularity)} products")
        return self
    
    def predict(self, customer_id: int = None, top_k: int = 5) -> List[Tuple[str, float]]:
        return [(p, self.popularity[p]) for p in self.top_products[:top_k]]


class PersonalFrequencyRecommender(BaseRecommender):
    """Recommends based on customer's past purchases with popularity fallback."""
    
    def __init__(self, smoothing: float = 0.3):
        self.smoothing = smoothing
        self.customer_prefs = {}
        self.global_prefs = {}
        self.all_products = []
    
    def fit(self, train_df: pd.DataFrame) -> 'PersonalFrequencyRecommender':
        pos_df = train_df[train_df['label'] == 1]
        
        # Global preferences
        global_counts = pos_df['product'].value_counts()
        total = len(pos_df)
        self.global_prefs = (global_counts / total).to_dict()
        self.all_products = list(self.global_prefs.keys())
        
        # Per-customer preferences (from PAST data in training set)
        for customer_id, group in pos_df.groupby('customer_id'):
            counts = group['product'].value_counts()
            cust_total = len(group)
            self.customer_prefs[customer_id] = (counts / cust_total).to_dict()
        
        logger.info(f"PersonalFrequencyRecommender fitted on {len(self.customer_prefs)} customers")
        return self
    
    def predict(self, customer_id: int, top_k: int = 5) -> List[Tuple[str, float]]:
        cust_prefs = self.customer_prefs.get(customer_id, {})
        
        scores = {}
        for product in self.all_products:
            personal = cust_prefs.get(product, 0)
            global_score = self.global_prefs.get(product, 0)
            scores[product] = (1 - self.smoothing) * personal + self.smoothing * global_score
        
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_items[:top_k]


@dataclass
class BaselineModels:
    """Container for baseline models."""
    popularity: PopularityRecommender
    personal: PersonalFrequencyRecommender


class BaselineTrainer:
    def run(self, split_data) -> BaselineModels:
        logger.info("=" * 50)
        logger.info("STEP E: Baseline Models")
        logger.info("=" * 50)
        
        train_df = split_data.train_df
        
        popularity = PopularityRecommender()
        popularity.fit(train_df)
        
        personal = PersonalFrequencyRecommender(smoothing=0.3)
        personal.fit(train_df)
        
        return BaselineModels(popularity=popularity, personal=personal)


def create_baseline_trainer():
    return BaselineTrainer()


if __name__ == "__main__":
    from A_data_preparation import create_data_preparation
    from B_feature_engineering import create_feature_engineering
    from C_training_data import create_training_data_builder
    from D_train_test_split import create_data_splitter
    
    data = create_data_preparation().run("data/data_raw.csv")
    features = create_feature_engineering().run(data)
    training = create_training_data_builder().run(data, features)
    split = create_data_splitter().run(training)
    baselines = create_baseline_trainer().run(split)
    
    logger.info(f"\nPopular: {baselines.popularity.predict(top_k=3)}")
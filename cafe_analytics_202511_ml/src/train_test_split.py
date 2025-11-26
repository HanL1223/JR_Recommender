"""
Step D: Train/Test Split (Temporal)
===================================
Splits data by TIME to simulate real-world prediction.
Train on past, test on future.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Output container for train/test split step."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: List[str]
    split_date: str


class BaseDataSplitter(ABC):
    @abstractmethod
    def run(self, training_data) -> SplitData:
        pass


class TemporalSplitter(BaseDataSplitter):
    """
    Temporal train/test split.
    
    Train on orders before cutoff date.
    Test on orders after cutoff date.
    """
    
    def __init__(self, test_ratio: float = 0.2, random_seed: int = 42):
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        logger.info(f"Initialized TemporalSplitter (test_ratio={test_ratio})")
    
    def run(self, training_data) -> SplitData:
        logger.info("=" * 50)
        logger.info("STEP D: Train/Test Split (TEMPORAL)")
        logger.info("=" * 50)
        
        df = training_data.samples_df.copy()
        feature_names = training_data.feature_names
        
        # Ensure order_date is datetime
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Sort by date and find cutoff
        df = df.sort_values('order_date')
        
        cutoff_idx = int(len(df) * (1 - self.test_ratio))
        cutoff_date = df.iloc[cutoff_idx]['order_date']
        
        train_df = df[df['order_date'] < cutoff_date].copy()
        test_df = df[df['order_date'] >= cutoff_date].copy()
        
        logger.info(f"Cutoff date: {cutoff_date.date()}")
        logger.info(f"Train: {len(train_df):,} samples ({train_df['order_date'].min().date()} to {train_df['order_date'].max().date()})")
        logger.info(f"Test: {len(test_df):,} samples ({test_df['order_date'].min().date()} to {test_df['order_date'].max().date()})")
        
        # Customer overlap analysis
        train_customers = set(train_df['customer_id'].unique())
        test_customers = set(test_df['customer_id'].unique())
        overlap = train_customers & test_customers
        new_in_test = test_customers - train_customers
        
        logger.info(f"Train customers: {len(train_customers):,}")
        logger.info(f"Test customers: {len(test_customers):,}")
        logger.info(f"Overlap: {len(overlap):,} ({100*len(overlap)/len(test_customers):.1f}% of test)")
        logger.info(f"New in test: {len(new_in_test):,}")
        
        # Extract features
        X_train = train_df[feature_names].values.astype(np.float32)
        X_test = test_df[feature_names].values.astype(np.float32)
        y_train = train_df['label'].values.astype(np.float32)
        y_test = test_df['label'].values.astype(np.float32)
        
        # Handle nan/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"\nTrain positive rate: {y_train.mean():.2%}")
        logger.info(f"Test positive rate: {y_test.mean():.2%}")
        
        return SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_df=train_df,
            test_df=test_df,
            feature_names=feature_names,
            split_date=str(cutoff_date.date())
        )


def create_data_splitter(test_ratio: float = 0.2) -> BaseDataSplitter:
    return TemporalSplitter(test_ratio=test_ratio)


if __name__ == "__main__":
    from A_data_preparation import create_data_preparation
    from B_feature_engineering import create_feature_engineering
    from C_training_data import create_training_data_builder
    
    data = create_data_preparation().run("data/data_raw.csv")
    features = create_feature_engineering().run(data)
    training = create_training_data_builder().run(data, features)
    split = create_data_splitter().run(training)
    
    logger.info(f"\nTrain shape: {split.X_train.shape}")
    logger.info(f"Test shape: {split.X_test.shape}")
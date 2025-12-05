import logging
import pandas as pd
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_date: str
    feature_names: List[str]
    n_train_samples: int
    n_test_samples: int
    n_train_customers: int
    n_test_customers: int
    overlap_customers: int
    new_customers: int



class TemporalDataSplitter:
    """
    Chronological (time-based) train/test splitting.
    - Train on the past
    - Test on the future
    - Prevents leakage that occurs with random splitting
    """

    def __init__(self, test_ratio: float = 0.2):
        self.test_ratio = test_ratio
        logger.info(f"TemporalDataSplitter initialized (test_ratio={test_ratio})")

    def split(self, training_data, date_column: str = "order_date") -> SplitData:
        logger.info("Performing temporal train/test split")

        df = training_data.samples_df.copy()

        # Ensure date column exists
        if date_column not in df.columns:
            raise KeyError(f"date_column '{date_column}' not found in training dataset")

        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        # Determine split date
        sorted_dates = df[date_column].sort_values()
        split_index = int(len(sorted_dates) * (1 - self.test_ratio))

        split_date = sorted_dates.iloc[split_index]

        logger.info(f"Split date determined: {split_date.strftime('%Y-%m-%d')}")

        # Split sets
        train_df = df[df[date_column] < split_date].copy()
        test_df = df[df[date_column] >= split_date].copy()

        # Customer stats
        train_customers = set(train_df.customer_id.unique())
        test_customers = set(test_df.customer_id.unique())

        overlap_customers = len(train_customers & test_customers)
        new_customers = len(test_customers - train_customers)

        logger.info(
            f"Train samples: {len(train_df):,} | Test samples: {len(test_df):,}"
        )
        logger.info(
            f"Customers: Train={len(train_customers)}, Test={len(test_customers)}, "
            f"Overlap={overlap_customers}, New={new_customers}"
        )

        # Positive rate check
        logger.info(
            f"Label positive-rate → Train={train_df['label'].mean():.2%} | "
            f"Test={test_df['label'].mean():.2%}"
        )

        return SplitData(
            train_df=train_df,
            test_df=test_df,
            split_date=split_date.strftime("%Y-%m-%d"),
            feature_names=training_data.feature_names,
            n_train_samples=len(train_df),
            n_test_samples=len(test_df),
            n_train_customers=len(train_customers),
            n_test_customers=len(test_customers),
            overlap_customers=overlap_customers,
            new_customers=new_customers,
        )
    
    #Split by hard coded date 
    def split_by_date(self, training_data, split_date: str, date_column: str = "order_date") -> SplitData:
        logger.info(f"Splitting at specific date: {split_date}")

        df = training_data.samples_df.copy()
        split_dt = pd.to_datetime(split_date)

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        train_df = df[df[date_column] < split_dt].copy()
        test_df = df[df[date_column] >= split_dt].copy()

        train_customers = set(train_df.customer_id.unique())
        test_customers = set(test_df.customer_id.unique())

        return SplitData(
            train_df=train_df,
            test_df=test_df,
            split_date=split_date,
            feature_names=training_data.feature_names,
            n_train_samples=len(train_df),
            n_test_samples=len(test_df),
            n_train_customers=len(train_customers),
            n_test_customers=len(test_customers),
            overlap_customers=len(train_customers & test_customers),
            new_customers=len(test_customers - train_customers),
        )


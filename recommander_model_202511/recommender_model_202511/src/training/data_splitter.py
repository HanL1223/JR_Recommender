import logging
import pandas as pd
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame

    split_date_train: str
    split_date_valid: str

    feature_names: List[str]

    n_train_samples: int
    n_valid_samples: int
    n_test_samples: int

    n_train_customers: int
    n_valid_customers: int
    n_test_customers: int

    overlap_train_valid: int
    overlap_valid_test: int
    new_customers_valid: int
    new_customers_test: int


#Train/Val/Test Data splitter
class TemporalDataSplitter:
    """
    Chronological (time-based) train/test splitting.
    - Train on the past
    - Validate on the intermediate period
    - Test on the future
    """

    def __init__(self, 
                 valid_ratio: float = 0.1,
                 test_ratio: float = 0.2):
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        logger.info(
            f"TemporalDataSplitter initialized (valid_ratio={valid_ratio}, "
            f"test_ratio={test_ratio})"
        )

    def split(self, training_data, date_column: str = "order_date") -> SplitData:
        logger.info("Performing temporal train/valid/test split")

        df = training_data.samples_df.copy()

        # Ensure date column exists
        if date_column not in df.columns:
            raise KeyError(f"date_column '{date_column}' not found in training dataset")

        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        # Sort in chronologically order
        df = df.sort_values(date_column)

        n = len(df)
        idx_valid = int(n * (1 - self.test_ratio - self.valid_ratio)) # validation index begin at 70% of dataset
        idx_test = int(n * (1 - self.test_ratio)) #test index begin at 80% of dataset

        split_date_train = df[date_column].iloc[idx_valid]
        split_date_valid = df[date_column].iloc[idx_test]

        logger.info(f"Train split date: {split_date_train}")
        logger.info(f"Valid split date: {split_date_valid}")

        #Spliting into dataframe

        train_df = df[df[date_column] < split_date_train].copy()
        valid_df = df[
            (df[date_column] >= split_date_train) &
            (df[date_column] < split_date_valid)
        ].copy()
        test_df = df[df[date_column] >= split_date_valid].copy()


        #Stats
        train_customers = set(train_df.customer_id.unique())
        valid_customers = set(valid_df.customer_id.unique())
        test_customers = set(test_df.customer_id.unique())

        return SplitData(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,

            split_date_train=split_date_train.strftime("%Y-%m-%d"),
            split_date_valid=split_date_valid.strftime("%Y-%m-%d"),

            feature_names=training_data.feature_names,

            n_train_samples=len(train_df),
            n_valid_samples=len(valid_df),
            n_test_samples=len(test_df),

            n_train_customers=len(train_customers),
            n_valid_customers=len(valid_customers),
            n_test_customers=len(test_customers),

            overlap_train_valid=len(train_customers & valid_customers),
            overlap_valid_test=len(valid_customers & test_customers),

            new_customers_valid=len(valid_customers - train_customers),
            new_customers_test=len(test_customers - (train_customers | valid_customers)),
        )
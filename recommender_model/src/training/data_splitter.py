"""
Data Splitter
=============
Temporal train/test split for time-series data.

CRITICAL: Uses temporal split to prevent data leakage.

Refactored from: D_train_test_split.py
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for split data."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_date: str
    feature_names: list
    n_train_samples: int
    n_test_samples: int
    n_train_customers: int
    n_test_customers: int
    overlap_customers: int
    new_customers: int


class TemporalDataSplitter:
    """
    Temporal train/test splitter.
    
    CRITICAL: Splits by time to simulate real-world prediction:
    - Train on past data
    - Test on future data
    
    This prevents data leakage that would occur with random splits.
    
    Example:
        >>> splitter = TemporalDataSplitter(test_ratio=0.2)
        >>> split = splitter.split(training_data)
        >>> print(f"Split date: {split.split_date}")
    """
    
    def __init__(self, test_ratio: float = 0.2):
        """
        Initialize splitter.
        
        Args:
            test_ratio: Fraction of data for test set (by time, not samples)
        """
        self.test_ratio = test_ratio
        logger.info(f"TemporalDataSplitter initialized (test_ratio={test_ratio})")
    
    def split(
        self, 
        training_data,
        date_column: str = 'order_date'
    ) -> SplitData:
        """
        Split data temporally.
        
        Args:
            training_data: TrainingData object
            date_column: Column containing dates
            
        Returns:
            SplitData with train/test DataFrames
        """
        logger.info("Performing temporal train/test split...")
        
        df = training_data.samples_df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Find split date based on ratio
        dates = df[date_column].sort_values()
        split_idx = int(len(dates) * (1 - self.test_ratio))
        split_date = dates.iloc[split_idx]
        
        # Split
        train_df = df[df[date_column] < split_date].copy()
        test_df = df[df[date_column] >= split_date].copy()
        
        # Statistics
        train_customers = set(train_df['customer_id'].unique())
        test_customers = set(test_df['customer_id'].unique())
        overlap = train_customers & test_customers
        new_customers = test_customers - train_customers
        
        split_date_str = split_date.strftime('%Y-%m-%d')
        
        logger.info(f"Split date: {split_date_str}")
        logger.info(f"Train: {len(train_df):,} samples ({train_df[date_column].min().strftime('%Y-%m-%d')} to {split_date_str})")
        logger.info(f"Test: {len(test_df):,} samples ({split_date_str} to {test_df[date_column].max().strftime('%Y-%m-%d')})")
        logger.info(f"Train customers: {len(train_customers):,}")
        logger.info(f"Test customers: {len(test_customers):,} ({len(overlap):,} overlap, {len(new_customers):,} new)")
        
        # Check positive rate balance
        train_pos = train_df['label'].mean()
        test_pos = test_df['label'].mean()
        logger.info(f"Positive rate - Train: {train_pos:.1%}, Test: {test_pos:.1%}")
        
        return SplitData(
            train_df=train_df,
            test_df=test_df,
            split_date=split_date_str,
            feature_names=training_data.feature_names,
            n_train_samples=len(train_df),
            n_test_samples=len(test_df),
            n_train_customers=len(train_customers),
            n_test_customers=len(test_customers),
            overlap_customers=len(overlap),
            new_customers=len(new_customers)
        )
    
    def split_by_date(
        self,
        training_data,
        split_date: str,
        date_column: str = 'order_date'
    ) -> SplitData:
        """
        Split data at a specific date.
        
        Args:
            training_data: TrainingData object
            split_date: Date string (YYYY-MM-DD)
            date_column: Column containing dates
            
        Returns:
            SplitData
        """
        logger.info(f"Splitting at date: {split_date}")
        
        df = training_data.samples_df.copy()
        split_dt = pd.to_datetime(split_date)
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        train_df = df[df[date_column] < split_dt].copy()
        test_df = df[df[date_column] >= split_dt].copy()
        
        train_customers = set(train_df['customer_id'].unique())
        test_customers = set(test_df['customer_id'].unique())
        
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
            new_customers=len(test_customers - train_customers)
        )
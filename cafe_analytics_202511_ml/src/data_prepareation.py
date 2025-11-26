"""
Step A: Data Preparation
========================
Load and prepare order data for recommendation modeling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Output container for data preparation step."""
    orders_df: pd.DataFrame
    transactions_df: pd.DataFrame
    customer_histories: Dict[int, List[dict]]  # customer_id -> list of order dicts
    product_list: List[str]
    customer_list: List[int]
    category_map: Dict[str, str]


class BaseDataPreparation(ABC):
    @abstractmethod
    def run(self, filepath: str) -> PreparedData:
        pass


class DataPreparation(BaseDataPreparation):
    """Prepares raw transaction data for recommendation modeling."""
    
    def __init__(self, min_orders: int = 2):
        self.min_orders = min_orders
        logger.info(f"Initialized DataPreparation (min_orders={min_orders})")
    
    def run(self, filepath: str) -> PreparedData:
        logger.info("=" * 50)
        logger.info("STEP A: Data Preparation")
        logger.info("=" * 50)
        
        # Load data
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} transaction rows")
        
        # Parse dates
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Create product key
        df['product'] = df['product_name'] + ' (' + df['product_variant'] + ')'
        
        # Create category map
        category_map = dict(zip(df['product'], df['product_category']))
        
        # Sort by customer and date
        df = df.sort_values(['customer_id', 'order_date', 'order_id', 'order_time'])
        
        logger.info(f"Unique customers: {df['customer_id'].nunique():,}")
        logger.info(f"Unique orders: {df['order_id'].nunique():,}")
        logger.info(f"Unique products: {df['product'].nunique()}")
        logger.info(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
        
        # Aggregate to order level
        orders_df = df.groupby('order_id').agg({
            'customer_id': 'first',
            'order_date': 'first',
            'order_time': 'first',
            'order_total_price': 'first',
            'product': list,
            'product_category': lambda x: list(set(x)),
            'customer_segment': 'first'
        }).reset_index()
        orders_df = orders_df.rename(columns={'product': 'basket'})
        orders_df['basket_size'] = orders_df['basket'].apply(len)
        orders_df = orders_df.sort_values(['customer_id', 'order_date', 'order_time'])
        
        logger.info(f"Aggregated to {len(orders_df):,} orders")
        
        # Build customer histories with full order info
        customer_histories = {}
        for customer_id, group in orders_df.groupby('customer_id'):
            group = group.sort_values(['order_date', 'order_time'])
            orders = []
            for _, row in group.iterrows():
                orders.append({
                    'order_id': row['order_id'],
                    'order_date': row['order_date'],
                    'order_time': row['order_time'],
                    'basket': row['basket'],
                    'basket_size': row['basket_size'],
                    'order_total': row['order_total_price'],
                    'categories': row['product_category'],
                    'segment': row['customer_segment']
                })
            if len(orders) >= self.min_orders:
                customer_histories[customer_id] = orders
        
        logger.info(f"Customers with {self.min_orders}+ orders: {len(customer_histories):,}")
        
        product_list = df['product'].unique().tolist()
        customer_list = list(customer_histories.keys())
        
        return PreparedData(
            orders_df=orders_df,
            transactions_df=df,
            customer_histories=customer_histories,
            product_list=product_list,
            customer_list=customer_list,
            category_map=category_map
        )


def create_data_preparation(min_orders: int = 2) -> BaseDataPreparation:
    return DataPreparation(min_orders=min_orders)


if __name__ == "__main__":
    data = create_data_preparation().run("data/data_raw.csv")
    sample_cust = data.customer_list[0]
    logger.info(f"\nSample customer {sample_cust} orders: {len(data.customer_histories[sample_cust])}")
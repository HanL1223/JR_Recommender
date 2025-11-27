"""
Data Preprocessor
=================
Transforms raw data into customer histories for modeling.

Refactored from: A_data_preparation.py
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Container for preprocessed data ready for feature engineering."""
    customer_histories: Dict[int, List[dict]]  # customer_id -> list of orders
    customer_list: List[int]
    product_list: List[str]
    category_map: Dict[str, str]  # product -> category
    n_customers: int
    n_products: int
    n_orders: int


class DataPreprocessor:
    """
    Transforms raw transactions into customer order histories.
    
    Steps:
    1. Aggregate transactions to order level (basket)
    2. Build customer histories (sequence of orders)
    3. Filter customers with minimum orders
    4. Extract product catalog and categories
    
    Example:
        >>> preprocessor = DataPreprocessor(min_orders=2)
        >>> prepared = preprocessor.transform(raw_data.transactions)
        >>> print(f"Customers with 2+ orders: {prepared.n_customers}")
    """
    
    def __init__(self, min_orders: int = 2):
        """
        Initialize preprocessor.
        
        Args:
            min_orders: Minimum orders required per customer
        """
        self.min_orders = min_orders
        logger.info(f"DataPreprocessor initialized (min_orders={min_orders})")
    
    def transform(self, df: pd.DataFrame) -> PreparedData:
        """
        Transform raw transactions to customer histories.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            PreparedData object with customer histories
        """
        logger.info("Transforming raw data to customer histories...")
        
        # Step 1: Aggregate to order level
        orders_df = self._aggregate_to_orders(df)
        
        # Step 2: Build customer histories
        customer_histories = self._build_customer_histories(orders_df)
        
        # Step 3: Filter by minimum orders
        filtered_histories = {
            cid: orders for cid, orders in customer_histories.items()
            if len(orders) >= self.min_orders
        }
        
        # Step 4: Extract product catalog
        product_list = self._extract_products(df)
        category_map = self._build_category_map(df)
        
        n_customers = len(filtered_histories)
        n_orders = sum(len(orders) for orders in filtered_histories.values())
        
        logger.info(f"Transformation complete:")
        logger.info(f"  Customers with {self.min_orders}+ orders: {n_customers:,}")
        logger.info(f"  Total orders: {n_orders:,}")
        logger.info(f"  Unique products: {len(product_list):,}")
        
        return PreparedData(
            customer_histories=filtered_histories,
            customer_list=list(filtered_histories.keys()),
            product_list=product_list,
            category_map=category_map,
            n_customers=n_customers,
            n_products=len(product_list),
            n_orders=n_orders
        )
    
    def _aggregate_to_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction rows to order level with baskets."""
        logger.info("Aggregating transactions to orders...")
        
        # Ensure product column exists
        if 'product' not in df.columns:
            df = df.copy()
            df['product'] = df['product_name'] + ' (' + df['product_variant'].fillna('Regular') + ')'
        
        # Group by order
        orders = df.groupby('order_id').agg({
            'customer_id': 'first',
            'order_date': 'first',
            'order_time': 'first' if 'order_time' in df.columns else lambda x: '12:00:00',
            'order_total_price': 'first',
            'product': list,  # Basket as list
            'product_category': lambda x: list(x) if 'product_category' in df.columns else [],
            'customer_segment': 'first' if 'customer_segment' in df.columns else lambda x: 'Regular'
        }).reset_index()
        
        orders.columns = ['order_id', 'customer_id', 'order_date', 'order_time', 
                         'order_total', 'basket', 'categories', 'segment']
        
        # Calculate basket size
        orders['basket_size'] = orders['basket'].apply(len)
        
        logger.info(f"Aggregated to {len(orders):,} orders")
        return orders
    
    def _build_customer_histories(self, orders_df: pd.DataFrame) -> Dict[int, List[dict]]:
        """Build chronological order histories per customer."""
        logger.info("Building customer histories...")
        
        # Sort by date
        orders_df = orders_df.sort_values(['customer_id', 'order_date'])
        
        histories = defaultdict(list)
        
        for _, row in orders_df.iterrows():
            order = {
                'order_id': row['order_id'],
                'order_date': row['order_date'],
                'order_time': row['order_time'],
                'order_total': row['order_total'],
                'basket': row['basket'],
                'basket_size': row['basket_size'],
                'categories': row['categories'],
                'segment': row['segment']
            }
            histories[row['customer_id']].append(order)
        
        logger.info(f"Built histories for {len(histories):,} customers")
        return dict(histories)
    
    def _extract_products(self, df: pd.DataFrame) -> List[str]:
        """Extract unique product list."""
        if 'product' not in df.columns:
            df = df.copy()
            df['product'] = df['product_name'] + ' (' + df['product_variant'].fillna('Regular') + ')'
        
        products = df['product'].dropna().unique().tolist()
        # Filter out non-string products (NaN that weren't caught)
        products = [p for p in products if isinstance(p, str)]
        return sorted(products)
    
    def _build_category_map(self, df: pd.DataFrame) -> Dict[str, str]:
        """Build product -> category mapping."""
        if 'product_category' not in df.columns:
            return {}
        
        if 'product' not in df.columns:
            df = df.copy()
            df['product'] = df['product_name'] + ' (' + df['product_variant'].fillna('Regular') + ')'
        
        category_map = {}
        for _, row in df[['product', 'product_category']].drop_duplicates().iterrows():
            if isinstance(row['product'], str):
                category_map[row['product']] = row['product_category']
        
        return category_map
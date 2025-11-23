import pandas as pd
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparation:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logger
        self.original_shape = df.shape

    def remove_duplicates(self):
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)

        self.logger.info(f"Removed {removed} duplicate rows")

    def handle_missing_values(self):
        null_counts = self.df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if len(cols_with_nulls)> 0:
            self.logger.info("Columns with missing values:")
            for col, count in cols_with_nulls.items():
                pct = (count / len(self.df)) * 100
                self.logger.info(f"  {col}: {count} ({pct:.1f}%)")
        else:
            self.logger.info(f"No missing value found")
        #Sepcial handler for product variant
        if 'product_variant' in cols_with_nulls.index:
            self.df['product_variant'].fillna('Regular', inplace=True)
            self.logger.info("Filled missing product_variant with 'Regular'")
        if 'order_gst' in cols_with_nulls.index:
            self.df['order_gst'].fillna(0, inplace=True)
            self.logger.info("Filled missing order_gst with 0")
        return self
    
    def validate_product_categories(self) -> 'DataPreparation':
        
        product_category_map = self.df.groupby('product_name')['product_category'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).to_dict()
        
        inconsistent_products = self.df.groupby('product_name')['product_category'].nunique()
        inconsistent_products = inconsistent_products[inconsistent_products > 1]
        
        if len(inconsistent_products) > 0:
            self.logger.info(f"Found {len(inconsistent_products)} products with inconsistent categories")
            self.logger.info("Fixing inconsistent categories using most frequent value")
            
            for product in inconsistent_products.index:
                correct_category = product_category_map[product]
                mask = self.df['product_name'] == product
                self.df.loc[mask, 'product_category'] = correct_category
            
            self.logger.info("Fixed all inconsistent product categories")
        else:
            self.logger.info("All product categories are consistent")
        
        return self
    
    def validate_prices(self):
        negative_prices = self.df[self.df['order_total_price'] < 0]
        if len(negative_prices) > 0:
            self.logger.warning(f"Found {len(negative_prices)} orders with negative prices")
            self.df = self.df[self.df['order_total_price'] >= 0]
            self.logger.info(f"Removed orders with negative prices")
        else:
            self.logger.info("All prices are valid")
        
        zero_prices = self.df[self.df['order_total_price'] == 0]
        if len(zero_prices) > 0:
            self.logger.warning(f"Found {len(zero_prices)} orders with zero price")
            self.df = self.df[self.df['order_total_price'] > 0]
            self.logger.info(f"Removed orders with zero price")
        
        return self
    def create_composite_features(self):
        
        self.df['product_full'] = self.df['product_name'] + '_' + self.df['product_variant']
        self.logger.info("Created product_full: product_name + product_variant")
        
        self.df['order_datetime'] = pd.to_datetime(
    df['order_date'].astype(str) + ' ' + df['order_time'].astype(str)
)
        self.logger.info("Created order_datetime from order_date and order_time")
        
        self.df['hour'] = self.df['order_datetime'].dt.hour
        self.df['day_of_week'] = self.df['order_datetime'].dt.dayofweek
        self.df['month'] = self.df['order_datetime'].dt.month
        self.df['year'] = self.df['order_datetime'].dt.year
        self.logger.info("Created temporal features: hour, day_of_week, month, year")
        
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.logger.info("Created is_weekend feature")
        
        return self
    
    def sort_by_datetime(self) -> 'DataPreparation':
        
        self.df = self.df.sort_values(['customer_id', 'order_datetime'])
        self.df = self.df.reset_index(drop=True)
        self.logger.info("Sorted by customer_id and order_datetime")
        
        return self
    
    def validate_customer_features(self) -> 'DataPreparation':
        
        inconsistent_ltv = self.df.groupby('customer_id')['total_lifetime_value'].nunique()
        inconsistent_ltv = inconsistent_ltv[inconsistent_ltv > 1]
        
        if len(inconsistent_ltv) > 0:
            self.logger.warning(f"Found {len(inconsistent_ltv)} customers with inconsistent lifetime values")
            
            customer_ltv_map = self.df.groupby('customer_id')['total_lifetime_value'].max().to_dict()
            for customer_id, ltv in customer_ltv_map.items():
                self.df.loc[self.df['customer_id'] == customer_id, 'total_lifetime_value'] = ltv
            
            self.logger.info("Fixed inconsistent lifetime values using maximum value")
        else:
            self.logger.info("Customer lifetime values are consistent")
        
        return self
    
    def generate_cleaning_report(self) -> dict:
        
        report = {
            'original_rows': self.original_shape[0],
            'final_rows': len(self.df),
            'rows_removed': self.original_shape[0] - len(self.df),
            'removal_percentage': ((self.original_shape[0] - len(self.df)) / self.original_shape[0]) * 100,
            'unique_customers': self.df['customer_id'].nunique(),
            'unique_products': self.df['product_name'].nunique(),
            'unique_orders': self.df['order_id'].nunique(),
            'date_range': (self.df['order_date'].min(), self.df['order_date'].max())
        }
        
        self.logger.info(f"Original rows: {report['original_rows']:,}")
        self.logger.info(f"Final rows: {report['final_rows']:,}")
        self.logger.info(f"Rows removed: {report['rows_removed']:,} ({report['removal_percentage']:.2f}%)")
        self.logger.info(f"Unique customers: {report['unique_customers']:,}")
        self.logger.info(f"Unique products: {report['unique_products']}")
        self.logger.info(f"Unique orders: {report['unique_orders']:,}")
        
        return report
    def run_preparation(self) -> Tuple[pd.DataFrame, dict]:
        
        self.remove_duplicates()
        self.handle_missing_values()
        self.validate_product_categories()
        self.validate_prices()
        self.validate_customer_features()
        self.create_composite_features()
        self.sort_by_datetime()
        
        report = self.generate_cleaning_report()
        
        self.logger.info("="*80)
        self.logger.info("Data is now clean")
        self.logger.info("="*80)
        return self.df, report

if __name__ == "__main__":
    logger.info("Usage: prep = DataPreparation(df)")
    logger.info("clean_df, report = prep.run_preparation()")

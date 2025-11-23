import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logger
        
        required_columns = [
            'order_id', 'order_item_id', 'order_date', 'order_time', 
            'order_total_price', 'order_gst', 'cart_order_time',
            'product_name', 'product_variant', 'product_category',
            'customer_id', 'first_order_date', 'last_order_date',
            'total_orders', 'total_lifetime_value', 'customer_segment'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        self.logger.info("DataAnalyzer initialized successfully")
        
    def analyze_structure(self) -> Dict:
        self.logger.info("="*80)
        self.logger.info("DATA STRUCTURE ANALYSIS")
        self.logger.info("="*80)
        
        self.logger.info(f"Dataset shape: {self.df.shape}")
        self.logger.info(f"Date range: {self.df['order_date'].min()} to {self.df['order_date'].max()}")
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique_count = self.df[col].nunique()
            
            self.logger.info(f"{col}: dtype={dtype}, nulls={null_count} ({null_pct:.1f}%), unique={unique_count}")
        
        structure_info = {
            'n_rows': len(self.df),
            'n_cols': len(self.df.columns),
            'date_range': (self.df['order_date'].min(), self.df['order_date'].max())
        }
        
        return structure_info
    
    def analyze_customers(self) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("CUSTOMER ANALYSIS")
        self.logger.info("="*80)
        
        customer_summary = self.df.groupby('customer_id').agg({
            'order_id': 'nunique',
            'total_orders': 'first',
            'total_lifetime_value': 'first',
            'customer_segment': 'first',
            'first_order_date': 'first',
            'last_order_date': 'first'
        }).reset_index()
        
        self.logger.info(f"Total unique customers: {len(customer_summary)}")
        
        segment_dist = customer_summary['customer_segment'].value_counts()
        self.logger.info("Customer segments:")
        for segment, count in segment_dist.items():
            pct = (count / len(customer_summary)) * 100
            self.logger.info(f"  {segment}: {count} ({pct:.1f}%)")
        
        self.logger.info("Order frequency distribution:")
        self.logger.info(f"{customer_summary['total_orders'].describe()}")
        
        self.logger.info("Lifetime value distribution:")
        self.logger.info(f"{customer_summary['total_lifetime_value'].describe()}")
        
        return customer_summary
    
    def analyze_products(self) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("PRODUCT ANALYSIS")
        self.logger.info("="*80)
        
        self.logger.info(f"Total unique products: {self.df['product_name'].nunique()}")
        self.logger.info(f"Total unique variants: {self.df['product_variant'].nunique()}")
        
        category_dist = self.df.groupby('product_category').agg({
            'order_id': 'count',
            'product_name': 'nunique'
        }).rename(columns={'order_id': 'order_count', 'product_name': 'unique_products'})
        
        self.logger.info("Product categories:")
        self.logger.info(f"\n{category_dist}")
        
        self.logger.info("Top 15 most popular products:")
        popular_products = self.df.groupby(['product_name', 'product_variant']).size().sort_values(ascending=False).head(15)
        for (product, variant), count in popular_products.items():
            self.logger.info(f"  {product} ({variant}): {count}")
        
        return category_dist
    
    def analyze_temporal_patterns(self) -> Dict:
        self.logger.info("="*80)
        self.logger.info("TEMPORAL PATTERN ANALYSIS")
        self.logger.info("="*80)
        
        self.df['order_datetime'] = pd.to_datetime(
    df['order_date'].astype(str) + ' ' + df['order_time'].astype(str)
)
        self.df['hour'] = self.df['order_datetime'].dt.hour
        self.df['day_of_week'] = self.df['order_datetime'].dt.dayofweek
        
        hourly = self.df.groupby('hour')['order_id'].nunique().sort_values(ascending=False)
        self.logger.info("Orders by hour of day (top 5):")
        for hour, count in hourly.head(5).items():
            self.logger.info(f"  {hour:02d}:00 - {count} orders")
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = self.df.groupby('day_of_week')['order_id'].nunique()
        self.logger.info("Orders by day of week:")
        for day_idx, count in daily.items():
            self.logger.info(f"  {days[day_idx]}: {count} orders")
        
        temporal_info = {
            'peak_hour': hourly.idxmax(),
            'peak_day': days[daily.idxmax()]
        }
        
        return temporal_info
    
    def check_data_quality(self) -> Dict:
        self.logger.info("="*80)
        self.logger.info("DATA QUALITY CHECKS")
        self.logger.info("="*80)
        
        issues = []
        
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
            self.logger.warning(f"Found {duplicates} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")
        
        product_category_check = self.df.groupby('product_name')['product_category'].nunique()
        inconsistent = product_category_check[product_category_check > 1]
        if len(inconsistent) > 0:
            issues.append(f"{len(inconsistent)} products with multiple categories")
            self.logger.warning(f"{len(inconsistent)} products have inconsistent categories")
        else:
            self.logger.info("Product categories are consistent")
        
        negative_prices = (self.df['order_total_price'] < 0).sum()
        if negative_prices > 0:
            issues.append(f"{negative_prices} orders with negative prices")
            self.logger.warning(f"{negative_prices} orders with negative prices")
        else:
            self.logger.info("All prices are valid")
        
        quality_report = {
            'duplicates': duplicates,
            'inconsistent_categories': len(inconsistent),
            'negative_prices': negative_prices,
            'issues': issues
        }
        
        if len(issues) == 0:
            self.logger.info("Data quality checks passed")
        
        return quality_report
    
    def analyze_recommendation_feasibility(self) -> Dict:
        self.logger.info("="*80)
        self.logger.info("RECOMMENDATION SYSTEM FEASIBILITY")
        self.logger.info("="*80)
        
        customers_with_history = self.df.groupby('customer_id')['order_id'].nunique()
        sufficient_history = (customers_with_history >= 3).sum()
        self.logger.info(f"Customers with 3+ orders: {sufficient_history} ({sufficient_history/len(customers_with_history)*100:.1f}%)")
        
        avg_unique_products = self.df.groupby('customer_id')['product_name'].nunique().mean()
        total_products = self.df['product_name'].nunique()
        self.logger.info(f"Average products per customer: {avg_unique_products:.1f}")
        self.logger.info(f"Total available products: {total_products}")
        self.logger.info(f"Average unexplored products per customer: {total_products - avg_unique_products:.1f}")
        
        customer_product_matrix = self.df.groupby(['customer_id', 'product_name']).size().unstack(fill_value=0)
        sparsity = 1 - (customer_product_matrix.astype(bool).sum().sum() / (customer_product_matrix.shape[0] * customer_product_matrix.shape[1]))
        self.logger.info(f"User-item matrix shape: {customer_product_matrix.shape}")
        self.logger.info(f"Matrix sparsity: {sparsity * 100:.1f}%")
        
        feasibility_report = {
            'sufficient_history_pct': sufficient_history/len(customers_with_history),
            'avg_products_per_customer': avg_unique_products,
            'total_products': total_products,
            'matrix_sparsity': sparsity
        }
        
        return feasibility_report
    
    def generate_summary(self) -> Dict:
        self.logger.info("="*80)
        self.logger.info("SUMMARY REPORT")
        self.logger.info("="*80)
        
        summary = {
            'total_records': len(self.df),
            'unique_customers': self.df['customer_id'].nunique(),
            'unique_products': self.df['product_name'].nunique(),
            'unique_orders': self.df['order_id'].nunique(),
            'date_range': (self.df['order_date'].min(), self.df['order_date'].max()),
            'avg_items_per_order': len(self.df) / self.df['order_id'].nunique(),
            'avg_orders_per_customer': self.df.groupby('customer_id')['order_id'].nunique().mean()
        }
        
        self.logger.info(f"Total records: {summary['total_records']:,}")
        self.logger.info(f"Unique customers: {summary['unique_customers']:,}")
        self.logger.info(f"Unique products: {summary['unique_products']}")
        self.logger.info(f"Unique orders: {summary['unique_orders']:,}")
        self.logger.info(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
        self.logger.info(f"Average items per order: {summary['avg_items_per_order']:.2f}")
        self.logger.info(f"Average orders per customer: {summary['avg_orders_per_customer']:.2f}")
        
        return summary
    
    def run_analysis(self) -> Dict:
        structure_info = self.analyze_structure()
        customer_summary = self.analyze_customers()
        product_summary = self.analyze_products()
        temporal_info = self.analyze_temporal_patterns()
        quality_report = self.check_data_quality()
        feasibility_report = self.analyze_recommendation_feasibility()
        summary = self.generate_summary()
        
        self.logger.info("="*80)
        self.logger.info("STEP 1 COMPLETE")
        self.logger.info("="*80)
        
        return {
            'structure': structure_info,
            'customer_summary': customer_summary,
            'product_summary': product_summary,
            'temporal': temporal_info,
            'quality': quality_report,
            'feasibility': feasibility_report,
            'summary': summary
        }


if __name__ == "__main__":
    logger.info("STEP 1: Data Analysis and Understanding")
    logger.info("Usage: analyzer = DataAnalyzer(df)")
    logger.info("       results = analyzer.run_analysis()")
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderAwareAugmentation:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logger
        self.original_size = len(df)
        
    def augment_order_temporal_jitter(self, jitter_hours: int = 2, probability: float = 0.3) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("AUGMENTATION 1: ORDER-LEVEL TEMPORAL JITTERING")
        self.logger.info("="*80)
        self.logger.info("Shifts entire orders in time while maintaining item relationships")
        
        orders = self.df.groupby('order_id')
        eligible_orders = list(orders.groups.keys())
        
        n_to_jitter = int(len(eligible_orders) * probability)
        orders_to_jitter = np.random.choice(eligible_orders, size=n_to_jitter, replace=False)
        
        augmented_records = []
        for order_id in orders_to_jitter:
            order_data = self.df[self.df['order_id'] == order_id].copy()
            
            jitter_minutes = np.random.randint(-jitter_hours*60, jitter_hours*60)
            order_data['order_datetime'] = order_data['order_datetime'] + pd.to_timedelta(jitter_minutes, unit='m')
            
            order_data['order_date'] = order_data['order_datetime'].dt.date.astype(str)
            order_data['order_time'] = order_data['order_datetime'].dt.time.astype(str)
            order_data['hour'] = order_data['order_datetime'].dt.hour
            order_data['day_of_week'] = order_data['order_datetime'].dt.dayofweek
            order_data['is_weekend'] = order_data['day_of_week'].isin([5, 6]).astype(int)
            
            order_data['order_id'] = order_data['order_id'] + 10000000
            order_data['order_item_id'] = order_data['order_item_id'] + 10000000
            
            augmented_records.append(order_data)
        
        if augmented_records:
            result = pd.concat(augmented_records, ignore_index=True)
            self.logger.info(f"Generated {len(result)} records from {n_to_jitter} jittered orders")
            return result
        else:
            return pd.DataFrame()
    
    def augment_basket_item_swap(self, probability: float = 0.2) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("AUGMENTATION 2: WITHIN-BASKET ITEM SWAP")
        self.logger.info("="*80)
        self.logger.info("Swaps items within same order while preserving order structure")
        
        variant_groups = self.df.groupby('product_name')['product_variant'].unique().to_dict()
        variant_groups = {k: v for k, v in variant_groups.items() if len(v) > 1}
        
        orders = self.df.groupby('order_id')
        eligible_orders = [
            order_id for order_id, group in orders 
            if any(group['product_name'].isin(variant_groups.keys()))
        ]
        
        n_to_augment = int(len(eligible_orders) * probability)
        orders_to_augment = np.random.choice(eligible_orders, size=n_to_augment, replace=False)
        
        augmented_records = []
        for order_id in orders_to_augment:
            order_data = self.df[self.df['order_id'] == order_id].copy()
            
            for idx in order_data.index:
                product_name = order_data.loc[idx, 'product_name']
                if product_name in variant_groups and np.random.random() < 0.5:
                    current_variant = order_data.loc[idx, 'product_variant']
                    possible_variants = variant_groups[product_name]
                    other_variants = [v for v in possible_variants if v != current_variant]
                    
                    if len(other_variants) > 0:
                        new_variant = np.random.choice(other_variants)
                        order_data.loc[idx, 'product_variant'] = new_variant
                        order_data.loc[idx, 'product_full'] = f"{product_name}_{new_variant}"
            
            order_data['order_id'] = order_data['order_id'] + 20000000
            order_data['order_item_id'] = order_data['order_item_id'] + 20000000
            
            augmented_records.append(order_data)
        
        if augmented_records:
            result = pd.concat(augmented_records, ignore_index=True)
            self.logger.info(f"Generated {len(result)} records from {n_to_augment} augmented orders")
            return result
        else:
            return pd.DataFrame()
    
    def augment_basket_item_addition(self, probability: float = 0.15) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("AUGMENTATION 3: BASKET ITEM ADDITION")
        self.logger.info("="*80)
        self.logger.info("Adds frequently co-occurring items to existing orders")
        
        item_cooccurrence = {}
        for order_id, group in self.df.groupby('order_id'):
            items = group['product_full'].tolist()
            for i, item1 in enumerate(items):
                for item2 in items[i+1:]:
                    key = tuple(sorted([item1, item2]))
                    item_cooccurrence[key] = item_cooccurrence.get(key, 0) + 1
        
        orders = list(self.df.groupby('order_id'))
        n_to_augment = int(len(orders) * probability)
        orders_to_augment = np.random.choice(len(orders), size=n_to_augment, replace=False)
        
        augmented_records = []
        for idx in orders_to_augment:
            order_id, order_data = orders[idx]
            order_data = order_data.copy()
            
            existing_items = set(order_data['product_full'].tolist())
            
            candidate_items = {}
            for item in existing_items:
                for (item1, item2), count in item_cooccurrence.items():
                    if item == item1 and item2 not in existing_items:
                        candidate_items[item2] = candidate_items.get(item2, 0) + count
                    elif item == item2 and item1 not in existing_items:
                        candidate_items[item1] = candidate_items.get(item1, 0) + count
            
            if candidate_items:
                new_item = max(candidate_items.items(), key=lambda x: x[1])[0]
                
                item_info = self.df[self.df['product_full'] == new_item].iloc[0]
                
                new_row = order_data.iloc[0].copy()
                new_row['product_full'] = new_item
                new_row['product_name'] = item_info['product_name']
                new_row['product_variant'] = item_info['product_variant']
                new_row['product_category'] = item_info['product_category']
                new_row['order_item_id'] = order_data['order_item_id'].max() + 1
                
                order_data = pd.concat([order_data, pd.DataFrame([new_row])], ignore_index=True)
            
            order_data['order_id'] = order_data['order_id'] + 30000000
            order_data['order_item_id'] = order_data['order_item_id'] + 30000000
            
            augmented_records.append(order_data)
        
        if augmented_records:
            result = pd.concat(augmented_records, ignore_index=True)
            self.logger.info(f"Generated {len(result)} records from {n_to_augment} augmented orders")
            return result
        else:
            return pd.DataFrame()
    
    def augment_synthetic_orders(self, n_synthetic: int = 200) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("AUGMENTATION 4: SYNTHETIC ORDER GENERATION")
        self.logger.info("="*80)
        self.logger.info("Creates synthetic orders based on customer clusters")
        
        customer_order_patterns = self.df.groupby('customer_id').apply(
            lambda x: x.groupby('order_id')['product_full'].apply(list).tolist()
        ).to_dict()
        
        customers = list(customer_order_patterns.keys())
        n_clusters = min(10, len(customers))
        
        customer_features = []
        for customer_id in customers:
            orders = customer_order_patterns[customer_id]
            avg_basket_size = np.mean([len(order) for order in orders])
            unique_items = len(set([item for order in orders for item in order]))
            customer_features.append([avg_basket_size, unique_items])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(customer_features)
        
        customer_clusters = {}
        for customer_id, cluster in zip(customers, clusters):
            if cluster not in customer_clusters:
                customer_clusters[cluster] = []
            customer_clusters[cluster].append(customer_id)
        
        augmented_records = []
        
        for i in range(n_synthetic):
            cluster = np.random.randint(0, n_clusters)
            if cluster not in customer_clusters:
                continue
            
            template_customer = np.random.choice(customer_clusters[cluster])
            customer_orders = self.df[self.df['customer_id'] == template_customer]
            
            template_order_id = np.random.choice(customer_orders['order_id'].unique())
            template_order = customer_orders[customer_orders['order_id'] == template_order_id].copy()
            
            synthetic_customer_id = 900000 + i
            synthetic_order_id = 40000000 + i
            
            template_order['customer_id'] = synthetic_customer_id
            template_order['order_id'] = synthetic_order_id
            template_order['order_item_id'] = template_order['order_item_id'] + synthetic_order_id
            
            augmented_records.append(template_order)
        
        if augmented_records:
            result = pd.concat(augmented_records, ignore_index=True)
            self.logger.info(f"Generated {len(result)} records from {n_synthetic} synthetic orders")
            return result
        else:
            return pd.DataFrame()
    
    def run_augmentation(self, target_multiplier: float = 1.5) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("STEP 4: ORDER-AWARE DATA AUGMENTATION")
        self.logger.info("="*80)
        self.logger.info(f"Original dataset size: {self.original_size:,}")
        self.logger.info(f"Target size: {int(self.original_size * target_multiplier):,}")
        
        augmented_datasets = [self.df]
        
        aug1 = self.augment_order_temporal_jitter(jitter_hours=2, probability=0.3)
        if len(aug1) > 0:
            augmented_datasets.append(aug1)
        
        aug2 = self.augment_basket_item_swap(probability=0.2)
        if len(aug2) > 0:
            augmented_datasets.append(aug2)
        
        aug3 = self.augment_basket_item_addition(probability=0.15)
        if len(aug3) > 0:
            augmented_datasets.append(aug3)
        
        current_size = sum(len(d) for d in augmented_datasets)
        remaining = max(0, int(self.original_size * target_multiplier) - current_size)
        
        if remaining > 0:
            avg_items_per_order = self.original_size / self.df['order_id'].nunique()
            n_synthetic = max(10, int(remaining / avg_items_per_order))
            aug4 = self.augment_synthetic_orders(n_synthetic=n_synthetic)
            if len(aug4) > 0:
                augmented_datasets.append(aug4)
        
        final_df = pd.concat(augmented_datasets, ignore_index=True)
        
        final_df = final_df.drop_duplicates(
            subset=['customer_id', 'order_id', 'product_full'],
            keep='first'
        )
        
        self.logger.info("="*80)
        self.logger.info("AUGMENTATION COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Final dataset size: {len(final_df):,}")
        self.logger.info(f"Size increase: {((len(final_df) / self.original_size) - 1) * 100:.1f}%")
        self.logger.info(f"Unique customers: {final_df['customer_id'].nunique():,}")
        self.logger.info(f"Unique orders: {final_df['order_id'].nunique():,}")
        self.logger.info(f"Average items per order: {len(final_df) / final_df['order_id'].nunique():.2f}")
        
        return final_df


if __name__ == "__main__":
    logger.info("Usage: aug = OrderAwareAugmentation(clean_df)")
    logger.info("       augmented_df = aug.run_augmentation(target_multiplier=1.5)")
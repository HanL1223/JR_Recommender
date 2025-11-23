import pandas as pd
import numpy as np
import logging
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, List
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderLevelFeatureEngineering:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logger
        
        self.user_item_matrix = None
        self.user_features = None
        self.item_features = None
        self.sequential_data = None
        
        self.logger.info("Order-Level Feature Engineering initialized")
        
    def create_user_item_matrix(self) -> 'OrderLevelFeatureEngineering':

        
        interactions = self.df.groupby(['customer_id', 'product_full']).size().reset_index(name='interaction_count')
        
        self.user_item_matrix = interactions.pivot_table(
            index='customer_id',
            columns='product_full',
            values='interaction_count',
            fill_value=0
        )
        
        sparsity = 1 - (self.user_item_matrix.astype(bool).sum().sum() / 
                        (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]))
        
        self.logger.info(f"Matrix shape: {self.user_item_matrix.shape}")
        self.logger.info(f"Sparsity: {sparsity*100:.2f}%")
        
        return self
    
    def create_user_features(self) -> 'OrderLevelFeatureEngineering':
        self.logger.info("="*80)
        self.logger.info("CREATING USER FEATURES")
        self.logger.info("="*80)
        
        user_features = self.df.groupby('customer_id').agg({
            'order_id': 'nunique',
            'order_datetime': ['min', 'max'],
            'order_total_price': ['sum', 'mean', 'std'],
            'total_orders': 'first',
            'total_lifetime_value': 'first',
            'customer_segment': 'first',
            'first_order_date': 'first',
            'last_order_date': 'first',
            'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),
            'day_of_week': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),
            'is_weekend': 'mean'
        })
        
        user_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in user_features.columns.values]
        
        user_features = user_features.rename(columns={
            'order_id_nunique': 'unique_orders',
            'order_datetime_min': 'first_order_datetime',
            'order_datetime_max': 'last_order_datetime',
            'order_total_price_sum': 'total_spent',
            'order_total_price_mean': 'avg_order_value',
            'order_total_price_std': 'order_value_std',
            'total_orders_first': 'total_orders',
            'total_lifetime_value_first': 'lifetime_value',
            'customer_segment_first': 'segment',
            'first_order_date_first': 'first_order_date',
            'last_order_date_first': 'last_order_date',
            'hour_<lambda>': 'preferred_hour',
            'day_of_week_<lambda>': 'preferred_day',
            'is_weekend_mean': 'weekend_preference'
        })
        
        user_features['customer_lifetime_days'] = (
            user_features['last_order_datetime'] - user_features['first_order_datetime']
        ).dt.days + 1
        
        user_features['order_frequency'] = user_features['unique_orders'] / user_features['customer_lifetime_days']
        user_features['order_value_std'].fillna(0, inplace=True)
        user_features['avg_items_per_order'] = self.df.groupby('customer_id').size() / user_features['unique_orders']
        user_features['unique_products'] = self.df.groupby('customer_id')['product_name'].nunique()
        user_features['unique_categories'] = self.df.groupby('customer_id')['product_category'].nunique()
        
        self.user_features = user_features
        
        self.logger.info(f"Created {len(user_features.columns)} user features")
        
        return self
    
    def create_item_features(self) -> 'OrderLevelFeatureEngineering':
        self.logger.info("="*80)
        self.logger.info("CREATING ITEM FEATURES")
        self.logger.info("="*80)
        
        item_features = self.df.groupby('product_full').agg({
            'product_name': 'first',
            'product_variant': 'first',
            'product_category': 'first',
            'order_id': 'nunique',
            'customer_id': 'nunique',
            'order_total_price': 'mean',
            'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),
            'is_weekend': 'mean'
        }).rename(columns={
            'product_name': 'name',
            'product_variant': 'variant',
            'product_category': 'category',
            'order_id': 'total_orders',
            'customer_id': 'unique_customers',
            'order_total_price': 'avg_price',
            'hour': 'popular_hour',
            'is_weekend': 'weekend_popularity'
        })
        
        total_orders = self.df['order_id'].nunique()
        item_features['popularity_score'] = item_features['total_orders'] / total_orders
        
        item_features['repeat_purchase_rate'] = (
            self.df.groupby('product_full').apply(
                lambda x: (x.groupby('customer_id').size() > 1).sum() / x['customer_id'].nunique()
            )
        )
        
        self.item_features = item_features
        
        self.logger.info(f"Created {len(item_features.columns)} item features")
        
        return self
    
    def create_order_level_sequences(self, sequence_length: int = 5, min_orders: int = 3) -> 'OrderLevelFeatureEngineering':
        self.logger.info("="*80)
        self.logger.info(f"CREATING ORDER-LEVEL SEQUENCES (sequence_length={sequence_length})")
        self.logger.info("="*80)
        self.logger.info("Each sequence represents ORDERS (visits), not individual items")
        
        sequences = []
        
        item_to_id = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
        n_items = len(item_to_id)
        
        for customer_id in self.df['customer_id'].unique():
            customer_data = self.df[self.df['customer_id'] == customer_id].sort_values('order_datetime')
            
            customer_orders = customer_data.groupby('order_id').agg({
                'product_full': lambda x: list(x),
                'order_datetime': 'first',
                'hour': 'first',
                'day_of_week': 'first',
                'is_weekend': 'first',
                'order_total_price': 'first'
            }).sort_values('order_datetime')
            
            if len(customer_orders) < min_orders + 1:
                continue
            
            for i in range(sequence_length, len(customer_orders)):
                previous_orders = customer_orders.iloc[i-sequence_length:i]
                target_order = customer_orders.iloc[i]
                
                order_sequences = []
                for _, order in previous_orders.iterrows():
                    order_vector = np.zeros(n_items, dtype=np.float32)
                    for item in order['product_full']:
                        item_id = item_to_id.get(item, 0)
                        order_vector[item_id] = 1
                    order_sequences.append(order_vector)
                
                order_sequence_matrix = np.array(order_sequences)
                
                target_items = target_order['product_full']
                for target_item in target_items:
                    target_item_id = item_to_id.get(target_item, 0)
                    
                    sequences.append({
                        'customer_id': customer_id,
                        'order_sequence': order_sequence_matrix,
                        'target_item_id': target_item_id,
                        'target_item': target_item,
                        'target_order_items': target_items,
                        'n_previous_orders': len(previous_orders),
                        'hour': target_order['hour'],
                        'day_of_week': target_order['day_of_week'],
                        'is_weekend': target_order['is_weekend'],
                        'timestamp': target_order['order_datetime'],
                        'order_value': target_order['order_total_price']
                    })
        
        self.sequential_data = pd.DataFrame(sequences)
        
        self.logger.info(f"Created {len(self.sequential_data)} training samples")
        self.logger.info(f"Each sample: {sequence_length} previous orders -> 1 target item")
        self.logger.info(f"Unique customers: {self.sequential_data['customer_id'].nunique()}")
        self.logger.info(f"Unique target items: {self.sequential_data['target_item_id'].nunique()}")
        
        sample = self.sequential_data.iloc[0]
        self.logger.info(f"\nExample sequence shape: {sample['order_sequence'].shape}")
        self.logger.info(f"  (sequence_length={sequence_length}, n_items={n_items})")
        self.logger.info(f"Example target order: {sample['target_order_items']}")
        
        return self
    
    def create_basket_level_sequences(self, sequence_length: int = 5, min_orders: int = 3) -> 'OrderLevelFeatureEngineering':
        self.logger.info("="*80)
        self.logger.info(f"CREATING BASKET-LEVEL SEQUENCES (ALTERNATIVE APPROACH)")
        self.logger.info("="*80)
        self.logger.info("Predicts entire next basket based on previous baskets")
        
        sequences = []
        
        item_to_id = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
        n_items = len(item_to_id)
        
        for customer_id in self.df['customer_id'].unique():
            customer_data = self.df[self.df['customer_id'] == customer_id].sort_values('order_datetime')
            
            customer_orders = customer_data.groupby('order_id').agg({
                'product_full': lambda x: list(x),
                'order_datetime': 'first',
                'hour': 'first',
                'day_of_week': 'first',
                'is_weekend': 'first',
                'order_total_price': 'first'
            }).sort_values('order_datetime')
            
            if len(customer_orders) < min_orders + 1:
                continue
            
            for i in range(sequence_length, len(customer_orders)):
                previous_orders = customer_orders.iloc[i-sequence_length:i]
                target_order = customer_orders.iloc[i]
                
                order_sequences = []
                for _, order in previous_orders.iterrows():
                    order_vector = np.zeros(n_items, dtype=np.float32)
                    for item in order['product_full']:
                        item_id = item_to_id.get(item, 0)
                        order_vector[item_id] = 1
                    order_sequences.append(order_vector)
                
                order_sequence_matrix = np.array(order_sequences)
                
                target_basket = np.zeros(n_items, dtype=np.float32)
                for item in target_order['product_full']:
                    item_id = item_to_id.get(item, 0)
                    target_basket[item_id] = 1
                
                sequences.append({
                    'customer_id': customer_id,
                    'order_sequence': order_sequence_matrix,
                    'target_basket': target_basket,
                    'target_items': target_order['product_full'],
                    'n_previous_orders': len(previous_orders),
                    'hour': target_order['hour'],
                    'day_of_week': target_order['day_of_week'],
                    'is_weekend': target_order['is_weekend'],
                    'timestamp': target_order['order_datetime'],
                    'order_value': target_order['order_total_price']
                })
        
        self.basket_sequential_data = pd.DataFrame(sequences)
        
        self.logger.info(f"Created {len(self.basket_sequential_data)} basket prediction samples")
        self.logger.info(f"Each sample: {sequence_length} previous baskets -> 1 target basket")
        self.logger.info(f"Unique customers: {self.basket_sequential_data['customer_id'].nunique()}")
        
        return self
    
    def temporal_train_val_test_split(self, 
                                       train_ratio: float = 0.7, 
                                       val_ratio: float = 0.15,
                                       use_basket_mode: bool = False) -> 'OrderLevelFeatureEngineering':
        self.logger.info("="*80)
        self.logger.info("CREATING TEMPORAL TRAIN/VAL/TEST SPLIT")
        self.logger.info("="*80)
        
        data_to_split = self.basket_sequential_data if use_basket_mode else self.sequential_data
        
        data_to_split = data_to_split.sort_values('timestamp')
        
        n = len(data_to_split)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        self.train_data = data_to_split.iloc[:train_end].reset_index(drop=True)
        self.val_data = data_to_split.iloc[train_end:val_end].reset_index(drop=True)
        self.test_data = data_to_split.iloc[val_end:].reset_index(drop=True)
        
        self.logger.info(f"Train set: {len(self.train_data)} ({len(self.train_data)/n*100:.1f}%)")
        self.logger.info(f"Val set: {len(self.val_data)} ({len(self.val_data)/n*100:.1f}%)")
        self.logger.info(f"Test set: {len(self.test_data)} ({len(self.test_data)/n*100:.1f}%)")
        
        self.logger.info(f"\nTrain date range: {self.train_data['timestamp'].min()} to {self.train_data['timestamp'].max()}")
        self.logger.info(f"Val date range: {self.val_data['timestamp'].min()} to {self.val_data['timestamp'].max()}")
        self.logger.info(f"Test date range: {self.test_data['timestamp'].min()} to {self.test_data['timestamp'].max()}")
        
        return self
    
    def prepare_arrays_for_item_prediction(self) -> 'OrderLevelFeatureEngineering':
        self.logger.info("="*80)
        self.logger.info("PREPARING ARRAYS FOR ITEM-LEVEL PREDICTION")
        self.logger.info("="*80)
        
        def to_arrays(data):
            X_seq = np.array([seq for seq in data['order_sequence'].values])
            X_context = data[['hour', 'day_of_week', 'is_weekend']].values
            y = data['target_item_id'].values
            customers = data['customer_id'].values
            return X_seq, X_context, y, customers
        
        self.X_train_seq, self.X_train_context, self.y_train, self.customers_train = to_arrays(self.train_data)
        self.X_val_seq, self.X_val_context, self.y_val, self.customers_val = to_arrays(self.val_data)
        self.X_test_seq, self.X_test_context, self.y_test, self.customers_test = to_arrays(self.test_data)
        
        self.logger.info(f"X_train_seq shape: {self.X_train_seq.shape}")
        self.logger.info(f"  (n_samples, sequence_length, n_items)")
        self.logger.info(f"X_train_context shape: {self.X_train_context.shape}")
        self.logger.info(f"y_train shape: {self.y_train.shape}")
        
        return self
    
    def prepare_arrays_for_basket_prediction(self) -> 'OrderLevelFeatureEngineering':
        self.logger.info("="*80)
        self.logger.info("PREPARING ARRAYS FOR BASKET-LEVEL PREDICTION")
        self.logger.info("="*80)
        
        def to_arrays(data):
            X_seq = np.array([seq for seq in data['order_sequence'].values])
            X_context = data[['hour', 'day_of_week', 'is_weekend']].values
            y_basket = np.array([basket for basket in data['target_basket'].values])
            customers = data['customer_id'].values
            return X_seq, X_context, y_basket, customers
        
        self.X_train_seq, self.X_train_context, self.y_train_basket, self.customers_train = to_arrays(self.train_data)
        self.X_val_seq, self.X_val_context, self.y_val_basket, self.customers_val = to_arrays(self.val_data)
        self.X_test_seq, self.X_test_context, self.y_test_basket, self.customers_test = to_arrays(self.test_data)
        
        self.logger.info(f"X_train_seq shape: {self.X_train_seq.shape}")
        self.logger.info(f"  (n_samples, sequence_length, n_items)")
        self.logger.info(f"y_train_basket shape: {self.y_train_basket.shape}")
        self.logger.info(f"  (n_samples, n_items) - multi-label classification")
        
        return self
    
    def get_feature_metadata(self) -> Dict:
        metadata = {
            'n_users': len(self.user_item_matrix.index),
            'n_items': len(self.user_item_matrix.columns),
            'sequence_length': self.X_train_seq.shape[1] if hasattr(self, 'X_train_seq') else 0,
            'n_context_features': self.X_train_context.shape[1] if hasattr(self, 'X_train_context') else 0,
            'item_to_id': {item: idx for idx, item in enumerate(self.user_item_matrix.columns)},
            'id_to_item': {idx: item for idx, item in enumerate(self.user_item_matrix.columns)}
        }
        return metadata
    
    def run_feature_engineering(self, 
                                sequence_length: int = 5,
                                prediction_mode: str = 'item') -> Tuple[Dict, Dict]:
        self.logger.info("="*80)
        self.logger.info("STEP 3: ORDER-LEVEL FEATURE ENGINEERING")
        self.logger.info("="*80)
        self.logger.info(f"Prediction mode: {prediction_mode}")
        self.logger.info(f"  'item' = Predict next items one-by-one")
        self.logger.info(f"  'basket' = Predict entire next order basket")
        
        self.create_user_item_matrix()
        self.create_user_features()
        self.create_item_features()
        
        if prediction_mode == 'basket':
            self.create_basket_level_sequences(sequence_length=sequence_length)
            self.temporal_train_val_test_split(use_basket_mode=True)
            self.prepare_arrays_for_basket_prediction()
        else:
            self.create_order_level_sequences(sequence_length=sequence_length)
            self.temporal_train_val_test_split(use_basket_mode=False)
            self.prepare_arrays_for_item_prediction()
        
        metadata = self.get_feature_metadata()
        
        arrays = {
            'X_train_seq': self.X_train_seq,
            'X_train_context': self.X_train_context,
            'y_train': getattr(self, 'y_train_basket', None) or self.y_train,
            'X_val_seq': self.X_val_seq,
            'X_val_context': self.X_val_context,
            'y_val': getattr(self, 'y_val_basket', None) or self.y_val,
            'X_test_seq': self.X_test_seq,
            'X_test_context': self.X_test_context,
            'y_test': getattr(self, 'y_test_basket', None) or self.y_test,
            'prediction_mode': prediction_mode
        }
        
        self.logger.info("="*80)
        self.logger.info("STEP 3 COMPLETE")
        self.logger.info("="*80)
        self.logger.info("Order-level sequences ready for model training")
        
        return arrays, metadata


if __name__ == "__main__":
    logger.info("Usage:")
    logger.info("  fe = OrderLevelFeatureEngineering(clean_df)")
    logger.info("  arrays, metadata = fe.run_feature_engineering(sequence_length=5, prediction_mode='item')")
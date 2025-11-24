import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List
from datetime import timedelta
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AugmentationStrategy(ABC):
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass
    
    def get_strategy_info(self) -> dict:
        return {
            'name': self.strategy_name,
            'config': self.config
        }


class TemporalJitterStrategy(AugmentationStrategy):
    
    def __init__(self, jitter_hours: int = 2, probability: float = 0.3):
        super().__init__({'jitter_hours': jitter_hours, 'probability': probability})
        self.jitter_hours = jitter_hours
        self.probability = probability
        
    @property
    def strategy_name(self) -> str:
        return "temporal_jitter"
    
    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Applying {self.strategy_name} (jitter={self.jitter_hours}h, prob={self.probability})")
        
        sample_size = int(len(df) * self.probability)
        sampled = df.sample(n=sample_size, random_state=42).copy()
        
        jitter_minutes = np.random.randint(-self.jitter_hours*60, self.jitter_hours*60, size=len(sampled))
        sampled['order_datetime'] = sampled['order_datetime'] + pd.to_timedelta(jitter_minutes, unit='m')
        
        sampled['order_date'] = sampled['order_datetime'].dt.date.astype(str)
        sampled['order_time'] = sampled['order_datetime'].dt.time.astype(str)
        sampled['hour'] = sampled['order_datetime'].dt.hour
        sampled['day_of_week'] = sampled['order_datetime'].dt.dayofweek
        sampled['is_weekend'] = sampled['day_of_week'].isin([5, 6]).astype(int)
        
        sampled['order_id'] = sampled['order_id'] + 10000000
        sampled['order_item_id'] = sampled['order_item_id'] + 10000000
        
        self.logger.info(f"Generated {len(sampled)} augmented records")
        return sampled


class VariantSubstitutionStrategy(AugmentationStrategy):
    
    def __init__(self, probability: float = 0.2):
        super().__init__({'probability': probability})
        self.probability = probability
        
    @property
    def strategy_name(self) -> str:
        return "variant_substitution"
    
    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Applying {self.strategy_name} (prob={self.probability})")
        
        variant_groups = df.groupby('product_name')['product_variant'].unique().to_dict()
        variant_groups = {k: v for k, v in variant_groups.items() if len(v) > 1}
        
        eligible = df[df['product_name'].isin(variant_groups.keys())].copy()
        sample_size = int(len(eligible) * self.probability)
        sampled = eligible.sample(n=sample_size, random_state=43).copy()
        
        for idx in sampled.index:
            product_name = sampled.loc[idx, 'product_name']
            current_variant = sampled.loc[idx, 'product_variant']
            possible_variants = variant_groups[product_name]
            other_variants = [v for v in possible_variants if v != current_variant]
            
            if len(other_variants) > 0:
                new_variant = np.random.choice(other_variants)
                sampled.loc[idx, 'product_variant'] = new_variant
                sampled.loc[idx, 'product_full'] = f"{product_name}_{new_variant}"
        
        sampled['order_id'] = sampled['order_id'] + 20000000
        sampled['order_item_id'] = sampled['order_item_id'] + 20000000
        
        self.logger.info(f"Generated {len(sampled)} augmented records")
        return sampled


class SequenceShuffleStrategy(AugmentationStrategy):
    
    def __init__(self, probability: float = 0.25):
        super().__init__({'probability': probability})
        self.probability = probability
        
    @property
    def strategy_name(self) -> str:
        return "sequence_shuffle"
    
    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Applying {self.strategy_name} (prob={self.probability})")
        
        orders = df.groupby('order_id')
        eligible_orders = [order_id for order_id, group in orders if len(group) > 1]
        
        n_to_shuffle = int(len(eligible_orders) * self.probability)
        orders_to_shuffle = np.random.choice(eligible_orders, size=n_to_shuffle, replace=False)
        
        augmented_records = []
        for order_id in orders_to_shuffle:
            order_data = df[df['order_id'] == order_id].copy()
            
            products = order_data['product_name'].values.copy()
            variants = order_data['product_variant'].values.copy()
            
            shuffle_idx = np.random.permutation(len(products))
            order_data['product_name'] = products[shuffle_idx]
            order_data['product_variant'] = variants[shuffle_idx]
            order_data['product_full'] = order_data['product_name'] + '_' + order_data['product_variant']
            
            order_data['order_id'] = order_data['order_id'] + 30000000
            order_data['order_item_id'] = order_data['order_item_id'] + 30000000
            
            augmented_records.append(order_data)
        
        if augmented_records:
            result = pd.concat(augmented_records, ignore_index=True)
            self.logger.info(f"Generated {len(result)} augmented records")
            return result
        else:
            self.logger.info("No records generated")
            return pd.DataFrame()


class SyntheticCustomerStrategy(AugmentationStrategy):
    
    def __init__(self, n_clusters: int = 10, n_synthetic: int = 200):
        super().__init__({'n_clusters': n_clusters, 'n_synthetic': n_synthetic})
        self.n_clusters = n_clusters
        self.n_synthetic = n_synthetic
        
    @property
    def strategy_name(self) -> str:
        return "synthetic_customers"
    
    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Applying {self.strategy_name} (clusters={self.n_clusters}, synthetic={self.n_synthetic})")
        
        customer_features = df.groupby('customer_id').agg({
            'product_category': lambda x: x.value_counts().to_dict(),
            'order_id': 'nunique',
            'total_lifetime_value': 'first',
            'customer_segment': 'first'
        })
        
        category_dummies = pd.get_dummies(
            df.groupby('customer_id')['product_category'].apply(list).apply(pd.Series).stack()
        ).groupby(level=0).sum()
        
        X = category_dummies.values
        
        n_clusters = min(self.n_clusters, len(customer_features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        customer_features['cluster'] = kmeans.fit_predict(X)
        
        synthetic_records = []
        
        for i in range(self.n_synthetic):
            cluster = np.random.randint(0, n_clusters)
            cluster_customers = customer_features[customer_features['cluster'] == cluster].index
            
            if len(cluster_customers) == 0:
                continue
            
            template_customer = np.random.choice(cluster_customers)
            template_orders = df[df['customer_id'] == template_customer].copy()
            
            n_orders_to_copy = min(3, len(template_orders))
            sampled_orders = template_orders.sample(n=n_orders_to_copy, replace=False)
            
            synthetic_customer_id = 900000 + i
            sampled_orders['customer_id'] = synthetic_customer_id
            sampled_orders['order_id'] = sampled_orders['order_id'] + 40000000 + i * 1000
            sampled_orders['order_item_id'] = sampled_orders['order_item_id'] + 40000000 + i * 1000
            
            synthetic_records.append(sampled_orders)
        
        if synthetic_records:
            result = pd.concat(synthetic_records, ignore_index=True)
            self.logger.info(f"Generated {len(result)} records from {self.n_synthetic} synthetic customers")
            return result
        else:
            self.logger.info("No synthetic customers generated")
            return pd.DataFrame()


class DataAugmentationPipeline:
    
    def __init__(self, strategies: List[AugmentationStrategy] = None):
        self.strategies = strategies or []
        self.logger = logger
        self.original_size = 0
        
    def add_strategy(self, strategy: AugmentationStrategy):
        self.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.strategy_name}")
        
    def remove_strategy(self, strategy_name: str):
        self.strategies = [s for s in self.strategies if s.strategy_name != strategy_name]
        self.logger.info(f"Removed strategy: {strategy_name}")
        
    def clear_strategies(self):
        self.strategies = []
        self.logger.info("Cleared all strategies")
        
    def list_strategies(self) -> List[str]:
        return [s.strategy_name for s in self.strategies]
    
    def run_augmentation(self, df: pd.DataFrame, target_multiplier: float = 1.5) -> pd.DataFrame:
        self.logger.info("="*80)
        self.logger.info("STEP 4: DATA AUGMENTATION WITH STRATEGY PATTERN")
        self.logger.info("="*80)
        
        self.original_size = len(df)
        target_size = int(self.original_size * target_multiplier)
        
        self.logger.info(f"Original size: {self.original_size:,}")
        self.logger.info(f"Target size: {target_size:,}")
        self.logger.info(f"Active strategies: {self.list_strategies()}")
        
        augmented_datasets = [df]
        
        for strategy in self.strategies:
            augmented = strategy.augment(df)
            if len(augmented) > 0:
                augmented_datasets.append(augmented)
        
        final_df = pd.concat(augmented_datasets, ignore_index=True)
        
        final_df = final_df.drop_duplicates(
            subset=['customer_id', 'product_name', 'product_variant', 'order_date', 'order_time'],
            keep='first'
        )
        
        self.logger.info("="*80)
        self.logger.info("AUGMENTATION COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Final size: {len(final_df):,}")
        self.logger.info(f"Size increase: {((len(final_df) / self.original_size) - 1) * 100:.1f}%")
        self.logger.info(f"Unique customers: {final_df['customer_id'].nunique():,}")
        self.logger.info(f"Unique products: {final_df['product_name'].nunique()}")
        
        return final_df


class AugmentationFactory:
    
    @staticmethod
    def create_standard_pipeline(target_multiplier: float = 1.5) -> DataAugmentationPipeline:
        pipeline = DataAugmentationPipeline([
            TemporalJitterStrategy(jitter_hours=2, probability=0.3),
            VariantSubstitutionStrategy(probability=0.2),
            SequenceShuffleStrategy(probability=0.25)
        ])
        
        logger.info("Created standard augmentation pipeline")
        return pipeline
    
    @staticmethod
    def create_aggressive_pipeline() -> DataAugmentationPipeline:
        pipeline = DataAugmentationPipeline([
            TemporalJitterStrategy(jitter_hours=3, probability=0.5),
            VariantSubstitutionStrategy(probability=0.4),
            SequenceShuffleStrategy(probability=0.4),
            SyntheticCustomerStrategy(n_clusters=15, n_synthetic=500)
        ])
        
        logger.info("Created aggressive augmentation pipeline")
        return pipeline
    
    @staticmethod
    def create_conservative_pipeline() -> DataAugmentationPipeline:
        pipeline = DataAugmentationPipeline([
            TemporalJitterStrategy(jitter_hours=1, probability=0.2),
            VariantSubstitutionStrategy(probability=0.1)
        ])
        
        logger.info("Created conservative augmentation pipeline")
        return pipeline
    
    @staticmethod
    def create_custom_pipeline(strategy_configs: List[dict]) -> DataAugmentationPipeline:
        pipeline = DataAugmentationPipeline()
        
        for config in strategy_configs:
            strategy_type = config.pop('type')
            
            if strategy_type == 'temporal_jitter':
                pipeline.add_strategy(TemporalJitterStrategy(**config))
            elif strategy_type == 'variant_substitution':
                pipeline.add_strategy(VariantSubstitutionStrategy(**config))
            elif strategy_type == 'sequence_shuffle':
                pipeline.add_strategy(SequenceShuffleStrategy(**config))
            elif strategy_type == 'synthetic_customers':
                pipeline.add_strategy(SyntheticCustomerStrategy(**config))
        
        logger.info("Created custom augmentation pipeline")
        return pipeline


if __name__ == "__main__":
    logger.info("REFACTORED: Data Augmentation with Strategy Pattern")
    logger.info("")
    logger.info("Usage Example 1 - Standard Pipeline:")
    logger.info("  pipeline = AugmentationFactory.create_standard_pipeline()")
    logger.info("  augmented_df = pipeline.run_augmentation(df)")
    logger.info("")
    logger.info("Usage Example 2 - Custom Pipeline:")
    logger.info("  pipeline = DataAugmentationPipeline()")
    logger.info("  pipeline.add_strategy(TemporalJitterStrategy(jitter_hours=2))")
    logger.info("  pipeline.add_strategy(VariantSubstitutionStrategy(probability=0.3))")
    logger.info("  augmented_df = pipeline.run_augmentation(df)")
    logger.info("")
    logger.info("Usage Example 3 - Dynamic Configuration:")
    logger.info("  configs = [")
    logger.info("      {'type': 'temporal_jitter', 'jitter_hours': 2, 'probability': 0.3},")
    logger.info("      {'type': 'variant_substitution', 'probability': 0.2}")
    logger.info("  ]")
    logger.info("  pipeline = AugmentationFactory.create_custom_pipeline(configs)")
    logger.info("  augmented_df = pipeline.run_augmentation(df)")
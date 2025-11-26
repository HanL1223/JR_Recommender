"""
Step F: ML Models with Hyperparameter Tuning
=============================================
Trains LightGBM and XGBoost with Optuna hyperparameter optimization.
Proper Learning-to-Rank with verbose logging to verify training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

# Optional imports
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed - pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed - pip install xgboost")

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed - pip install optuna")


class BaseRankingModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names):
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass


class LightGBMRanker(BaseRankingModel):
    """LightGBM with LambdaRank for Learning-to-Rank."""
    
    def __init__(self, params: Dict = None, random_seed: int = 42):
        self.random_seed = random_seed
        self.params = params or {}
        self.model = None
        self.feature_names = []
        self.best_score = 0.0
    
    def fit(self, X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names):
        if not HAS_LIGHTGBM:
            return self
        
        self.feature_names = feature_names
        
        default_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3, 5, 10],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': self.random_seed,
            'force_row_wise': True
        }
        default_params.update(self.params)
        
        train_data = lgb.Dataset(X_train, label=y_train, group=groups_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)
        
        eval_results = {}
        
        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
                lgb.record_evaluation(eval_results)
            ]
        )
        
        self.best_score = self.model.best_score.get('valid', {}).get('ndcg@3', 0)
        return self
    
    def predict(self, X) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importance(importance_type='gain')
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'pct': 100 * importance / importance.sum()
        })
        return df.sort_values('importance', ascending=False)


class XGBoostRanker(BaseRankingModel):
    """XGBoost with rank:pairwise for Learning-to-Rank."""
    
    def __init__(self, params: Dict = None, random_seed: int = 42):
        self.random_seed = random_seed
        self.params = params or {}
        self.model = None
        self.feature_names = []
        self.best_score = 0.0
    
    def fit(self, X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names):
        if not HAS_XGBOOST:
            return self
        
        self.feature_names = feature_names
        
        default_params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg@3',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': self.random_seed,
            'verbosity': 1
        }
        default_params.update(self.params)
        
        # Convert groups to proper format for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(groups_train)
        
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(groups_val)
        
        evals_result = {}
        
        self.model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=50,
            evals_result=evals_result
        )
        
        self.best_score = evals_result.get('valid', {}).get('ndcg@3', [0])[-1]
        return self
    
    def predict(self, X) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X))
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.get_score(importance_type='gain')
        # Map back to feature names
        data = []
        for i, name in enumerate(self.feature_names):
            key = f'f{i}'
            data.append({
                'feature': name,
                'importance': importance.get(key, 0)
            })
        df = pd.DataFrame(data)
        df['pct'] = 100 * df['importance'] / df['importance'].sum()
        return df.sort_values('importance', ascending=False)


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning."""
    
    def __init__(self, n_trials: int = 30, timeout: int = 600, random_seed: int = 42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_seed = random_seed
    
    def tune_lightgbm(self, X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names) -> Dict:
        if not HAS_OPTUNA or not HAS_LIGHTGBM:
            return {}
        
        logger.info(f"\nTuning LightGBM ({self.n_trials} trials, {self.timeout}s timeout)...")
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            
            model = LightGBMRanker(params=params, random_seed=self.random_seed)
            model.fit(X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names)
            return model.best_score
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        logger.info(f"Best LightGBM NDCG@3: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def tune_xgboost(self, X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names) -> Dict:
        if not HAS_OPTUNA or not HAS_XGBOOST:
            return {}
        
        logger.info(f"\nTuning XGBoost ({self.n_trials} trials, {self.timeout}s timeout)...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            model = XGBoostRanker(params=params, random_seed=self.random_seed)
            model.fit(X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names)
            return model.best_score
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        logger.info(f"Best XGBoost NDCG@3: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params


@dataclass
class MLModels:
    """Container for ML models."""
    lightgbm: Optional[LightGBMRanker]
    xgboost: Optional[XGBoostRanker]
    best_model: Optional[BaseRankingModel]
    best_model_name: str
    best_params: Dict
    feature_importance: pd.DataFrame
    ndcg_score: float
    training_time: float


class MLTrainer:
    """Trains ML models with hyperparameter tuning and model selection."""
    
    def __init__(self, n_trials: int = 20, timeout: int = 300, random_seed: int = 42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_seed = random_seed
    
    def _prepare_groups(self, df: pd.DataFrame) -> np.ndarray:
        """Create group sizes for ranking (customer+order_idx = one query)."""
        df_sorted = df.sort_values(['customer_id', 'order_idx'])
        group_sizes = df_sorted.groupby(['customer_id', 'order_idx']).size().values
        return group_sizes
    
    def run(self, split_data, do_tuning: bool = True) -> MLModels:
        logger.info("=" * 50)
        logger.info("STEP F: ML Models (Learning to Rank)")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Prepare data
        train_df = split_data.train_df.sort_values(['customer_id', 'order_idx'])
        test_df = split_data.test_df.sort_values(['customer_id', 'order_idx'])
        
        feature_names = split_data.feature_names
        
        X_train = train_df[feature_names].values.astype(np.float32)
        y_train = train_df['label'].values.astype(np.float32)
        groups_train = self._prepare_groups(train_df)
        
        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df['label'].values.astype(np.float32)
        groups_test = self._prepare_groups(test_df)
        
        logger.info(f"\nData prepared:")
        logger.info(f"  Train: {len(X_train):,} samples, {len(groups_train):,} queries")
        logger.info(f"  Test: {len(X_test):,} samples, {len(groups_test):,} queries")
        logger.info(f"  Features: {len(feature_names)}")
        
        # Hyperparameter tuning
        lgb_params = {}
        xgb_params = {}
        
        if do_tuning:
            tuner = HyperparameterTuner(
                n_trials=self.n_trials,
                timeout=self.timeout,
                random_seed=self.random_seed
            )
            
            if HAS_LIGHTGBM:
                lgb_params = tuner.tune_lightgbm(
                    X_train, y_train, groups_train,
                    X_test, y_test, groups_test,
                    feature_names
                )
            
            if HAS_XGBOOST:
                xgb_params = tuner.tune_xgboost(
                    X_train, y_train, groups_train,
                    X_test, y_test, groups_test,
                    feature_names
                )
        
        # Train final models with best params
        logger.info("\n" + "=" * 40)
        logger.info("Training final models with best parameters...")
        logger.info("=" * 40)
        
        lgb_model = None
        xgb_model = None
        lgb_score = 0.0
        xgb_score = 0.0
        
        if HAS_LIGHTGBM:
            logger.info("\nTraining LightGBM...")
            lgb_model = LightGBMRanker(params=lgb_params, random_seed=self.random_seed)
            lgb_model.fit(X_train, y_train, groups_train, X_test, y_test, groups_test, feature_names)
            lgb_score = lgb_model.best_score
            logger.info(f"LightGBM final NDCG@3: {lgb_score:.4f}")
        
        if HAS_XGBOOST:
            logger.info("\nTraining XGBoost...")
            xgb_model = XGBoostRanker(params=xgb_params, random_seed=self.random_seed)
            xgb_model.fit(X_train, y_train, groups_train, X_test, y_test, groups_test, feature_names)
            xgb_score = xgb_model.best_score
            logger.info(f"XGBoost final NDCG@3: {xgb_score:.4f}")
        
        # Model selection
        if lgb_score >= xgb_score and lgb_model is not None:
            best_model = lgb_model
            best_model_name = "LightGBM"
            best_params = lgb_params
            best_score = lgb_score
        elif xgb_model is not None:
            best_model = xgb_model
            best_model_name = "XGBoost"
            best_params = xgb_params
            best_score = xgb_score
        else:
            best_model = None
            best_model_name = "None"
            best_params = {}
            best_score = 0.0
        
        feature_importance = best_model.get_feature_importance() if best_model else pd.DataFrame()
        
        training_time = time.time() - start_time
        
        logger.info("\n" + "=" * 40)
        logger.info("MODEL SELECTION RESULTS")
        logger.info("=" * 40)
        logger.info(f"LightGBM NDCG@3: {lgb_score:.4f}")
        logger.info(f"XGBoost NDCG@3: {xgb_score:.4f}")
        logger.info(f"*** BEST MODEL: {best_model_name} (NDCG@3: {best_score:.4f}) ***")
        logger.info(f"Training time: {training_time:.1f} seconds")
        
        if not feature_importance.empty:
            logger.info(f"\nFeature Importance:\n{feature_importance}")
        
        return MLModels(
            lightgbm=lgb_model,
            xgboost=xgb_model,
            best_model=best_model,
            best_model_name=best_model_name,
            best_params=best_params,
            feature_importance=feature_importance,
            ndcg_score=best_score,
            training_time=training_time
        )


def create_ml_trainer(n_trials: int = 20, timeout: int = 300) -> MLTrainer:
    return MLTrainer(n_trials=n_trials, timeout=timeout)


if __name__ == "__main__":
    from A_data_preparation import create_data_preparation
    from B_feature_engineering import create_feature_engineering
    from C_training_data import create_training_data_builder
    from D_train_test_split import create_data_splitter
    
    data = create_data_preparation().run("data/data_raw.csv")
    features = create_feature_engineering().run(data)
    training = create_training_data_builder().run(data, features)
    split = create_data_splitter().run(training)
    
    models = create_ml_trainer(n_trials=10, timeout=120).run(split, do_tuning=True)
    logger.info(f"\nBest: {models.best_model_name} with NDCG@3={models.ndcg_score:.4f}")
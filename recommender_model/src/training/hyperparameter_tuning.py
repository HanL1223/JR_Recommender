"""
Hyperparameter Tuning
=====================
Optuna-based hyperparameter optimization.

Refactored from: F_ml_models.py
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Check if Optuna is available
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed - pip install optuna")


@dataclass
class TuningResult:
    """Container for tuning results."""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study: Any = None


class HyperparameterTuner:
    """
    Hyperparameter tuner using Optuna.
    
    Supports:
    - LightGBM parameter optimization
    - Early stopping
    - Timeout limits
    
    Example:
        >>> tuner = HyperparameterTuner(n_trials=50, timeout=600)
        >>> result = tuner.tune_lightgbm(split_data)
        >>> print(f"Best NDCG@3: {result.best_score:.4f}")
        >>> print(f"Best params: {result.best_params}")
    """
    
    def __init__(
        self,
        n_trials: int = 20,
        timeout: int = 300,
        metric: str = 'ndcg@3',
        direction: str = 'maximize',
        random_seed: int = 42
    ):
        """
        Initialize tuner.
        
        Args:
            n_trials: Maximum number of trials
            timeout: Timeout in seconds
            metric: Metric to optimize
            direction: 'maximize' or 'minimize'
            random_seed: For reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.direction = direction
        self.random_seed = random_seed
        
        logger.info(f"HyperparameterTuner initialized (n_trials={n_trials}, timeout={timeout}s)")
    
    def tune_lightgbm(
        self,
        split_data,
        fixed_params: Dict[str, Any] = None
    ) -> TuningResult:
        """
        Tune LightGBM hyperparameters.
        
        Args:
            split_data: SplitData from splitter
            fixed_params: Parameters to keep fixed
            
        Returns:
            TuningResult with best parameters
        """
        logger.info("Starting LightGBM hyperparameter tuning...")
        
        from src.models.lightgbm_ranker import LightGBMRanker
        from src.evaluation.metrics import RankingMetrics
        
        evaluator = RankingMetrics()
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            
            # Merge with fixed params
            if fixed_params:
                params.update(fixed_params)
            
            # Train model
            model = LightGBMRanker.from_params(params)
            model.fit(
                train_df=split_data.train_df,
                feature_names=split_data.feature_names,
                valid_df=split_data.test_df
            )
            
            # Evaluate
            metrics = evaluator.evaluate(model, split_data.test_df, split_data.feature_names)
            score = metrics.get(self.metric, 0)
            
            logger.info(f"Trial {trial.number}: {self.metric}={score:.4f}")
            
            return score
        
        # Create study
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name="lightgbm_tuning"
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Tuning complete. Best {self.metric}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study
        )
    
    def tune_xgboost(
        self,
        split_data,
        fixed_params: Dict[str, Any] = None
    ) -> TuningResult:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            split_data: SplitData from splitter
            fixed_params: Parameters to keep fixed
            
        Returns:
            TuningResult with best parameters
        """
        logger.info("Starting XGBoost hyperparameter tuning...")
        
        from src.models.xgboost_ranker import XGBoostRanker
        from src.evaluation.metrics import RankingMetrics
        
        evaluator = RankingMetrics()
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            if fixed_params:
                params.update(fixed_params)
            
            model = XGBoostRanker.from_params(params)
            model.fit(
                train_df=split_data.train_df,
                feature_names=split_data.feature_names,
                valid_df=split_data.test_df
            )
            
            metrics = evaluator.evaluate(model, split_data.test_df, split_data.feature_names)
            score = metrics.get(self.metric, 0)
            
            logger.info(f"Trial {trial.number}: {self.metric}={score:.4f}")
            
            return score
        
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name="xgboost_tuning"
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Tuning complete. Best {self.metric}: {study.best_value:.4f}")
        
        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study
        )
    
    def tune_catboost(
        self,
        split_data,
        fixed_params: Dict[str, Any] = None
    ) -> TuningResult:
        """
        Tune CatBoost hyperparameters.
        
        Args:
            split_data: SplitData from splitter
            fixed_params: Parameters to keep fixed
            
        Returns:
            TuningResult with best parameters
        """
        logger.info("Starting CatBoost hyperparameter tuning...")
        
        from src.models.catboost_ranker import CatBoostRanker
        from src.evaluation.metrics import RankingMetrics
        
        evaluator = RankingMetrics()
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            }
            
            if fixed_params:
                params.update(fixed_params)
            
            model = CatBoostRanker.from_params(params)
            model.fit(
                train_df=split_data.train_df,
                feature_names=split_data.feature_names,
                valid_df=split_data.test_df
            )
            
            metrics = evaluator.evaluate(model, split_data.test_df, split_data.feature_names)
            score = metrics.get(self.metric, 0)
            
            logger.info(f"Trial {trial.number}: {self.metric}={score:.4f}")
            
            return score
        
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name="catboost_tuning"
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Tuning complete. Best {self.metric}: {study.best_value:.4f}")
        
        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study
        )
    
    def tune_custom(
        self,
        objective_fn: Callable,
        param_space: Dict[str, Any]
    ) -> TuningResult:
        """
        Tune with custom objective function.
        
        Args:
            objective_fn: Function that takes trial and returns score
            param_space: Not used (defined in objective_fn)
            
        Returns:
            TuningResult
        """
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler
        )
        
        study.optimize(
            objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study
        )
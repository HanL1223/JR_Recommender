"""
Hyperparameter Tuner using Optuna
================================
Supports:
- LightGBMRanker
- XGBoostRanker

Trainer calls:
    tuner.tune_lightgbm(split)
    tuner.tune_xgboost(split)

So we import model classes internally instead of requiring callers to pass them.
"""

import optuna
import logging
from typing import Dict, Any
from dataclasses import dataclass

from src.models.lightgbm_ranker import LightGBMRanker
from src.models.xgboost_ranker import XGBoostRanker
from src.evaluation.metrics import RankingMetrics

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int


class HyperparameterTuner:

    def __init__(self, metric: str = "ndcg@3", direction: str = "maximize",
                 n_trials: int = 20, timeout: int = 300):

        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout

        logger.info(f"HyperparameterTuner initialized "
                    f"(metric={metric}, direction={direction}, "
                    f"trials={n_trials}, timeout={timeout}s)")

    # ----------------------------------------------------------------------
    # LIGHTGBM TUNING (fixed)
    # ----------------------------------------------------------------------
    def tune_lightgbm(self, split):
        """
        Trainer calls tuner.tune_lightgbm(split)

        So LightGBMRanker is imported internally, not passed as argument.
        """

        train_df = split.train_df
        valid_df = split.test_df
        feature_names = split.feature_names

        evaluator = RankingMetrics(k_values=[3])

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
                "num_boost_round": 400,
                "early_stopping_rounds": 50,
            }

            model = LightGBMRanker.from_params(params)
            model.fit(train_df, feature_names, valid_df)

            metrics = evaluator.evaluate(model, valid_df, feature_names)
            return metrics.metrics["ndcg@3"]

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        logger.info(f"LightGBM tuning best params: {study.best_params}")
        logger.info(f"LightGBM tuning best score: {study.best_value}")

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials)
        )

    # ----------------------------------------------------------------------
    # XGBOOST TUNING (fixed)
    # ----------------------------------------------------------------------
    def tune_xgboost(self, split):

        train_df = split.train_df
        valid_df = split.test_df
        feature_names = split.feature_names

        evaluator = RankingMetrics(k_values=[3])

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "n_estimators": 500,
                "early_stopping_rounds": 50,
            }

            model = XGBoostRanker.from_params(params)
            model.fit(train_df, feature_names, valid_df)

            metrics = evaluator.evaluate(model, valid_df, feature_names)
            return metrics.metrics["ndcg@3"]

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        logger.info(f"XGBoost tuning best params: {study.best_params}")
        logger.info(f"XGBoost tuning best score: {study.best_value}")

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials)
        )

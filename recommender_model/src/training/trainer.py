"""
Trainer
=======
Main training orchestrator with MLflow integration.

Refactored from: F_ml_models.py and pipeline.py
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional MLflow integration
try:
    import mlflow
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow not installed - tracking disabled")


@dataclass
class TrainingResult:
    """Container for training results."""
    model_name: str
    model: Any
    metrics: Dict[str, float]
    params: Dict[str, Any]
    training_time: float
    run_id: Optional[str] = None


class Trainer:
    """
    Main training orchestrator.
    
    Handles:
    - Training multiple models
    - MLflow experiment tracking
    - Model comparison and selection
    
    Example:
        >>> trainer = Trainer(experiment_name="cafe-recommender")
        >>> result = trainer.train(model, split_data)
        >>> print(f"NDCG@3: {result.metrics['ndcg@3']:.4f}")
    """
    
    def __init__(
        self,
        experiment_name: str = "cafe-recommender",
        tracking_uri: str = "mlruns",
        enable_mlflow: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            enable_mlflow: Whether to enable MLflow tracking
        """
        self.experiment_name = experiment_name
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        
        if self.enable_mlflow:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking enabled: {experiment_name}")
        else:
            logger.info("MLflow tracking disabled")
    
    def train(
        self,
        model,
        split_data,
        run_name: str = None,
        tags: Dict[str, str] = None
    ) -> TrainingResult:
        """
        Train a single model with tracking.
        
        Args:
            model: Model implementing BaseRecommender
            split_data: SplitData from splitter
            run_name: Optional name for this run
            tags: Optional MLflow tags
            
        Returns:
            TrainingResult
        """
        run_name = run_name or f"{model.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Training {model.name}...")
        
        start_time = time.time()
        run_id = None
        
        # MLflow context
        if self.enable_mlflow:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                result = self._train_and_log(model, split_data, tags)
                result.run_id = run_id
        else:
            result = self._train_model(model, split_data)
        
        result.training_time = time.time() - start_time
        logger.info(f"Training complete in {result.training_time:.1f}s")
        
        return result
    
    def _train_and_log(
        self,
        model,
        split_data,
        tags: Dict[str, str] = None
    ) -> TrainingResult:
        """Train with MLflow logging."""
        # Log parameters
        params = model.get_params()
        params['n_train_samples'] = split_data.n_train_samples
        params['n_test_samples'] = split_data.n_test_samples
        params['n_features'] = len(split_data.feature_names)
        mlflow.log_params(params)
        
        # Log tags
        default_tags = {
            'model_type': model.name,
            'split_date': split_data.split_date
        }
        if tags:
            default_tags.update(tags)
        mlflow.set_tags(default_tags)
        
        # Train
        result = self._train_model(model, split_data)
        
        # Log metrics
        for metric_name, value in result.metrics.items():
            mlflow.log_metric(metric_name.replace('@', '_at_'), value)
        
        # Log model
        if hasattr(model, 'model') and model.model is not None:
            try:
                mlflow.lightgbm.log_model(model.model, artifact_path="model")
            except Exception as e:
                logger.warning(f"Failed to log model: {e}")
        
        # Log feature importance
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if importance:
                mlflow.log_dict(importance, "feature_importance.json")
        
        return result
    
    def _train_model(self, model, split_data) -> TrainingResult:
        """Train model without MLflow."""
        # Train
        model.fit(
            train_df=split_data.train_df,
            feature_names=split_data.feature_names,
            valid_df=split_data.test_df
        )
        
        # Evaluate
        from src.evaluation.metrics import RankingMetrics
        evaluator = RankingMetrics()
        metrics = evaluator.evaluate(model, split_data.test_df, split_data.feature_names)
        
        return TrainingResult(
            model_name=model.name,
            model=model,
            metrics=metrics,
            params=model.get_params(),
            training_time=0  # Will be set by caller
        )
    
    def train_multiple(
        self,
        models: List,
        split_data,
        tags: Dict[str, str] = None
    ) -> List[TrainingResult]:
        """
        Train multiple models and compare.
        
        Args:
            models: List of models to train
            split_data: SplitData
            tags: Optional tags for all runs
            
        Returns:
            List of TrainingResult, sorted by NDCG@3 descending
        """
        results = []
        
        for model in models:
            try:
                result = self.train(model, split_data, tags=tags)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to train {model.name}: {e}")
        
        # Sort by NDCG@3
        results.sort(key=lambda r: r.metrics.get('ndcg@3', 0), reverse=True)
        
        # Log comparison
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON (sorted by NDCG@3)")
        logger.info("=" * 60)
        for r in results:
            logger.info(f"  {r.model_name}: NDCG@3={r.metrics.get('ndcg@3', 0):.4f}")
        logger.info("=" * 60)
        
        return results
    
    def get_best_model(self, results: List[TrainingResult]) -> TrainingResult:
        """Get best model by NDCG@3."""
        return max(results, key=lambda r: r.metrics.get('ndcg@3', 0))
import os
import pandas as pd
from abc import ABC, abstractmethod
from src.Get_Logging_Config import get_logger  # import logger
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix
from joblib import dump
from typing import List, Tuple, Callable
import numpy as np

logger = get_logger(__name__)


class ModelSelection(ABC):
    @abstractmethod
    def evaluate(self,model:List[Tuple[str, object]],X:pd.DataFrame,y:pd.Series) ->dict:
        pass
    

class CrossValidationEvaluation(ModelSelection):
    """Evaluate models via stratified k-fold CV using a cost-based scorer."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42, 
                 scorer=None, cost_params=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cost_params = cost_params or {'tp_cost': 15, 'fp_cost': 5, 'fn_cost': 40}
        self.scorer = scorer if scorer is not None else self._create_default_scorer()
    @staticmethod
    def _create_default_scorer():
        cost_params = {'tp_cost': 15, 'fp_cost': 5, 'fn_cost': 40}

        def cost_ratio(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()

            # ⚙️ Uses cost_params local variable
            repair = cost_params['tp_cost']
            replacement = cost_params['fn_cost']
            inspection = cost_params['fp_cost']

            min_cost = (TP + FN) * repair
            model_cost = (TP * repair) + (FN * replacement) + (FP * inspection)

            return 0 if model_cost == 0 else min_cost / model_cost

        return make_scorer(cost_ratio, greater_is_better=True)
    
    def evaluate(self, models: List[Tuple[str, object]], 
                 X: pd.DataFrame, y: pd.Series) -> dict:
        """Run cross-validation for each model and compute cost ratios."""
        logger.info("Starting cross-validation with cost-sensitive scoring")
        results = {}
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                random_state=self.random_state)

        for name, model in models:
            try:
                scores = cross_val_score(model, X, y, cv=kfold, scoring=self.scorer)
                results[name] = {
                    'mean_score': scores.mean(),
                    'std_dev': scores.std(),
                    'all_scores': scores,
                }
                logger.info(f"{name}: Mean cost ratio = {scores.mean():.4f} (±{scores.std():.4f})")
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}

        return results
    
# ----------  Model Evaluator  ----------
class ModelEvaluator:
    """High-level orchestrator for evaluating and selecting models."""
    
    def __init__(self, strategy: ModelSelection):
        self._strategy = strategy
        self._results = None

    def set_strategy(self, strategy: ModelSelection):
        """Swap evaluation strategy at runtime."""
        logger.info("Switching evaluation strategy")
        self._strategy = strategy

    def evaluate_models(self, models: List[Tuple[str, object]], 
                        X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate models using current strategy."""
        logger.info("Evaluating models using selected strategy")
        self._results = self._strategy.evaluate(models, X, y)
        return self._results

    def get_best_model(self, models: List[Tuple[str, object]]) -> Tuple[str, object]:
        """Return model with highest mean score from last evaluation."""
        if not self._results:
            raise ValueError("No results available. Run evaluate_models() first.")

        valid_results = {
            name: res['mean_score'] for name, res in self._results.items()
            if 'mean_score' in res
        }
        if not valid_results:
            raise ValueError("No valid scores found. Check evaluation errors.")

        best_name = max(valid_results, key=valid_results.get)
        logger.info(f"Best model: {best_name} (Score: {valid_results[best_name]:.4f})")
        return best_name, dict(models)[best_name]

    @staticmethod
    def save_best_model(name: str, model, path: str = "models") -> str:
        """Persist best model to disk."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{name}_base.pkl")
        dump(model, filepath)
        logger.info(f"Saved best model '{name}' to {filepath}")
        return filepath



if __name__ == "__main__":
    pass



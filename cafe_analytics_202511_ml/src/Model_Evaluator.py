from src.Get_Logging_Config import get_logger  # import logger
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.Model_Selector import CrossValidationEvaluation

logger = get_logger(__name__)



class ClassificationModelEvaluator:
    """Evaluate a trained classification model from .pkl using cost-based scoring."""

    def __init__(self, cost_params=None):
        # Instantiate CrossValidationEvaluation to get the same scorer
        self.evaluator = CrossValidationEvaluation(cost_params=cost_params)
        self.scorer = self.evaluator.scorer
        self.cost_params = cost_params or {'tp_cost': 15, 'fp_cost': 5, 'fn_cost': 40}

    def load_model(self, model_path: str):
        """Load trained model from .pkl file."""
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        return model

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Compute metrics and cost ratio using the same scorer as CV."""
        y_pred = model.predict(X_test)

        # Standard classification metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Confusion matrix for cost calculation
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        repair = self.cost_params['tp_cost']
        replacement = self.cost_params['fn_cost']
        inspection = self.cost_params['fp_cost']

        min_cost = (TP + FN) * repair
        model_cost = (TP * repair) + (FN * replacement) + (FP * inspection)
        cost_ratio = 0 if model_cost == 0 else min_cost / model_cost

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Min_vs_Model_Cost": cost_ratio
        }


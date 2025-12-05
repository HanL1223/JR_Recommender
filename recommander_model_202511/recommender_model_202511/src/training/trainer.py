"""
TRAINING PIPELINE — CLEAN, MODULAR, EXTENDABLE
"""

import logging
from pathlib import Path
from datetime import datetime
import json
import pickle

from src.data import DataLoader, DataPreprocessor
from src.features import (
    ProductFeatureExtractor,
    CustomerFeatureExtractor,
    TrainingDataBuilder
)
from src.training import TemporalDataSplitter
from src.evaluation import RankingMetrics
from src.training.tuner import HyperparameterTuner

from src.models import (
    PopularityRecommender,
    PersonalFrequencyRecommender,
    LightGBMRanker,
    XGBoostRanker,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def train_pipeline(data_path: str,
                   do_tuning: bool = False,
                   n_trials: int = 20,
                   timeout: int = 300):

    # ----------------------------------------------------------------------
    # LOAD + PREPROCESS + FEATURE EXTRACTION
    # ----------------------------------------------------------------------
    raw = DataLoader().load_csv(data_path)
    prepared = DataPreprocessor(min_orders=2).transform(raw.transactions)

    product_features = ProductFeatureExtractor().extract(prepared)
    customer_profiles = CustomerFeatureExtractor().extract(prepared)
    training_data = TrainingDataBuilder().build(prepared, product_features, customer_profiles)

    split = TemporalDataSplitter(test_ratio=0.2).split(training_data)

    train_df = split.train_df
    test_df = split.test_df
    feature_names = split.feature_names

    logger.info(f"Train samples: {split.n_train_samples:,}")
    logger.info(f"Test samples: {split.n_test_samples:,}")

    evaluator = RankingMetrics(k_values=[1, 3, 5, 10])
    tuner = HyperparameterTuner(n_trials=n_trials, timeout=timeout)

    # ----------------------------------------------------------------------
    # CANDIDATE MODELS (easy to add more)
    # ----------------------------------------------------------------------
    candidate_models = [
        ("Popularity", PopularityRecommender(), False),
        ("PersonalFreq_s0.2", PersonalFrequencyRecommender(0.2), False),
        ("PersonalFreq_s0.3", PersonalFrequencyRecommender(0.3), False),
        ("PersonalFreq_s0.5", PersonalFrequencyRecommender(0.5), False),
        ("LightGBM_default", LightGBMRanker(), True),
        ("XGBoost_default", XGBoostRanker(), True),
    ]

    results = []

    # ----------------------------------------------------------------------
    # TRAIN + EVALUATE ALL MODELS
    # ----------------------------------------------------------------------
    for name, model, supports_tuning in candidate_models:
        try:
            logger.info(f"\nTraining: {name}")
            model.fit(train_df, feature_names, valid_df=test_df)
            metrics = evaluator.evaluate(model, test_df, feature_names)

            results.append({
                "name": name,
                "model": model,
                "metrics": metrics,
                "params": model.get_params(),
                "supports_tuning": supports_tuning
            })

            logger.info(f"{name} → NDCG@3 = {metrics['ndcg@3']:.4f}")

        except Exception as e:
            logger.warning(f"⚠️ Model {name} failed: {e}")

    # ----------------------------------------------------------------------
    # SELECT BEST BASE MODEL
    # ----------------------------------------------------------------------
    best_base = max(results, key=lambda r: r["metrics"]["ndcg@3"])
    logger.info(f"\n🏆 Best base model: {best_base['name']} "
                f"(NDCG@3={best_base['metrics']['ndcg@3']:.4f})")

    final_model = best_base["model"]
    final_params = best_base["params"]

    # ----------------------------------------------------------------------
    # OPTIONAL — HYPERPARAMETER TUNING
    # ----------------------------------------------------------------------
    if do_tuning and best_base["supports_tuning"]:

        logger.info(f"\n🔧 Tuning model: {best_base['name']}")

        if "LightGBM" in best_base["name"]:
            tuning = tuner.tune_lightgbm(split)
            final_model = LightGBMRanker.from_params(tuning.best_params)
        elif "XGBoost" in best_base["name"]:
            tuning = tuner.tune_xgboost(split)
            final_model = XGBoostRanker.from_params(tuning.best_params)
        else:
            tuning = None

        if tuning:
            logger.info(f"Best tuned NDCG@3={tuning.best_score:.4f}")
            final_model.fit(train_df, feature_names, valid_df=test_df)
            final_params = tuning.best_params

    # ----------------------------------------------------------------------
    # SAVE FINAL MODEL
    # ----------------------------------------------------------------------
    output_dir = Path("models/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": final_model,
        "model_name": best_base["name"],
        "feature_names": feature_names,
        "metrics": best_base["metrics"],
        "params": final_params,
        "training_date": datetime.now().isoformat(),
        "tuned": do_tuning,
    }

    with open(output_dir / "recommender.pkl", "wb") as f:
        pickle.dump(bundle, f)

    with open(output_dir / "model_info.json", "w") as f:
        json.dump(bundle, f, indent=2)

    logger.info("\n Training pipeline complete.")
    logger.info(f"Best model saved to: {output_dir/'recommender.pkl'}")

    return bundle

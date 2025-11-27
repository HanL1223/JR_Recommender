"""
Main Training Pipeline
======================
End-to-end pipeline for training the recommendation model.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --tune --trials 30
"""

import argparse
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataValidator, DataPreprocessor
from src.features import ProductFeatureExtractor, CustomerFeatureExtractor, TrainingDataBuilder
from src.models import PopularityRecommender, PersonalFrequencyRecommender, LightGBMRanker
from src.training import Trainer, TemporalDataSplitter, HyperparameterTuner
from src.evaluation import RankingMetrics


def run_pipeline(
    data_path: str = "./src/data/data_raw.csv",
    output_dir: str = "models/artifacts",
    do_tuning: bool = False,
    n_trials: int = 20,
    timeout: int = 300
):
    """
    Run the complete training pipeline.
    
    Steps:
    1. Load and validate data
    2. Preprocess and build customer histories
    3. Extract features (product and customer)
    4. Build training data with proper temporal features
    5. Split data temporally
    6. Train baseline models
    7. Train ML model (with optional tuning)
    8. Evaluate and compare
    9. Save best model
    """
    total_start = time.time()
    
    logger.info("=" * 70)
    logger.info("   CAFE RECOMMENDER TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"   Data: {data_path}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Tuning: {do_tuning}")
    logger.info("=" * 70)
    
    # ========================================
    # STEP 1: Load Data
    # ========================================
    logger.info("\n[STEP 1] Loading data...")
    loader = DataLoader()
    raw_data = loader.load_csv(data_path)
    
    # ========================================
    # STEP 2: Validate Data
    # ========================================
    logger.info("\n[STEP 2] Validating data...")
    validator = DataValidator()
    validation = validator.validate(raw_data.transactions)
    logger.info(validation.summary())
    
    if not validation.is_valid:
        raise ValueError("Data validation failed!")
    
    # ========================================
    # STEP 3: Preprocess
    # ========================================
    logger.info("\n[STEP 3] Preprocessing...")
    preprocessor = DataPreprocessor(min_orders=2)
    prepared_data = preprocessor.transform(raw_data.transactions)
    
    # ========================================
    # STEP 4: Feature Engineering
    # ========================================
    logger.info("\n[STEP 4] Feature engineering...")
    
    # Product features
    product_extractor = ProductFeatureExtractor(min_support=0.001)
    product_features = product_extractor.extract(prepared_data)
    
    # Customer profiles
    customer_extractor = CustomerFeatureExtractor()
    customer_profiles = customer_extractor.extract(prepared_data)
    
    # ========================================
    # STEP 5: Build Training Data
    # ========================================
    logger.info("\n[STEP 5] Building training data...")
    builder = TrainingDataBuilder(negative_ratio=5)
    training_data = builder.build(prepared_data, product_features, customer_profiles)
    
    # ========================================
    # STEP 6: Temporal Split
    # ========================================
    logger.info("\n[STEP 6] Temporal train/test split...")
    splitter = TemporalDataSplitter(test_ratio=0.2)
    split_data = splitter.split(training_data)
    
    # ========================================
    # STEP 7: Train Models
    # ========================================
    logger.info("\n[STEP 7] Training models...")
    trainer = Trainer(experiment_name="cafe-recommender")
    
    # Baseline models
    models = [
        PopularityRecommender(),
        PersonalFrequencyRecommender(smoothing=0.3)
    ]
    
    # ML model
    try:
        if do_tuning:
            logger.info("Tuning LightGBM hyperparameters...")
            tuner = HyperparameterTuner(n_trials=n_trials, timeout=timeout)
            tuning_result = tuner.tune_lightgbm(split_data)
            ml_model = LightGBMRanker.from_params(tuning_result.best_params)
        else:
            ml_model = LightGBMRanker()
        
        models.append(ml_model)
    except ImportError:
        logger.warning("LightGBM not available, using baselines only")
    
    # Train all models
    results = trainer.train_multiple(models, split_data)
    
    # ========================================
    # STEP 8: Select Best Model
    # ========================================
    logger.info("\n[STEP 8] Selecting best model...")
    best_result = trainer.get_best_model(results)
    
    logger.info(f"\nBest model: {best_result.model_name}")
    logger.info(f"NDCG@3: {best_result.metrics['ndcg@3']:.4f}")
    
    # ========================================
    # STEP 9: Save Model
    # ========================================
    logger.info("\n[STEP 9] Saving model...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model bundle
    import pickle
    bundle = {
        'model': best_result.model,
        'product_features': product_features,
        'prepared_data': prepared_data,
        'customer_profiles': customer_profiles,
        'metrics': best_result.metrics,
        'params': best_result.params
    }
    
    model_path = output_path / "recommender.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    
    logger.info(f"Model saved to {model_path}")
    
    # ========================================
    # Summary
    # ========================================
    total_time = time.time() - total_start
    
    logger.info("\n" + "=" * 70)
    logger.info("   PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"   Best model: {best_result.model_name}")
    logger.info(f"   NDCG@3: {best_result.metrics['ndcg@3']:.4f}")
    logger.info(f"   Hit@3: {best_result.metrics['hit_rate@3']:.4f}")
    logger.info(f"   Total time: {total_time:.1f}s")
    logger.info("=" * 70)
    
    return {
        'model': best_result.model,
        'metrics': best_result.metrics,
        'model_path': model_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train recommendation model')
    parser.add_argument('--data', type=str, default='./src/data/data_raw.csv')
    parser.add_argument('--output', type=str, default='models/artifacts')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=20, help='Tuning trials')
    parser.add_argument('--timeout', type=int, default=300, help='Tuning timeout (seconds)')
    
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        do_tuning=args.tune,
        n_trials=args.trials,
        timeout=args.timeout
    )
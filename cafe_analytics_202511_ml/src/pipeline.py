"""
Next Order Recommendation Pipeline
===================================
Complete pipeline from raw data to production recommendations.

Usage:
    python pipeline.py                    # Quick run (no tuning)
    python pipeline.py --tune             # Full run with hyperparameter tuning
    python pipeline.py --tune --trials 50 # Custom number of trials
"""
import os
import sys
project_path = os.path.abspath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,project_path)

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


logger = logging.getLogger(__name__)

from src.data_prepareation import create_data_preparation
from src.feature_engineering import create_feature_engineering
from src.training_data_creation import create_training_data_builder
from src.train_test_split import create_data_splitter
from src.baseline_model import create_baseline_trainer
from src.Model_training import create_ml_trainer
from src.evaluation import create_evaluator
from src.final_recommander import create_recommender_service


def run_pipeline(data_path: str = "../data/raw/data_raw.csv", 
                 do_tuning: bool = False,
                 n_trials: int = 20,
                 timeout: int = 300):
    """
    Execute the complete recommendation pipeline.
    """
    total_start = time.time()
    
    logger.info("\n" + "=" * 70)
    logger.info("   NEXT ORDER RECOMMENDATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"   Data: {data_path}")
    logger.info(f"   Tuning: {do_tuning} (trials={n_trials}, timeout={timeout}s)")
    logger.info("=" * 70 + "\n")
    
    # Step A: Data Preparation
    prepared_data = create_data_preparation(min_orders=2).run(data_path)
    
    # Step B: Feature Engineering
    features = create_feature_engineering().run(prepared_data)
    
    # Step C: Training Data (with augmentation)
    training_data = create_training_data_builder(negative_ratio=5).run(prepared_data, features)
    
    # Step D: Temporal Train/Test Split
    split_data = create_data_splitter(test_ratio=0.2).run(training_data)
    
    # Step E: Baseline Models
    baseline_models = create_baseline_trainer().run(split_data)
    
    # Step F: ML Models (with optional tuning)
    ml_models = create_ml_trainer(n_trials=n_trials, timeout=timeout).run(split_data, do_tuning=do_tuning)
    
    # Step G: Evaluation
    eval_results = create_evaluator().run(split_data, baseline_models, ml_models)
    
    # Step H: Recommender Service
    recommender = create_recommender_service(
        ml_models, baseline_models, features, prepared_data, eval_results
    )
    
    # Demo
    logger.info("\n" + "=" * 70)
    logger.info("   DEMO RECOMMENDATIONS")
    logger.info("=" * 70)
    
    # Sample customers
    sample_customers = prepared_data.customer_list[:3]
    for cust_id in sample_customers:
        pred = recommender.recommend(cust_id, top_k=3)
        logger.info(recommender.format(pred))
    
    # Summary
    total_time = time.time() - total_start
    
    logger.info("\n" + "=" * 70)
    logger.info("   PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"   Total customers: {len(prepared_data.customer_list):,}")
    logger.info(f"   Training samples: {len(split_data.train_df):,}")
    logger.info(f"   Test samples: {len(split_data.test_df):,}")
    logger.info(f"   Best model: {eval_results.best_model} (NDCG@3: {eval_results.best_ndcg:.4f})")
    logger.info(f"   Total time: {total_time:.1f} seconds")
    logger.info("=" * 70)
    
    logger.info("\n✅ Pipeline complete!")
    
    return {
        'prepared_data': prepared_data,
        'features': features,
        'training_data': training_data,
        'split_data': split_data,
        'baseline_models': baseline_models,
        'ml_models': ml_models,
        'eval_results': eval_results,
        'recommender': recommender
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Next Order Recommendation Pipeline')
    parser.add_argument('--data', type=str, default='../data/raw/data_raw.csv', help='Path to data file')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=20, help='Number of tuning trials')
    parser.add_argument('--timeout', type=int, default=300, help='Tuning timeout in seconds')
    
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data,
        do_tuning=args.tune,
        n_trials=args.trials,
        timeout=args.timeout
    )
"""
================================================================================
COMPLETE TRAINING PIPELINE WITH MODEL SELECTION & HYPERPARAMETER TUNING
================================================================================

This script shows the FULL training process:
1. Train multiple models (baselines + ML)
2. Compare and select best model
3. Hyperparameter tuning for the ML model
4. Final evaluation with best parameters

Usage:
    python scripts/train_with_tuning.py
    python scripts/train_with_tuning.py --tune --trials 30
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    data_path: str = "data/raw/transactions.csv",
    do_tuning: bool = False,
    n_trials: int = 20,
    timeout: int = 300
):
    """
    Complete training pipeline with model selection and tuning.
    """
    
    # ==========================================================================
    # STEP 1-3: DATA PREPARATION (same as before)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEPS 1-3: DATA PREPARATION")
    print("=" * 70)
    
    from src.data import DataLoader, DataPreprocessor
    from src.features import ProductFeatureExtractor, CustomerFeatureExtractor, TrainingDataBuilder
    from src.training import TemporalDataSplitter
    
    # Load
    raw = DataLoader().load_csv(data_path)
    
    # Preprocess
    prepared = DataPreprocessor(min_orders=2).transform(raw.transactions)
    
    # Features
    product_features = ProductFeatureExtractor().extract(prepared)
    customer_profiles = CustomerFeatureExtractor().extract(prepared)
    training_data = TrainingDataBuilder().build(prepared, product_features, customer_profiles)
    
    # Split
    split = TemporalDataSplitter(test_ratio=0.2).split(training_data)
    
    print(f"Training samples: {split.n_train_samples:,}")
    print(f"Test samples: {split.n_test_samples:,}")
    print(f"Features: {len(split.feature_names)}")
    
    
    # ==========================================================================
    # STEP 4: TRAIN ALL CANDIDATE MODELS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN ALL CANDIDATE MODELS")
    print("=" * 70)
    
    from src.models import PopularityRecommender, PersonalFrequencyRecommender, LightGBMRanker
    from src.evaluation import RankingMetrics
    
    evaluator = RankingMetrics(k_values=[1, 3, 5, 10])
    
    # Store all results
    model_results = []
    
    # ----- Model 1: Popularity Baseline -----
    print("\n--- Model 1: Popularity Baseline ---")
    popularity_model = PopularityRecommender()
    popularity_model.fit(split.train_df, split.feature_names)
    popularity_metrics = evaluator.evaluate(popularity_model, split.test_df, split.feature_names)
    
    model_results.append({
        'name': 'Popularity',
        'model': popularity_model,
        'metrics': popularity_metrics,
        'params': popularity_model.get_params()
    })
    print(f"  NDCG@3: {popularity_metrics['ndcg@3']:.4f}")
    
    # ----- Model 2: PersonalFrequency (smoothing=0.2) -----
    print("\n--- Model 2: PersonalFrequency (smoothing=0.2) ---")
    pf_02 = PersonalFrequencyRecommender(smoothing=0.2)
    pf_02.fit(split.train_df, split.feature_names)
    pf_02_metrics = evaluator.evaluate(pf_02, split.test_df, split.feature_names)
    
    model_results.append({
        'name': 'PersonalFreq_s0.2',
        'model': pf_02,
        'metrics': pf_02_metrics,
        'params': pf_02.get_params()
    })
    print(f"  NDCG@3: {pf_02_metrics['ndcg@3']:.4f}")
    
    # ----- Model 3: PersonalFrequency (smoothing=0.3) -----
    print("\n--- Model 3: PersonalFrequency (smoothing=0.3) ---")
    pf_03 = PersonalFrequencyRecommender(smoothing=0.3)
    pf_03.fit(split.train_df, split.feature_names)
    pf_03_metrics = evaluator.evaluate(pf_03, split.test_df, split.feature_names)
    
    model_results.append({
        'name': 'PersonalFreq_s0.3',
        'model': pf_03,
        'metrics': pf_03_metrics,
        'params': pf_03.get_params()
    })
    print(f"  NDCG@3: {pf_03_metrics['ndcg@3']:.4f}")
    
    # ----- Model 4: PersonalFrequency (smoothing=0.5) -----
    print("\n--- Model 4: PersonalFrequency (smoothing=0.5) ---")
    pf_05 = PersonalFrequencyRecommender(smoothing=0.5)
    pf_05.fit(split.train_df, split.feature_names)
    pf_05_metrics = evaluator.evaluate(pf_05, split.test_df, split.feature_names)
    
    model_results.append({
        'name': 'PersonalFreq_s0.5',
        'model': pf_05,
        'metrics': pf_05_metrics,
        'params': pf_05.get_params()
    })
    print(f"  NDCG@3: {pf_05_metrics['ndcg@3']:.4f}")
    
    # ----- Model 5: LightGBM (default params) -----
    print("\n--- Model 5: LightGBM (default params) ---")
    try:
        lgb_default = LightGBMRanker(
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            min_data_in_leaf=20,
            num_boost_round=500,
            early_stopping_rounds=50
        )
        lgb_default.fit(split.train_df, split.feature_names, split.test_df)
        lgb_default_metrics = evaluator.evaluate(lgb_default, split.test_df, split.feature_names)
        
        model_results.append({
            'name': 'LightGBM_default',
            'model': lgb_default,
            'metrics': lgb_default_metrics,
            'params': lgb_default.get_params()
        })
        print(f"  NDCG@3: {lgb_default_metrics['ndcg@3']:.4f}")
        lightgbm_available = True
        
    except ImportError:
        print("  ⚠️ LightGBM not installed, skipping ML models")
        lightgbm_available = False
    
    
    # ==========================================================================
    # STEP 5: MODEL COMPARISON & SELECTION (BEFORE TUNING)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: MODEL COMPARISON (BEFORE TUNING)")
    print("=" * 70)
    
    # Sort by NDCG@3
    model_results.sort(key=lambda x: x['metrics']['ndcg@3'], reverse=True)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    MODEL COMPARISON TABLE                        │")
    print("├──────────────────────┬──────────┬──────────┬──────────┬─────────┤")
    print("│ Model                │ NDCG@1   │ NDCG@3   │ NDCG@5   │ Hit@3   │")
    print("├──────────────────────┼──────────┼──────────┼──────────┼─────────┤")
    
    for result in model_results:
        m = result['metrics']
        name = result['name'][:20].ljust(20)
        print(f"│ {name} │ {m['ndcg@1']:.4f}   │ {m['ndcg@3']:.4f}   │ {m['ndcg@5']:.4f}   │ {m['hit_rate@3']:.4f}  │")
    
    print("└──────────────────────┴──────────┴──────────┴──────────┴─────────┘")
    
    # Best model before tuning
    best_before_tuning = model_results[0]
    best_model_type = best_before_tuning['name']  # e.g., "LightGBM_default" or "XGBoost"
    print(f"\n🏆 Best Model (before tuning): {best_before_tuning['name']}")
    print(f"   NDCG@3: {best_before_tuning['metrics']['ndcg@3']:.4f}")
    
    
    # ==========================================================================
    # STEP 6: HYPERPARAMETER TUNING (Tune the BEST model)
    # ==========================================================================
    tuned_model = None
    tuning_result = None
    
    if do_tuning:
        print("\n" + "=" * 70)
        print("STEP 6: HYPERPARAMETER TUNING (Optuna)")
        print("=" * 70)
        
        from src.training import HyperparameterTuner
        
        tuner = HyperparameterTuner(
            n_trials=n_trials,
            timeout=timeout,
            metric='ndcg@3',
            direction='maximize'
        )
        
        # Determine which model to tune based on best performer
        if 'LightGBM' in best_model_type and lightgbm_available:
            print(f"\nTuning LightGBM (best performing model):")
            print(f"  Trials: {n_trials}")
            print(f"  Timeout: {timeout}s")
            print(f"  Metric: NDCG@3 (maximize)")
            
            tuning_result = tuner.tune_lightgbm(split)
            
            # Train with best params
            tuned_model = LightGBMRanker.from_params(tuning_result.best_params)
            tuned_model_name = "LightGBM_tuned"
            
        elif 'XGBoost' in best_model_type:
            try:
                from src.models import XGBoostRanker
                print(f"\nTuning XGBoost (best performing model):")
                print(f"  Trials: {n_trials}")
                print(f"  Timeout: {timeout}s")
                print(f"  Metric: NDCG@3 (maximize)")
                
                tuning_result = tuner.tune_xgboost(split)
                
                # Train with best params
                tuned_model = XGBoostRanker.from_params(tuning_result.best_params)
                tuned_model_name = "XGBoost_tuned"
                
            except ImportError:
                print("⚠️ XGBoost not available for tuning")
                
        elif 'CatBoost' in best_model_type:
            try:
                from src.models import CatBoostRanker
                print(f"\nTuning CatBoost (best performing model):")
                print(f"  Trials: {n_trials}")
                print(f"  Timeout: {timeout}s}")
                print(f"  Metric: NDCG@3 (maximize)")
                
                tuning_result = tuner.tune_catboost(split)
                
                # Train with best params
                tuned_model = CatBoostRanker.from_params(tuning_result.best_params)
                tuned_model_name = "CatBoost_tuned"
                
            except ImportError:
                print("⚠️ CatBoost not available for tuning")
        
        else:
            # Best model is a baseline - tune the best available ML model
            print(f"\nBest model is baseline ({best_model_type})")
            print("Tuning best available ML model instead...")
            
            if lightgbm_available:
                print(f"\nTuning LightGBM:")
                tuning_result = tuner.tune_lightgbm(split)
                tuned_model = LightGBMRanker.from_params(tuning_result.best_params)
                tuned_model_name = "LightGBM_tuned"
            else:
                print("⚠️ No ML model available for tuning")
        
        # If tuning was successful, train and evaluate the tuned model
        if tuning_result is not None and tuned_model is not None:
            print("\n" + "-" * 50)
            print("TUNING RESULTS")
            print("-" * 50)
            print(f"Best NDCG@3: {tuning_result.best_score:.4f}")
            print(f"Trials completed: {tuning_result.n_trials}")
            print("\nBest Hyperparameters:")
            for param, value in tuning_result.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
            
            # Train final model with best params
            print(f"\n--- Training {tuned_model_name} with best params ---")
            tuned_model.fit(split.train_df, split.feature_names, split.test_df)
            tuned_metrics = evaluator.evaluate(tuned_model, split.test_df, split.feature_names)
            
            model_results.append({
                'name': tuned_model_name,
                'model': tuned_model,
                'metrics': tuned_metrics,
                'params': tuning_result.best_params
            })
            
            print(f"  NDCG@3: {tuned_metrics['ndcg@3']:.4f}")
            
    elif not do_tuning:
        print("\n⚠️ Tuning skipped - use --tune flag to enable")
    
    
    # ==========================================================================
    # STEP 7: FINAL MODEL SELECTION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: FINAL MODEL SELECTION")
    print("=" * 70)
    
    # Re-sort after adding tuned model
    model_results.sort(key=lambda x: x['metrics']['ndcg@3'], reverse=True)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                 FINAL MODEL COMPARISON                           │")
    print("├──────────────────────┬──────────┬──────────┬──────────┬─────────┤")
    print("│ Model                │ NDCG@1   │ NDCG@3   │ NDCG@5   │ Hit@3   │")
    print("├──────────────────────┼──────────┼──────────┼──────────┼─────────┤")
    
    for i, result in enumerate(model_results):
        m = result['metrics']
        name = result['name'][:20].ljust(20)
        marker = "★" if i == 0 else " "
        print(f"│{marker}{name}│ {m['ndcg@1']:.4f}   │ {m['ndcg@3']:.4f}   │ {m['ndcg@5']:.4f}   │ {m['hit_rate@3']:.4f}  │")
    
    print("└──────────────────────┴──────────┴──────────┴──────────┴─────────┘")
    
    # Select best
    best_result = model_results[0]
    best_model = best_result['model']
    best_metrics = best_result['metrics']
    best_params = best_result['params']
    
    print(f"\n{'=' * 50}")
    print(f"🏆 BEST MODEL: {best_result['name']}")
    print(f"{'=' * 50}")
    print(f"\nPerformance Metrics:")
    print(f"  NDCG@1:     {best_metrics['ndcg@1']:.4f}")
    print(f"  NDCG@3:     {best_metrics['ndcg@3']:.4f}")
    print(f"  NDCG@5:     {best_metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10:    {best_metrics['ndcg@10']:.4f}")
    print(f"  Hit Rate@3: {best_metrics['hit_rate@3']:.4f}")
    print(f"  MRR:        {best_metrics['mrr']:.4f}")
    
    print(f"\nModel Parameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")
    
    # Feature importance (if ML model)
    if hasattr(best_model, 'get_feature_importance'):
        importance = best_model.get_feature_importance()
        if importance:
            print(f"\nTop 10 Feature Importance:")
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:10]
            for i, (feat, imp) in enumerate(sorted_imp, 1):
                print(f"  {i:2d}. {feat}: {imp:.1f}")
    
    
    # ==========================================================================
    # STEP 8: SAVE BEST MODEL
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: SAVE BEST MODEL")
    print("=" * 70)
    
    import pickle
    from datetime import datetime
    import json
    
    output_dir = Path("models/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model bundle
    model_bundle = {
        'model': best_model,
        'model_name': best_result['name'],
        'product_features': product_features,
        'prepared_data': prepared,
        'customer_profiles': customer_profiles,
        'feature_names': training_data.feature_names,
        'metrics': best_metrics,
        'params': best_params,
        'training_date': datetime.now().isoformat(),
        'tuning_enabled': do_tuning,
        'tuning_trials': n_trials if do_tuning else 0
    }
    
    model_path = output_dir / "recommender.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    
    print(f"Model saved to: {model_path}")
    
    # Save model info as JSON (human-readable)
    model_info = {
        'model_name': best_result['name'],
        'training_date': datetime.now().isoformat(),
        'metrics': {k: round(v, 4) for k, v in best_metrics.items()},
        'params': {k: v if not isinstance(v, float) else round(v, 6) for k, v in best_params.items()},
        'n_customers': prepared.n_customers,
        'n_products': prepared.n_products,
        'n_train_samples': split.n_train_samples,
        'n_test_samples': split.n_test_samples,
        'split_date': split.split_date,
        'tuning_enabled': do_tuning,
        'tuning_trials': n_trials if do_tuning else 0,
        'all_models_compared': [r['name'] for r in model_results]
    }
    
    info_path = output_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model info saved to: {info_path}")
    
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
Summary:
  Data: {prepared.n_customers:,} customers, {prepared.n_products:,} products
  Models compared: {len(model_results)}
  Tuning: {'Yes (' + str(n_trials) + ' trials)' if do_tuning else 'No'}
  
Best Model: {best_result['name']}
  NDCG@3:     {best_metrics['ndcg@3']:.4f}
  Hit Rate@3: {best_metrics['hit_rate@3']:.4f}
  
Saved to: {model_path}
""")
    print("=" * 70)
    
    return {
        'best_model': best_model,
        'best_metrics': best_metrics,
        'best_params': best_params,
        'all_results': model_results,
        'tuning_result': tuning_result
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with model selection and tuning')
    parser.add_argument('--data', type=str, default='src/data/data_rww.csv')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=20, help='Number of tuning trials')
    parser.add_argument('--timeout', type=int, default=300, help='Tuning timeout in seconds')
    
    args = parser.parse_args()
    
    run_complete_pipeline(
        data_path=args.data,
        do_tuning=args.tune,
        n_trials=args.trials,
        timeout=args.timeout
    )
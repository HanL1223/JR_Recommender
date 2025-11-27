"""
================================================================================
COMPLETE SYSTEM TEST - Validates All Components
================================================================================

This script tests EVERY component of the recommender system:
1. Data Loading & Validation
2. Feature Engineering
3. Training Data Building
4. Model Training (all models)
5. Model Selection
6. Hyperparameter Tuning
7. Evaluation
8. Inference & Cold Start
9. Model Saving/Loading
10. MLflow Tracking (if available)

Run this to verify your entire pipeline works:
    python scripts/test_complete_pipeline.py

Expected output: All tests pass with ✅
"""

import sys
import os
import logging
import tempfile
import pickle
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result(test_name, passed, details=""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} - {test_name}")
    if details and not passed:
        print(f"         {details}")


def run_all_tests(data_path: str = "data/raw/transactions.csv"):
    """
    Run complete system test.
    
    Args:
        data_path: Path to transaction data
    """
    results = []
    
    print("\n" + "=" * 70)
    print("  🧪 COMPLETE SYSTEM TEST")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # ==========================================================================
    # TEST 1: DATA LOADING
    # ==========================================================================
    print_header("TEST 1: DATA LOADING")
    
    try:
        from src.data import DataLoader
        
        loader = DataLoader()
        raw_data = loader.load_csv(data_path)
        
        assert raw_data.transactions is not None, "Transactions is None"
        assert len(raw_data.transactions) > 0, "No transactions loaded"
        assert raw_data.n_customers > 0, "No customers"
        assert raw_data.n_products > 0, "No products"
        
        print_result("DataLoader.load_csv()", True)
        print(f"         Loaded {raw_data.n_rows:,} rows, {raw_data.n_customers:,} customers")
        results.append(("DataLoader", True))
        
    except Exception as e:
        print_result("DataLoader.load_csv()", False, str(e))
        results.append(("DataLoader", False))
        return results  # Can't continue without data
    
    # ==========================================================================
    # TEST 2: DATA VALIDATION
    # ==========================================================================
    print_header("TEST 2: DATA VALIDATION")
    
    try:
        from src.data import DataValidator
        
        validator = DataValidator()
        report = validator.validate(raw_data.transactions)
        
        assert report is not None, "Validation report is None"
        assert hasattr(report, 'is_valid'), "Missing is_valid attribute"
        
        print_result("DataValidator.validate()", True)
        print(f"         Valid: {report.is_valid}, Issues: {len(report.issues)}")
        results.append(("DataValidator", True))
        
    except Exception as e:
        print_result("DataValidator.validate()", False, str(e))
        results.append(("DataValidator", False))
    
    # ==========================================================================
    # TEST 3: DATA PREPROCESSING
    # ==========================================================================
    print_header("TEST 3: DATA PREPROCESSING")
    
    try:
        from src.data import DataPreprocessor
        
        preprocessor = DataPreprocessor(min_orders=2)
        prepared_data = preprocessor.transform(raw_data.transactions)
        
        assert prepared_data.customer_histories is not None, "No customer histories"
        assert len(prepared_data.customer_histories) > 0, "Empty customer histories"
        assert prepared_data.n_products > 0, "No products"
        
        print_result("DataPreprocessor.transform()", True)
        print(f"         {prepared_data.n_customers:,} customers, {prepared_data.n_products:,} products")
        results.append(("DataPreprocessor", True))
        
    except Exception as e:
        print_result("DataPreprocessor.transform()", False, str(e))
        results.append(("DataPreprocessor", False))
        return results
    
    # ==========================================================================
    # TEST 4: PRODUCT FEATURE EXTRACTION
    # ==========================================================================
    print_header("TEST 4: PRODUCT FEATURE EXTRACTION")
    
    try:
        from src.features import ProductFeatureExtractor
        
        product_extractor = ProductFeatureExtractor(min_support=0.001)
        product_features = product_extractor.extract(prepared_data)
        
        assert product_features.popularity is not None, "No popularity"
        assert len(product_features.popularity) > 0, "Empty popularity"
        assert product_features.cooccurrence is not None, "No cooccurrence"
        
        print_result("ProductFeatureExtractor.extract()", True)
        print(f"         {len(product_features.popularity)} products with popularity")
        results.append(("ProductFeatureExtractor", True))
        
    except Exception as e:
        print_result("ProductFeatureExtractor.extract()", False, str(e))
        results.append(("ProductFeatureExtractor", False))
        return results
    
    # ==========================================================================
    # TEST 5: CUSTOMER FEATURE EXTRACTION
    # ==========================================================================
    print_header("TEST 5: CUSTOMER FEATURE EXTRACTION")
    
    try:
        from src.features import CustomerFeatureExtractor
        
        customer_extractor = CustomerFeatureExtractor()
        customer_profiles = customer_extractor.extract(prepared_data)
        
        assert customer_profiles is not None, "No profiles"
        assert len(customer_profiles) > 0, "Empty profiles"
        
        # Check archetype distribution
        archetypes = customer_extractor.get_archetype_summary(customer_profiles)
        
        print_result("CustomerFeatureExtractor.extract()", True)
        print(f"         {len(customer_profiles)} profiles, {len(archetypes)} archetypes")
        results.append(("CustomerFeatureExtractor", True))
        
    except Exception as e:
        print_result("CustomerFeatureExtractor.extract()", False, str(e))
        results.append(("CustomerFeatureExtractor", False))
        return results
    
    # ==========================================================================
    # TEST 6: TRAINING DATA BUILDING
    # ==========================================================================
    print_header("TEST 6: TRAINING DATA BUILDING")
    
    try:
        from src.features import TrainingDataBuilder
        
        builder = TrainingDataBuilder(negative_ratio=5, random_seed=42)
        training_data = builder.build(prepared_data, product_features, customer_profiles)
        
        assert training_data.samples_df is not None, "No samples"
        assert len(training_data.samples_df) > 0, "Empty samples"
        assert training_data.feature_names is not None, "No feature names"
        assert len(training_data.feature_names) > 0, "Empty feature names"
        
        print_result("TrainingDataBuilder.build()", True)
        print(f"         {len(training_data.samples_df):,} samples, {len(training_data.feature_names)} features")
        results.append(("TrainingDataBuilder", True))
        
    except Exception as e:
        print_result("TrainingDataBuilder.build()", False, str(e))
        results.append(("TrainingDataBuilder", False))
        return results
    
    # ==========================================================================
    # TEST 7: TEMPORAL DATA SPLITTING
    # ==========================================================================
    print_header("TEST 7: TEMPORAL DATA SPLITTING")
    
    try:
        from src.training import TemporalDataSplitter
        
        splitter = TemporalDataSplitter(test_ratio=0.2)
        split_data = splitter.split(training_data)
        
        assert split_data.train_df is not None, "No train data"
        assert split_data.test_df is not None, "No test data"
        assert len(split_data.train_df) > 0, "Empty train data"
        assert len(split_data.test_df) > 0, "Empty test data"
        
        print_result("TemporalDataSplitter.split()", True)
        print(f"         Train: {split_data.n_train_samples:,}, Test: {split_data.n_test_samples:,}")
        results.append(("TemporalDataSplitter", True))
        
    except Exception as e:
        print_result("TemporalDataSplitter.split()", False, str(e))
        results.append(("TemporalDataSplitter", False))
        return results
    
    # ==========================================================================
    # TEST 8: BASELINE MODELS
    # ==========================================================================
    print_header("TEST 8: BASELINE MODELS")
    
    from src.models import PopularityRecommender, PersonalFrequencyRecommender
    from src.evaluation import RankingMetrics
    
    evaluator = RankingMetrics(k_values=[1, 3, 5])
    
    # Test Popularity model
    try:
        pop_model = PopularityRecommender()
        pop_model.fit(split_data.train_df, split_data.feature_names)
        pop_metrics = evaluator.evaluate(pop_model, split_data.test_df, split_data.feature_names)
        
        assert 'ndcg@3' in pop_metrics, "Missing NDCG@3"
        
        print_result("PopularityRecommender", True)
        print(f"         NDCG@3: {pop_metrics['ndcg@3']:.4f}")
        results.append(("PopularityRecommender", True))
        
    except Exception as e:
        print_result("PopularityRecommender", False, str(e))
        results.append(("PopularityRecommender", False))
    
    # Test PersonalFrequency model
    try:
        pf_model = PersonalFrequencyRecommender(smoothing=0.3)
        pf_model.fit(split_data.train_df, split_data.feature_names)
        pf_metrics = evaluator.evaluate(pf_model, split_data.test_df, split_data.feature_names)
        
        assert 'ndcg@3' in pf_metrics, "Missing NDCG@3"
        
        print_result("PersonalFrequencyRecommender", True)
        print(f"         NDCG@3: {pf_metrics['ndcg@3']:.4f}")
        results.append(("PersonalFrequencyRecommender", True))
        
    except Exception as e:
        print_result("PersonalFrequencyRecommender", False, str(e))
        results.append(("PersonalFrequencyRecommender", False))
    
    # ==========================================================================
    # TEST 9: ML MODELS (LightGBM, XGBoost, CatBoost)
    # ==========================================================================
    print_header("TEST 9: ML MODELS")
    
    best_model = pf_model  # Default to baseline
    best_metrics = pf_metrics
    
    # Test LightGBM
    try:
        from src.models import LightGBMRanker
        
        if LightGBMRanker is not None:
            lgb_model = LightGBMRanker(
                num_leaves=31,
                learning_rate=0.05,
                num_boost_round=100,
                early_stopping_rounds=20
            )
            lgb_model.fit(split_data.train_df, split_data.feature_names, split_data.test_df)
            lgb_metrics = evaluator.evaluate(lgb_model, split_data.test_df, split_data.feature_names)
            
            print_result("LightGBMRanker", True)
            print(f"         NDCG@3: {lgb_metrics['ndcg@3']:.4f}")
            results.append(("LightGBMRanker", True))
            
            if lgb_metrics['ndcg@3'] > best_metrics['ndcg@3']:
                best_model = lgb_model
                best_metrics = lgb_metrics
        else:
            print_result("LightGBMRanker", False, "Not installed")
            results.append(("LightGBMRanker", False))
            
    except ImportError:
        print_result("LightGBMRanker", False, "LightGBM not installed")
        results.append(("LightGBMRanker", False))
    except Exception as e:
        print_result("LightGBMRanker", False, str(e))
        results.append(("LightGBMRanker", False))
    
    # Test XGBoost
    try:
        from src.models import XGBoostRanker
        
        if XGBoostRanker is not None:
            xgb_model = XGBoostRanker(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                early_stopping_rounds=20
            )
            xgb_model.fit(split_data.train_df, split_data.feature_names, split_data.test_df)
            xgb_metrics = evaluator.evaluate(xgb_model, split_data.test_df, split_data.feature_names)
            
            print_result("XGBoostRanker", True)
            print(f"         NDCG@3: {xgb_metrics['ndcg@3']:.4f}")
            results.append(("XGBoostRanker", True))
            
            if xgb_metrics['ndcg@3'] > best_metrics['ndcg@3']:
                best_model = xgb_model
                best_metrics = xgb_metrics
        else:
            print_result("XGBoostRanker", False, "Not installed")
            results.append(("XGBoostRanker", False))
            
    except ImportError:
        print_result("XGBoostRanker", False, "XGBoost not installed")
        results.append(("XGBoostRanker", False))
    except Exception as e:
        print_result("XGBoostRanker", False, str(e))
        results.append(("XGBoostRanker", False))
    
    # Test CatBoost
    try:
        from src.models import CatBoostRanker
        
        if CatBoostRanker is not None:
            cb_model = CatBoostRanker(
                depth=6,
                learning_rate=0.1,
                iterations=100,
                early_stopping_rounds=20,
                verbose=0
            )
            cb_model.fit(split_data.train_df, split_data.feature_names, split_data.test_df)
            cb_metrics = evaluator.evaluate(cb_model, split_data.test_df, split_data.feature_names)
            
            print_result("CatBoostRanker", True)
            print(f"         NDCG@3: {cb_metrics['ndcg@3']:.4f}")
            results.append(("CatBoostRanker", True))
            
            if cb_metrics['ndcg@3'] > best_metrics['ndcg@3']:
                best_model = cb_model
                best_metrics = cb_metrics
        else:
            print_result("CatBoostRanker", False, "Not installed")
            results.append(("CatBoostRanker", False))
            
    except ImportError:
        print_result("CatBoostRanker", False, "CatBoost not installed")
        results.append(("CatBoostRanker", False))
    except Exception as e:
        print_result("CatBoostRanker", False, str(e))
        results.append(("CatBoostRanker", False))
    
    # ==========================================================================
    # TEST 10: MODEL SELECTION
    # ==========================================================================
    print_header("TEST 10: MODEL SELECTION")
    
    try:
        print_result("Best Model Selection", True)
        print(f"         Best: {best_model.name} (NDCG@3: {best_metrics['ndcg@3']:.4f})")
        results.append(("ModelSelection", True))
        
    except Exception as e:
        print_result("Best Model Selection", False, str(e))
        results.append(("ModelSelection", False))
    
    # ==========================================================================
    # TEST 11: HYPERPARAMETER TUNING (Tune the BEST model)
    # ==========================================================================
    print_header("TEST 11: HYPERPARAMETER TUNING")
    
    try:
        from src.training import HyperparameterTuner
        
        tuner = HyperparameterTuner(
            n_trials=3,  # Just 3 trials for testing
            timeout=60,
            metric='ndcg@3'
        )
        
        # Determine which model to tune based on best_model
        best_model_name = best_model.name
        tuning_result = None
        
        if 'LightGBM' in best_model_name:
            from src.models import LightGBMRanker
            if LightGBMRanker is not None:
                print(f"  Tuning LightGBM (best model)...")
                tuning_result = tuner.tune_lightgbm(split_data)
                
        elif 'XGBoost' in best_model_name:
            from src.models import XGBoostRanker
            if XGBoostRanker is not None:
                print(f"  Tuning XGBoost (best model)...")
                tuning_result = tuner.tune_xgboost(split_data)
                
        elif 'CatBoost' in best_model_name:
            from src.models import CatBoostRanker
            if CatBoostRanker is not None:
                print(f"  Tuning CatBoost (best model)...")
                tuning_result = tuner.tune_catboost(split_data)
        
        else:
            # Baseline model - no tuning needed, try LightGBM if available
            print(f"  Best model is baseline ({best_model_name}), trying to tune LightGBM...")
            try:
                from src.models import LightGBMRanker
                if LightGBMRanker is not None:
                    tuning_result = tuner.tune_lightgbm(split_data)
            except:
                pass
        
        if tuning_result is not None:
            assert tuning_result.best_params is not None, "No best params"
            assert tuning_result.best_score > 0, "Invalid best score"
            
            print_result(f"HyperparameterTuner.tune_{best_model_name.lower()}()", True)
            print(f"         Best NDCG@3: {tuning_result.best_score:.4f}")
            print(f"         Trials: {tuning_result.n_trials}")
            print(f"         Best Params: {tuning_result.best_params}")
            results.append(("HyperparameterTuner", True))
        else:
            print_result("HyperparameterTuner", False, "No ML model available for tuning")
            results.append(("HyperparameterTuner", False))
            
    except ImportError as e:
        print_result("HyperparameterTuner", False, f"Missing dependency: {e}")
        results.append(("HyperparameterTuner", False))
    except Exception as e:
        print_result("HyperparameterTuner", False, str(e))
        results.append(("HyperparameterTuner", False))
    
    # ==========================================================================
    # TEST 12: INFERENCE - RecommenderPredictor
    # ==========================================================================
    print_header("TEST 12: INFERENCE - RecommenderPredictor")
    
    try:
        from src.inference import RecommenderPredictor
        
        predictor = RecommenderPredictor(
            ml_model=best_model,
            baseline_model=pf_model,
            product_features=product_features,
            prepared_data=prepared_data
        )
        
        # Get a valid customer ID
        test_customer = list(prepared_data.customer_histories.keys())[0]
        
        prediction = predictor.recommend(customer_id=test_customer, top_k=5)
        
        assert prediction is not None, "No prediction"
        assert len(prediction.primary_items) > 0, "No recommendations"
        
        print_result("RecommenderPredictor.recommend()", True)
        print(f"         Customer {test_customer}: {len(prediction.primary_items)} items")
        results.append(("RecommenderPredictor", True))
        
    except Exception as e:
        print_result("RecommenderPredictor.recommend()", False, str(e))
        results.append(("RecommenderPredictor", False))
    
    # ==========================================================================
    # TEST 13: INFERENCE - ColdStartHandler
    # ==========================================================================
    print_header("TEST 13: INFERENCE - ColdStartHandler")
    
    try:
        from src.inference import ColdStartHandler
        
        cold_start = ColdStartHandler(
            product_features=product_features,
            prepared_data=prepared_data,
            customer_profiles=customer_profiles
        )
        
        recs = cold_start.recommend(
            archetype_hint="coffee_purist",
            time_of_day=9,
            top_k=5
        )
        
        assert recs is not None, "No recommendations"
        assert len(recs) > 0, "Empty recommendations"
        
        print_result("ColdStartHandler.recommend()", True)
        print(f"         {len(recs)} cold-start recommendations")
        results.append(("ColdStartHandler", True))
        
    except Exception as e:
        print_result("ColdStartHandler.recommend()", False, str(e))
        results.append(("ColdStartHandler", False))
    
    # ==========================================================================
    # TEST 14: MODEL SAVE/LOAD
    # ==========================================================================
    print_header("TEST 14: MODEL SAVE/LOAD")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        # Save
        model_bundle = {
            'model': best_model,
            'product_features': product_features,
            'metrics': best_metrics
        }
        
        with open(temp_path, 'wb') as f:
            pickle.dump(model_bundle, f)
        
        # Load
        with open(temp_path, 'rb') as f:
            loaded_bundle = pickle.load(f)
        
        assert 'model' in loaded_bundle, "Model not loaded"
        assert loaded_bundle['model'].name == best_model.name, "Model mismatch"
        
        # Cleanup
        os.remove(temp_path)
        
        print_result("Model Save/Load", True)
        results.append(("ModelSaveLoad", True))
        
    except Exception as e:
        print_result("Model Save/Load", False, str(e))
        results.append(("ModelSaveLoad", False))
    
    # ==========================================================================
    # TEST 15: MLFLOW TRACKING (Optional)
    # ==========================================================================
    print_header("TEST 15: MLFLOW TRACKING (Optional)")
    
    try:
        import mlflow
        
        # Create a temp directory that we'll clean up manually
        import shutil
        tmpdir = tempfile.mkdtemp()
        
        try:
            # Use SQLite database (recommended and works on all platforms)
            db_path = os.path.join(tmpdir, "mlflow.db")
            tracking_uri = f"sqlite:///{db_path}"
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("test-experiment")
            
            with mlflow.start_run(run_name="test-run"):
                mlflow.log_param("test_param", "value")
                mlflow.log_metric("test_metric", 0.95)
            
            print_result("MLflow Tracking", True)
            print(f"         Using SQLite backend (recommended)")
            results.append(("MLflow", True))
            
        finally:
            # Clean up - ignore errors on Windows due to file locking
            try:
                mlflow.end_run()  # Make sure run is ended
                shutil.rmtree(tmpdir, ignore_errors=True)
            except:
                pass  # Ignore cleanup errors on Windows
            
    except ImportError:
        print_result("MLflow Tracking", False, "MLflow not installed (optional)")
        results.append(("MLflow", False))
    except Exception as e:
        print_result("MLflow Tracking", False, str(e))
        results.append(("MLflow", False))
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)
    total = len(results)
    
    print(f"\n  Total Tests: {total}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Success Rate: {100*passed/total:.1f}%")
    
    print("\n  Detailed Results:")
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"    {status} {name}")
    
    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED! Your pipeline is ready.")
    else:
        print(f"\n  ⚠️  {failed} tests failed. Check the output above.")
    
    print("\n" + "=" * 70)
    
    return results


# ==============================================================================
# QUICK TESTS FOR CI/CD
# ==============================================================================

def run_quick_tests():
    """Run quick smoke tests for CI/CD pipeline."""
    print("\n🚀 Running Quick CI/CD Tests...\n")
    
    tests_passed = True
    
    # Test imports
    try:
        from src.data import DataLoader, DataValidator, DataPreprocessor
        from src.features import ProductFeatureExtractor, CustomerFeatureExtractor, TrainingDataBuilder
        from src.training import TemporalDataSplitter
        from src.models import PopularityRecommender, PersonalFrequencyRecommender
        from src.evaluation import RankingMetrics
        from src.inference import RecommenderPredictor, ColdStartHandler
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        tests_passed = False
    
    # Test model instantiation
    try:
        from src.models import get_available_models
        available = get_available_models()
        print(f"✅ Available models: {[name for name, _ in available]}")
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        tests_passed = False
    
    return tests_passed


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test complete pipeline')
    parser.add_argument('--data', type=str, default='./src/data/data_raw.csv')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (for CI/CD)')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    else:
        results = run_all_tests(data_path=args.data)
        
        # Exit with error code if any tests failed
        failed = sum(1 for _, p in results if not p)
        sys.exit(0 if failed == 0 else 1)
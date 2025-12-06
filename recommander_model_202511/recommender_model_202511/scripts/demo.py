"""
demo.py
=======

Demonstrates inference using the trained JR recommender.

This script loads:
 - Trained model (LightGBM/XGBoost/Base)
 - Product features
 - Customer profiles
 - PreparedData (required for feature construction)
 - ColdStartHandler and RecommenderPredictor

Runs 6 test cases:
 1. Known customer
 2. Customer with long purchase history
 3. Customer with short purchase history
 4. Completely unknown customer (cold start)
 5. Archetype-driven cold start (Latte Lover)
 6. Segment-driven scenario (e.g., VIP vs Regular)

Outputs clean, human-readable recommendations.
"""

import sys
from pathlib import Path

# Ensure src/ is importable regardless of execution location
ROOT = Path(__file__).resolve().parents[1]   # recommender_model_202511/
sys.path.insert(0, str(ROOT))

import json
import pickle
from pathlib import Path
import logging

from src.inference.recommender_predictor import RecommenderPredictor
from src.inference.cold_start import ColdStartHandler

# --- Artifact paths ---
ARTIFACT_DIR = Path("models/artifacts")

MODEL_PATH = ARTIFACT_DIR / "recommender.pkl"
INFO_PATH = ARTIFACT_DIR / "model_info.json"

PRODUCT_FEATURES_PATH = ARTIFACT_DIR / "product_features.pkl"
CUSTOMER_PROFILES_PATH = ARTIFACT_DIR / "customer_profiles.pkl"
PREPARED_DATA_PATH = ARTIFACT_DIR / "prepared_data.pkl"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("DEMO")


# ---------------------------------------------------------------------
# Utility printing
# ---------------------------------------------------------------------
def pretty_print(pred, title="Recommendation"):
    print(f"\n==================== {title} ====================")
    print(f"MODEL USED : {pred.model_used}")
    print(f"CUSTOMER   : {pred.customer_id}")
    print("-------------------------------------------------")
    print("PRIMARY RECOMMENDATIONS:")
    for i, item in enumerate(pred.primary_items, start=1):
        print(f"  {i}. {item.product:30s} score={item.score:.4f}  reason={item.reason}")

    if pred.addon_items:
        print("\nADD-ON SUGGESTIONS:")
        for i, item in enumerate(pred.addon_items, start=1):
            print(f"  + {item.product:30s} score={item.score:.4f}  reason={item.reason}")

    print("=================================================\n")


# ---------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    model_bundle = pickle.load(open(MODEL_PATH, "rb"))
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    product_features = pickle.load(open(PRODUCT_FEATURES_PATH, "rb"))
    customer_profiles = pickle.load(open(CUSTOMER_PROFILES_PATH, "rb"))
    prepared_data = pickle.load(open(PREPARED_DATA_PATH, "rb"))

    model_info = json.load(open(INFO_PATH, "r"))

    logger.info("Loaded trained model: %s", model_info["model_name"])
    logger.info("VALID NDCG@3: %.4f", model_info["valid_metrics"]["ndcg@3"])
    logger.info("TEST  NDCG@3: %.4f", model_info["test_metrics"]["ndcg@3"])

    return model, feature_names, product_features, customer_profiles, prepared_data


# ---------------------------------------------------------------------
# Main demo execution
# ---------------------------------------------------------------------
def main():
    (
        ml_model,
        feature_names,
        product_features,
        customer_profiles,
        prepared_data,
    ) = load_artifacts()

    # Baseline disabled
    baseline_model = None

    cold_handler = ColdStartHandler(
        product_features=product_features,
        prepared_data=prepared_data,
        customer_profiles=customer_profiles,
    )

    predictor = RecommenderPredictor(
        ml_model=ml_model,
        baseline_model=baseline_model,
        product_features=product_features,
        prepared_data=prepared_data,
        customer_profiles=customer_profiles,
        feature_names=feature_names,
        cold_start_handler=cold_handler,
    )

    # ---------------------------------------------
    # TEST CASE 1 — Known Customer
    # ---------------------------------------------
    known_customer = next(iter(prepared_data.customer_histories.keys()))
    pred1 = predictor.recommend(known_customer, top_k=5)
    pretty_print(pred1, "CASE 1 — Known Customer")

    # ---------------------------------------------
    # TEST CASE 2 — Customer with LONG purchase history
    # ---------------------------------------------
    long_history_customer = max(
        prepared_data.customer_histories.keys(),
        key=lambda cid: len(prepared_data.customer_histories[cid]),
    )
    pred2 = predictor.recommend(long_history_customer, top_k=5)
    pretty_print(pred2, "CASE 2 — Long History Customer")

    # ---------------------------------------------
    # TEST CASE 3 — Customer with SHORT purchase history
    # ---------------------------------------------
    short_history_customer = min(
        prepared_data.customer_histories.keys(),
        key=lambda cid: len(prepared_data.customer_histories[cid]),
    )
    pred3 = predictor.recommend(short_history_customer, top_k=5)
    pretty_print(pred3, "CASE 3 — Short History Customer")

    # ---------------------------------------------
    # TEST CASE 4 — Unknown customer (Cold Start)
    # ---------------------------------------------
    unknown_customer = 99999999
    pred4 = predictor.recommend(unknown_customer, top_k=5)
    pretty_print(pred4, "CASE 4 — Cold Start (Unknown Customer)")

    # ---------------------------------------------
    # TEST CASE 5 — Archetype-specific Cold Start
    # ---------------------------------------------
    latte_lover_recs = cold_handler.recommend(
        archetype_hint="latte_lover",
        time_of_day=9,
        top_k=5,
    )
    print("\n===== CASE 5 — Archetype Cold Start: LATTE LOVER =====")
    for item in latte_lover_recs:
        print(f"- {item.product:30s} score={item.score:.4f} reason={item.reason}")

    # ---------------------------------------------
    # TEST CASE 6 — Segment-driven recommendations
    # ---------------------------------------------
    # Pick any customer and manually override their segment for demonstration
    seg_customer = known_customer
    prepared_data.customer_histories[seg_customer][0]["segment"] = "VIP"

    pred6 = predictor.recommend(seg_customer, top_k=5)
    pretty_print(pred6, "CASE 6 — Segment-driven Recommendation (VIP)")

    print("\nAll demo test cases completed successfully.")


if __name__ == "__main__":
    main()

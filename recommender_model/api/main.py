"""
================================================================================
CAFE RECOMMENDER API SERVICE
================================================================================

Production-ready FastAPI service for the recommendation system.

Features:
- REST API endpoints for recommendations
- Cold-start handling for new customers
- Health checks for load balancers
- Model versioning and metadata
- Request validation with Pydantic
- Async support for high throughput
- CORS for web integrations

Run locally:
    uvicorn api.main:app --reload --port 8000

Run in production:
    gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

import os
import pickle
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ==============================================================================

class RecommendationItem(BaseModel):
    """Single recommendation item."""
    product: str = Field(..., description="Product name")
    score: float = Field(..., description="Recommendation score (0-1)")
    reason: str = Field(..., description="Why this was recommended")


class RecommendationResponse(BaseModel):
    """Response for recommendation request."""
    customer_id: int
    recommendations: List[RecommendationItem]
    model_used: str
    generated_at: str
    latency_ms: float


class ColdStartRequest(BaseModel):
    """Request for cold-start (new customer) recommendations."""
    archetype_hint: Optional[str] = Field(
        None, 
        description="Customer type hint: parent, coffee_purist, latte_lover, health_conscious, food_focused, casual"
    )
    time_of_day: Optional[int] = Field(
        None, 
        ge=0, 
        le=23, 
        description="Hour of day (0-23)"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations")


class ColdStartResponse(BaseModel):
    """Response for cold-start recommendations."""
    recommendations: List[RecommendationItem]
    archetype_used: Optional[str]
    generated_at: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    model_metrics: Optional[Dict[str, float]]
    timestamp: str
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    training_date: Optional[str]
    metrics: Dict[str, float]
    n_customers: int
    n_products: int
    feature_count: int
    version: str


# ==============================================================================
# GLOBAL STATE
# ==============================================================================

class AppState:
    """Application state container."""
    def __init__(self):
        self.model = None
        self.baseline_model = None
        self.product_features = None
        self.prepared_data = None
        self.customer_profiles = None
        self.feature_names = None
        self.metrics = None
        self.model_name = None
        self.training_date = None
        self.is_loaded = False


app_state = AppState()


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(model_path: str = None):
    """
    Load the trained model bundle.
    
    Args:
        model_path: Path to model pickle file
    """
    if model_path is None:
        # Check environment variable first
        model_path = os.environ.get('MODEL_PATH', 'models/artifacts/recommender.pkl')
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        
        app_state.model = bundle.get('model')
        app_state.baseline_model = bundle.get('baseline_model', bundle.get('model'))
        app_state.product_features = bundle.get('product_features')
        app_state.prepared_data = bundle.get('prepared_data')
        app_state.customer_profiles = bundle.get('customer_profiles', {})
        app_state.feature_names = bundle.get('feature_names', [])
        app_state.metrics = bundle.get('metrics', {})
        app_state.model_name = bundle.get('model_name', 'unknown')
        app_state.training_date = bundle.get('training_date')
        app_state.is_loaded = True
        
        logger.info(f"Model loaded successfully: {app_state.model_name}")
        logger.info(f"Metrics: NDCG@3={app_state.metrics.get('ndcg@3', 'N/A')}")
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ==============================================================================
# LIFESPAN (Startup/Shutdown)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    # Startup
    logger.info("Starting Cafe Recommender API...")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Model not loaded at startup: {e}")
        logger.warning("API will start but recommendations won't work until model is loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cafe Recommender API...")


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Cafe Recommender API",
    description="ML-powered product recommendations for cafe customers",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - basic info."""
    return {
        "service": "Cafe Recommender API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
        Health status and model information
    """
    return HealthResponse(
        status="healthy" if app_state.is_loaded else "degraded",
        model_loaded=app_state.is_loaded,
        model_name=app_state.model_name if app_state.is_loaded else None,
        model_metrics=app_state.metrics if app_state.is_loaded else None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model metadata, metrics, and configuration
    """
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=app_state.model_name,
        training_date=app_state.training_date,
        metrics={k: round(v, 4) for k, v in app_state.metrics.items()},
        n_customers=app_state.prepared_data.n_customers if app_state.prepared_data else 0,
        n_products=app_state.prepared_data.n_products if app_state.prepared_data else 0,
        feature_count=len(app_state.feature_names),
        version="1.0.0"
    )


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    customer_id: int,
    top_k: int = Query(5, ge=1, le=20, description="Number of recommendations"),
    include_addons: bool = Query(True, description="Include add-on suggestions")
):
    """
    Get personalized recommendations for an existing customer.
    
    Args:
        customer_id: Customer ID from your system
        top_k: Number of recommendations to return (1-20)
        include_addons: Whether to include add-on product suggestions
    
    Returns:
        List of recommended products with scores and reasons
    """
    import time
    start_time = time.time()
    
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check if customer exists
    if customer_id not in app_state.prepared_data.customer_histories:
        raise HTTPException(
            status_code=404, 
            detail=f"Customer {customer_id} not found. Use /recommend/cold-start for new customers."
        )
    
    try:
        # Generate recommendations using baseline model (simpler, faster)
        recommendations = _generate_recommendations(customer_id, top_k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            customer_id=customer_id,
            recommendations=recommendations,
            model_used=app_state.model_name,
            generated_at=datetime.now().isoformat(),
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/cold-start", response_model=ColdStartResponse, tags=["Recommendations"])
async def get_cold_start_recommendations(request: ColdStartRequest):
    """
    Get recommendations for a new customer (no purchase history).
    
    Use archetype hints to personalize:
    - parent: Families with children
    - coffee_purist: Espresso/long black lovers
    - latte_lover: Latte and milk coffee fans
    - health_conscious: Smoothies, juices, healthy options
    - food_focused: Breakfast/lunch items
    - casual: General customers (default)
    
    Args:
        request: Cold start request with optional hints
    
    Returns:
        List of recommended products for new customers
    """
    import time
    start_time = time.time()
    
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        recommendations = _generate_cold_start_recommendations(
            archetype_hint=request.archetype_hint,
            time_of_day=request.time_of_day,
            top_k=request.top_k
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ColdStartResponse(
            recommendations=recommendations,
            archetype_used=request.archetype_hint or "casual",
            generated_at=datetime.now().isoformat(),
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Error generating cold-start recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers", tags=["Data"])
async def list_customers(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    List available customer IDs (for testing/debugging).
    
    Args:
        limit: Maximum number of customers to return
        offset: Offset for pagination
    """
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    customer_ids = list(app_state.prepared_data.customer_histories.keys())
    total = len(customer_ids)
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "customer_ids": customer_ids[offset:offset + limit]
    }


@app.get("/products", tags=["Data"])
async def list_products(limit: int = Query(100, ge=1, le=500)):
    """
    List available products.
    
    Args:
        limit: Maximum number of products to return
    """
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    products = app_state.prepared_data.product_list[:limit]
    
    return {
        "total": len(app_state.prepared_data.product_list),
        "products": products
    }


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _generate_recommendations(customer_id: int, top_k: int) -> List[RecommendationItem]:
    """Generate recommendations for existing customer."""
    
    # Get customer history
    history = app_state.prepared_data.customer_histories[customer_id]
    past_products = set()
    for order in history:
        for product in order.get('basket', []):
            if isinstance(product, str):
                past_products.add(product)
    
    # Score all products
    products = app_state.prepared_data.product_list
    scores = {}
    
    for product in products:
        if not isinstance(product, str):
            continue
        
        # Use baseline model for scoring
        score = app_state.baseline_model.predict(customer_id, [product], None)[0]
        scores[product] = score
    
    # Sort and get top-k
    sorted_products = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    
    recommendations = []
    for product, score in sorted_products:
        reason = "Based on your order history" if product in past_products else "Popular choice you might like"
        
        recommendations.append(RecommendationItem(
            product=product,
            score=round(score, 4),
            reason=reason
        ))
    
    return recommendations


def _generate_cold_start_recommendations(
    archetype_hint: Optional[str],
    time_of_day: Optional[int],
    top_k: int
) -> List[RecommendationItem]:
    """Generate recommendations for new customer."""
    
    popularity = app_state.product_features.popularity
    
    # Start with popularity scores
    scores = {p: pop * 0.5 for p, pop in popularity.items() if isinstance(p, str)}
    
    # Apply archetype boost if provided
    archetype_boosts = {
        'parent': ['babychino', 'kids', 'fluffy', 'babyccino'],
        'coffee_purist': ['espresso', 'long black', 'ristretto', 'macchiato'],
        'latte_lover': ['latte', 'flat white', 'cappuccino'],
        'health_conscious': ['smoothie', 'juice', 'acai', 'green'],
        'food_focused': ['toast', 'breakfast', 'lunch', 'sandwich', 'roll']
    }
    
    if archetype_hint and archetype_hint.lower() in archetype_boosts:
        boost_keywords = archetype_boosts[archetype_hint.lower()]
        for product in scores:
            if any(kw in product.lower() for kw in boost_keywords):
                scores[product] *= 1.5
    
    # Apply time-of-day boost
    if time_of_day is not None:
        for product in list(scores.keys()):
            product_lower = product.lower()
            
            # Morning boost
            if 6 <= time_of_day <= 11:
                if any(kw in product_lower for kw in ['coffee', 'latte', 'cappuccino', 'espresso']):
                    scores[product] *= 1.3
                if any(kw in product_lower for kw in ['toast', 'breakfast', 'croissant']):
                    scores[product] *= 1.2
            
            # Lunch boost
            elif 11 <= time_of_day <= 14:
                if any(kw in product_lower for kw in ['sandwich', 'roll', 'lunch', 'salad']):
                    scores[product] *= 1.3
            
            # Afternoon boost
            elif 14 <= time_of_day <= 17:
                if any(kw in product_lower for kw in ['iced', 'cold', 'frappe', 'smoothie']):
                    scores[product] *= 1.2
    
    # Sort and return
    sorted_products = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    
    recommendations = []
    for product, score in sorted_products:
        if archetype_hint:
            reason = f"Popular with {archetype_hint} customers"
        else:
            reason = "Popular choice for new customers"
        
        recommendations.append(RecommendationItem(
            product=product,
            score=round(score, 4),
            reason=reason
        ))
    
    return recommendations


# ==============================================================================
# MAIN (for direct execution)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
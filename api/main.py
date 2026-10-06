from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from src.config import (
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METADATA_PATH,
        METRICS_PATH,
        LEAD_TIER_HIGH_THRESHOLD,
        LEAD_TIER_MEDIUM_THRESHOLD,
    )
    from src.feature_engineering import engineer_features
except ModuleNotFoundError:
    from config import (
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METADATA_PATH,
        METRICS_PATH,
        LEAD_TIER_HIGH_THRESHOLD,
        LEAD_TIER_MEDIUM_THRESHOLD,
    )
    from feature_engineering import engineer_features

# Lazy/Global model and preprocessor references
_MODEL = None
_PREPROCESSOR = None
_METADATA = None
_METRICS = None


def get_artifacts():
    global _MODEL, _PREPROCESSOR, _METADATA, _METRICS
    if _MODEL is None or _PREPROCESSOR is None:
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise RuntimeError("Model or preprocessor artifacts not found. Please train models first.")
        _MODEL = joblib.load(MODEL_PATH)
        _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _METADATA = json.load(f)
        else:
            _METADATA = {"model_name": "LightGBM", "optimal_threshold": 0.69}

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                _METRICS = json.load(f)
        else:
            _METRICS = {}

    return _MODEL, _PREPROCESSOR, _METADATA, _METRICS


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload ML artifacts during container startup
    get_artifacts()
    yield


# Initialize FastAPI App
app = FastAPI(
    title="Bank Lead Conversion Scoring API",
    description="Production machine learning API for telemarketing lead qualification and conversion propensity scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class LeadInput(BaseModel):
    age: int = Field(..., ge=18, le=120, example=35, description="Client age in years")
    job: str = Field(..., example="management", description="Job type (e.g., admin, blue-collar, management)")
    marital: str = Field(..., example="married", description="Marital status (married, single, divorced)")
    education: str = Field(..., example="tertiary", description="Education level (primary, secondary, tertiary, unknown)")
    default: str = Field(..., example="no", description="Credit in default (no, yes)")
    balance: int = Field(..., example=1500, description="Average yearly balance in euros")
    housing: str = Field(..., example="no", description="Has housing loan (no, yes)")
    loan: str = Field(..., example="no", description="Has personal loan (no, yes)")
    contact: str = Field(..., example="cellular", description="Contact communication type (cellular, telephone, unknown)")
    day: int = Field(..., ge=1, le=31, example=15, description="Last contact day of the month")
    month: str = Field(..., example="may", description="Last contact month (jan-dec)")
    campaign: int = Field(..., ge=1, example=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., ge=-1, example=-1, description="Days passed since previous campaign contact (-1 = never)")
    previous: int = Field(..., ge=0, example=0, description="Number of contacts prior to this campaign")
    poutcome: str = Field(..., example="unknown", description="Outcome of previous marketing campaign (failure, success, other, unknown)")
    duration: Optional[int] = Field(None, ge=0, example=200, description="Optional call duration (not used in pre-call scoring)")


class BatchLeadInput(BaseModel):
    leads: List[LeadInput]


class LeadPrediction(BaseModel):
    conversion_probability: float
    will_convert: bool
    lead_tier: str
    decision_threshold: float
    recommendation: str


class BatchPredictionResponse(BaseModel):
    total_leads: int
    high_tier_count: int
    medium_tier_count: int
    low_tier_count: int
    mean_conversion_probability: float
    predictions: List[LeadPrediction]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    champion_model: str
    timestamp_utc: str


def compute_lead_tier(probability: float) -> str:
    if probability >= LEAD_TIER_HIGH_THRESHOLD:
        return "High"
    elif probability >= LEAD_TIER_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def generate_recommendation(tier: str, will_convert: bool) -> str:
    if tier == "High":
        return "High Priority: Contact immediately with premium deposit offering."
    elif tier == "Medium":
        return "Medium Priority: Queue for standard telemarketing campaign outreach."
    return "Low Priority: Deprioritize outbound dialing to conserve call center bandwidth."


def score_leads(leads: List[LeadInput]) -> List[LeadPrediction]:
    model, preprocessor, metadata, _ = get_artifacts()
    threshold = float(metadata.get("optimal_threshold", 0.69))

    # Convert to DataFrame
    df = pd.DataFrame([lead.model_dump() for lead in leads])

    # Feature Engineering
    df_feat = engineer_features(df)

    # Preprocessing
    X_trans = preprocessor.transform(df_feat)

    # Inference
    probabilities = model.predict_proba(X_trans)[:, 1]

    results = []
    for prob in probabilities:
        p_val = round(float(prob), 4)
        will_conv = bool(p_val >= threshold)
        tier = compute_lead_tier(p_val)
        rec = generate_recommendation(tier, will_conv)

        results.append(
            LeadPrediction(
                conversion_probability=p_val,
                will_convert=will_conv,
                lead_tier=tier,
                decision_threshold=threshold,
                recommendation=rec,
            )
        )
    return results


# API Routes
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Bank Lead Conversion Scoring API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "model_info_url": "/model-info",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    try:
        model, _, metadata, _ = get_artifacts()
        return HealthResponse(
            status="healthy",
            model_loaded=model is not None,
            champion_model=str(metadata.get("model_name", "Unknown")),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    try:
        _, _, metadata, metrics = get_artifacts()
        return {
            "metadata": metadata,
            "metrics": metrics,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading model info: {str(e)}",
        )


@app.post("/predict", response_model=LeadPrediction, tags=["Inference"])
def predict_single_lead(lead: LeadInput):
    try:
        predictions = score_leads([lead])
        return predictions[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_leads(batch: BatchLeadInput):
    try:
        if not batch.leads:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch leads list cannot be empty.",
            )

        predictions = score_leads(batch.leads)
        total = len(predictions)
        high_cnt = sum(1 for p in predictions if p.lead_tier == "High")
        med_cnt = sum(1 for p in predictions if p.lead_tier == "Medium")
        low_cnt = sum(1 for p in predictions if p.lead_tier == "Low")
        mean_prob = round(float(sum(p.conversion_probability for p in predictions) / total), 4)

        return BatchPredictionResponse(
            total_leads=total,
            high_tier_count=high_cnt,
            medium_tier_count=med_cnt,
            low_tier_count=low_cnt,
            mean_conversion_probability=mean_prob,
            predictions=predictions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)

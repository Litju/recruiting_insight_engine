"""
API Routes — Recruiting Insight Engine (Phase 1)
Clean, stable, aligned with the new pipeline + engine.
"""

from fastapi import APIRouter
from app.schemas.candidate import CandidateFeatures
from app.ml.inference import SalaryPredictor
from app.insights.engine import InsightEngine

router = APIRouter()

# ---------------------------------------------------------
# SINGLETONS — created once, reused across requests
# ---------------------------------------------------------
predictor = SalaryPredictor()
engine = InsightEngine()


def _model_dump_with_aliases(model: CandidateFeatures) -> dict:
    """
    Compat helper for Pydantic v1/v2 to emit alias field names.
    """
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True)
    return model.dict(by_alias=True)


# ---------------------------------------------------------
# /api/predict
# ---------------------------------------------------------
@router.post("/predict", summary="Predict Salary", tags=["prediction"])
def predict_salary(payload: CandidateFeatures):
    """
    Predict salary based on the A-MINIMAL feature set.
    """
    features = _model_dump_with_aliases(payload)
    pred = predictor.predict_one(features)

    return {
        "status": "ok",
        "prediction": pred,
        "currency": "USD",
        "features_used": features,
        "model": "RandomForestRegressor",
    }


# ---------------------------------------------------------
# /api/insights
# ---------------------------------------------------------
@router.post("/insights", summary="Generate HR Insights", tags=["insights"])
def generate_insights(payload: CandidateFeatures):
    """
    Full Insight Bundle:
        - salary prediction
        - global market comparison
        - cohort comparisons (job/education/gender)
        - flags
    """
    features = _model_dump_with_aliases(payload)
    bundle = engine.generate_insights(features)
    return bundle

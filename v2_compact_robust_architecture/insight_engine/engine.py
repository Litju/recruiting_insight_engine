# ============================
# FILE: insight_engine/engine.py
# ============================

"""
Insight Engine Orchestrator (V2)

- Loads model pipeline
- Predicts salary
- Computes offer/market band
- Calls submodules for:
    - drivers
    - cohorts
    - fairness
    - narrative
- Returns a full insight bundle for the API layer.
"""

from typing import Dict, Any

from model.inference import load_pipeline, predict_one
from insight_engine.drivers import compute_drivers
from insight_engine.cohorts import cohort_analysis
from insight_engine.fairness import compute_fairness_summary
from insight_engine.narrative import generate_narrative


def _classify_offer_band(predicted_salary: float) -> str:
    """
    Simple offer band classifier to keep logic explicit and readable.
    """
    if predicted_salary < 60000:
        return "Below Market"
    if predicted_salary < 100000:
        return "Market Average"
    return "Above Market"


def generate_insights(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the complete insight bundle for a single candidate.

    Parameters
    ----------
    candidate : dict
        {
            "Age": int,
            "Gender": str,
            "Education Level": str,
            "Job Title": str,
            "Years of Experience": float
        }

    Returns
    -------
    dict:
        {
            "prediction": float,
            "offer_band": str,
            "market_band": str,
            "drivers": List[dict],
            "cohort": dict,
            "fairness": dict,
            "narrative": str,
            "mhi": dict,
            "raw_json": dict
        }
    """
    # Load trained pipeline
    pipeline = load_pipeline()

    # Predict
    predicted_salary = float(predict_one(pipeline, candidate))
    offer_band = _classify_offer_band(predicted_salary)
    market_band = offer_band  # same notion in this simple implementation

    # Submodules
    drivers = compute_drivers(pipeline, candidate)
    cohort_info = cohort_analysis(candidate)
    fairness = compute_fairness_summary(candidate)
    narrative = generate_narrative(
        predicted_salary=predicted_salary,
        drivers=drivers,
        cohort_info=cohort_info,
        fairness=fairness,
    )

    # MHI placeholder at inference time
    mhi_result = {
        "zone": "GREEN",
        "MHI": 1.0,
        "Gate": 1,
        "Core": 1.0,
        "Refinement": 1.0,
    }

    # Bundle
    return {
        "prediction": predicted_salary,
        "offer_band": offer_band,
        "market_band": market_band,
        "drivers": drivers,
        "cohort": cohort_info,
        "fairness": fairness,
        "narrative": narrative,
        "mhi": mhi_result,
        "raw_json": {
            "input": candidate,
            "prediction": predicted_salary,
            "offer_band": offer_band,
            "market_band": market_band,
            "drivers": drivers,
            "cohort": cohort_info,
            "fairness": fairness,
        },
    }

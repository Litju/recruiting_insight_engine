# ============================
# FILE: model/inference.py
# ============================

"""
Model inference utilities for V2.
Provides:
- load_pipeline()
- predict_one()
- predict_many()

Light inference mode:
- Uses preprocessor + model artifacts
- Accepts Python dicts from the API
"""

import os
import sys
from typing import Dict, List, Any

import joblib
import pandas as pd

# ------------------------------------------------------------
# PROJECT ROOT & ARTIFACT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "model", "artifacts")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")


# ------------------------------------------------------------
# LOADING ARTIFACTS
# ------------------------------------------------------------

def load_pipeline() -> Dict[str, Any]:
    """
    Load preprocessor + model artifacts.

    Returns
    -------
    dict:
        {
            "preprocessor": fitted ColumnTransformer,
            "model": fitted RandomForestRegressor
        }
    """
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "❌ Model artifacts not found. "
            "Make sure you have run `python model/train.py` first."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    return {
        "preprocessor": preprocessor,
        "model": model,
    }


# ------------------------------------------------------------
# PREDICT ONE
# ------------------------------------------------------------

def predict_one(pipeline: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """
    Predict salary for a single candidate.

    Parameters
    ----------
    pipeline : dict
        Output of load_pipeline().
    candidate : dict
        {
            "Age": ...,
            "Gender": ...,
            "Education Level": ...,
            "Job Title": ...,
            "Years of Experience": ...
        }

    Returns
    -------
    float
        Predicted salary.
    """
    # ✅ Convert dict → 1-row DataFrame (sklearn-friendly)
    df = pd.DataFrame([candidate])

    pre = pipeline["preprocessor"]
    model = pipeline["model"]

    X = pre.transform(df)
    y_pred = model.predict(X)

    return float(y_pred[0])


# ------------------------------------------------------------
# PREDICT MANY
# ------------------------------------------------------------

def predict_many(pipeline: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[float]:
    """
    Predict salary for multiple candidates.

    Parameters
    ----------
    pipeline : dict
        Output of load_pipeline().
    candidates : list[dict]

    Returns
    -------
    list[float]
    """
    df = pd.DataFrame(candidates)

    pre = pipeline["preprocessor"]
    model = pipeline["model"]

    X = pre.transform(df)
    y_pred = model.predict(X)

    return [float(v) for v in y_pred]

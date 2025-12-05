"""
Inference utilities for the Recruiting Insight Engine (Phase 1 — Core Model).

- Loads the persisted pipeline from app/ml/artifacts.
- Enforces STRICT 5-column input to avoid Swagger/browser garbage.
- Provides predict_one() and predict_many().
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Union

import joblib
import pandas as pd

# ---------------------------------------------------------
# ARTIFACT PATHS
# ---------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = THIS_DIR / "artifacts"

PREPROCESSOR_FILE = ARTIFACT_DIR / "preprocessor.pkl"
MODEL_FILE = ARTIFACT_DIR / "model.pkl"


class SalaryPredictor:
    """
    Thin wrapper around the trained pipeline artifacts.
    Responsibilities:
        - Load pipeline.
        - Clean and normalize input.
        - Enforce STRICT 5-feature schema.
        - Provide predict_one() and predict_many().
    """

    FEATURE_COLUMNS = [
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience",
    ]

    def __init__(
        self,
        preprocessor_path: Path = PREPROCESSOR_FILE,
        model_path: Path = MODEL_FILE,
    ) -> None:
        if not preprocessor_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Missing ML artifacts. Run training first:\n"
                "    python app/ml/train_pipeline.py"
            )

        # Load full pipeline (preprocessor is the first step inside)
        self.model = joblib.load(model_path)

        # Optional reference to the preprocessor for downstream uses
        self.preprocessor = None
        if hasattr(self.model, "named_steps"):
            self.preprocessor = self.model.named_steps.get("preprocessor")

    # ---------------------------------------------------------
    # INTERNAL: Normalize DataFrame with STRICT columns
    # ---------------------------------------------------------
    def _to_frame(
        self,
        data: Union[Dict, pd.Series, pd.DataFrame, List[Dict]]
    ) -> pd.DataFrame:

        # Convert to DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, pd.Series):
            df = data.to_frame().T
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

        # STRICT FEATURE FILTER: drop ALL extra fields
        df = df.reindex(columns=self.FEATURE_COLUMNS)

        return df

    # ---------------------------------------------------------
    # PUBLIC: predict single candidate
    # ---------------------------------------------------------
    def predict_one(self, features: Dict) -> float:
        df = self._to_frame(features)  # 5 clean columns
        pred = self.model.predict(df)
        return float(pred[0])

    # ---------------------------------------------------------
    # PUBLIC: batch prediction
    # ---------------------------------------------------------
    def predict_many(self, features_list: List[Dict]) -> List[float]:
        df = self._to_frame(features_list)
        preds = self.model.predict(df)
        return [float(p) for p in preds]


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    print("→ Loading SalaryPredictor...")
    predictor = SalaryPredictor()

    example = {
        "Age": 32,
        "Gender": "Male",
        "Education Level": "Bachelor's",
        "Job Title": "Software Engineer",
        "Years of Experience": 5,
    }

    print("\n→ Example input:")
    for k, v in example.items():
        print(f"  {k}: {v}")

    out = predictor.predict_one(example)
    print(f"\nPredicted Salary: {out:.2f}")

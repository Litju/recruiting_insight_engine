# ============================
# FILE: model/artifacts.py
# ============================
"""
artifacts.py

Single source of truth for saving and loading the trained model pipeline.

Responsibilities:
- Define artifact directory
- Save pipeline to disk
- Load pipeline from disk

We store a single sklearn Pipeline object which already includes:
- preprocessing (ColumnTransformer)
- estimator (RandomForestRegressor)
"""

import os
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline  # type: ignore


# Default artifact directory (relative to this file)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
PIPELINE_FILENAME = "model_pipeline.pkl"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_artifact_path(artifact_dir: Optional[str] = None) -> str:
    """
    Resolve the full path to the pipeline artifact file.
    """
    directory = artifact_dir or ARTIFACT_DIR
    _ensure_dir(directory)
    return os.path.join(directory, PIPELINE_FILENAME)


def save_artifacts(pipeline: Pipeline, artifact_dir: Optional[str] = None) -> str:
    """
    Save the trained sklearn Pipeline to disk.
    Returns the path where it was saved.
    """
    path = get_artifact_path(artifact_dir)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(artifact_dir: Optional[str] = None) -> Pipeline:
    """
    Load the sklearn Pipeline from disk.
    Raises FileNotFoundError if the artifact is missing.
    """
    path = get_artifact_path(artifact_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model pipeline artifact not found at: {path}")
    pipeline: Pipeline = joblib.load(path)
    return pipeline

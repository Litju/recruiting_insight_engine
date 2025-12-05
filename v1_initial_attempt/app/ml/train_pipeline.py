"""
Recruiting Insight Engine — Training Pipeline (Fixed)
- Loads raw CSVs (people, salary, descriptions)
- Merges correctly on 'id' (inner join)
- Validates schema and missing values
- Builds preprocessing (impute + OneHotEncoder)
- Trains RandomForestRegressor (n_estimators=300, random_state=42)
- Evaluates (MAE, RMSE, R^2)
- Saves artifacts to app/ml/artifacts/
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "app" / "ml" / "artifacts"

PEOPLE_CSV = RAW_DIR / "people.csv"
SALARY_CSV = RAW_DIR / "salary.csv"
DESCR_CSV = RAW_DIR / "descriptions.csv"
MERGED_OUT = PROC_DIR / "merged.csv"

PROC_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------
FEATURES = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience",
]
TARGET = "Salary"

NUMERIC = ["Age", "Years of Experience"]
CATEGORICAL = ["Gender", "Education Level", "Job Title"]


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def _check_paths() -> None:
    missing = [p for p in [PEOPLE_CSV, SALARY_CSV, DESCR_CSV] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")


def _coerce_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError(f"Column 'id' missing in {name}")
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    if df["id"].isna().any():
        raise ValueError(f"'id' contains non-numeric values in {name}")
    return df


def _print_debug(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} HEAD ===")
    print(df.head())
    print(f"\n=== {name} INFO ===")
    print(df.info())


def _validate_merged(df: pd.DataFrame) -> None:
    print("\nMerged shape:", df.shape)
    print("\nMissing values per column:\n", df.isna().sum())

    for col in FEATURES:
        if df[col].isna().all():
            raise ValueError(f"All values missing for feature '{col}' after merge; check raw data and join keys.")
    if df[TARGET].isna().all():
        raise ValueError("All target values are missing after merge.")

    before = len(df)
    df.dropna(subset=[TARGET], inplace=True)
    after = len(df)
    if after < before:
        print(f"Warning: Dropped {before - after} rows with missing target '{TARGET}'.")


# ---------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC),
            ("cat", categorical_pipeline, CATEGORICAL),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------
def main() -> None:
    _check_paths()

    # Load raw data
    people = pd.read_csv(PEOPLE_CSV)
    salary = pd.read_csv(SALARY_CSV)
    descr = pd.read_csv(DESCR_CSV)

    # Debug prints
    _print_debug(people, "people.csv")
    _print_debug(salary, "salary.csv")
    _print_debug(descr, "descriptions.csv")

    # Coerce id
    people = _coerce_id(people, "people.csv")
    salary = _coerce_id(salary, "salary.csv")
    descr = _coerce_id(descr, "descriptions.csv")

    # Merge (inner join to guarantee aligned rows)
    df = (
        people
        .merge(salary, on="id", how="inner")
        .merge(descr, on="id", how="inner")
    )

    # Validate
    _validate_merged(df)

    # Keep only required columns for training
    train_df = df[FEATURES + [TARGET]].copy()

    # Save merged for downstream insights (includes all cols)
    df.to_csv(MERGED_OUT, index=False)
    print(f"\nMerged dataset saved to {MERGED_OUT}")

    # Train-test split
    X = train_df[FEATURES]
    y = train_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Build pipeline
    preprocessor = build_preprocessor()
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
        )),
    ])

    # Fit
    print("\nFitting model...")
    model.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    # Save artifacts
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.pkl")
    joblib.dump(model, ARTIFACT_DIR / "model.pkl")
    print(f"\nArtifacts saved to {ARTIFACT_DIR}")
    print("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

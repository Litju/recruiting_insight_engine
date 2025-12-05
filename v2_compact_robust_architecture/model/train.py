"""
Model Training Pipeline (V2)
- Loads raw datasets
- Merges via Data Integrity Layer
- Computes KPIs + MHI
- Gates training on MHI
- Cleans missing targets
- Builds preprocessing pipeline
- Trains RandomForestRegressor
- Saves *full pipeline* as artifact
"""

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# ENSURE PROJECT ROOT IS IMPORTABLE
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(">> PROJECT_ROOT injected into sys.path:", PROJECT_ROOT)

# ============================================================
# IMPORT INTERNAL MODULES
# ============================================================

from data_integrity.merge import merge_tables
from data_integrity.kpis import compute_merge_kpis
from data_integrity.mhi import compute_mhi
from data_integrity.diagnostics import basic_merge_diagnostics

from model.artifacts import save_artifacts


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PEOPLE_PATH = os.path.join(DATA_DIR, "people.csv")
SALARY_PATH = os.path.join(DATA_DIR, "salary.csv")
DESC_PATH = os.path.join(DATA_DIR, "descriptions.csv")

# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
]

TARGET_COLUMN = "Salary"


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_model():
    print("\n>> Loading raw datasets...")

    people_df = pd.read_csv(PEOPLE_PATH)
    salary_df = pd.read_csv(SALARY_PATH)
    desc_df = pd.read_csv(DESC_PATH)

    print(">> Merging datasets via Data Integrity Layer")
    merged = merge_tables(people_df, salary_df, desc_df)

    print(">> Computing KPIs...")
    kpis = compute_merge_kpis(people_df, salary_df, desc_df, merged)

    print(">> Diagnostics...")
    diagnostics = basic_merge_diagnostics(merged)
    print("Diagnostics:", diagnostics)

    print(">> Computing MHI (Merge Health Index)...")
    mhi = compute_mhi(kpis)
    print("\n=== MHI REPORT ===")
    print(mhi)
    print("===================")

    # ========================================================
    # MHI GATE
    # ========================================================

    if mhi["zone"] == "RED" or mhi["Gate"] == 0:
        raise RuntimeError(
            f"\n❌ TRAINING ABORTED: MHI={mhi['MHI']:.3f}, ZONE=RED.\n"
            "Fix merge issues before training."
        )

    print(">> MHI PASSED (YELLOW/GREEN). Continuing training...\n")

    # ========================================================
    # REMOVE ROWS WITH MISSING TARGET
    # ========================================================

    print(">> Cleaning dataset: removing rows with missing Salary")

    clean = merged.dropna(subset=[TARGET_COLUMN])

    if len(clean) == 0:
        raise RuntimeError("❌ No valid rows left after dropping missing Salary values.")

    print(f">> Rows before: {len(merged)}   Rows after: {len(clean)}")

    X = clean[FEATURE_COLUMNS]
    y = clean[TARGET_COLUMN]

    # ========================================================
    # PREPROCESSOR SETUP
    # ========================================================

    numeric_features = ["Age", "Years of Experience"]
    categorical_features = ["Gender", "Education Level", "Job Title"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print(">> Training model...")
    pipeline.fit(X, y)

    # ========================================================
    # SAVE *FULL PIPELINE* AS ARTIFACT
    # ========================================================

    print(">> Saving artifacts...")
    artifact_path = save_artifacts(pipeline)
    print(f"\n🎉 Training complete!\nArtifacts saved to: {artifact_path}")

    return {
        "mhi": mhi,
        "kpis": kpis,
        "diagnostics": diagnostics
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    summary = train_model()
    print("\n=== TRAINING SUMMARY ===")
    for k, v in summary.items():
        print(f"\n{k.upper()}:\n{v}")

Recruiting Insight Engine
Deterministic HR Analytics for the PwC Technical Challenge
📌 Overview

The Recruiting Insight Engine is a deterministic, fully explainable HR analytics system that:

Predicts candidate salary based on 5 standardized attributes

Provides transparent reasoning behind every prediction

Surfaces market alignment, fairness considerations, and cohort insights

Generates a structured, business-friendly narrative

Includes a lightweight UI for HR analyst workflows

Contains a full evaluation harness for regression testing and model drift detection

This project was developed exclusively for the PwC technical challenge using the dataset provided.
It is not a production HR system and should not be used for real compensation decisions.

🎯 Purpose

The system demonstrates:

Ability to design intelligent analytic systems

Ability to reduce cognitive load for decision-makers

Technical proficiency in ML pipelines, interpretability, fairness, and API design

Clean architectural thinking

Solid engineering and documentation practices

📦 Architecture Summary
recruiting_insight_engine/
│
├── app/
│   ├── api/
│   │   └── routes.py              # /api/predict and /api/insights endpoints
│   ├── insights/
│   │   ├── engine.py              # Full Insight Engine (Phase 2)
│   │   └── eval_harness.py        # Drift testing + baseline comparison
│   ├── ml/
│   │   ├── train_pipeline.py      # Deterministic model training pipeline
│   │   ├── inference.py           # Strict feature inference wrapper
│   │   └── artifacts/             # model.pkl + preprocessor.pkl
│   ├── static/
│   │   ├── ui.html                # Analyst Console UI
│   │   ├── ui.css
│   │   └── ui.js
│   └── main.py                    # FastAPI app + static UI mount
│
├── data/
│   ├── raw/                       # Provided CSV files
│   └── processed/                 # Merged, cleaned dataset
│
├── tests/
│   ├── baseline_insights.json     # Stored canonical insights
│   └── drift_report.json          # Generated on comparison
│
├── requirements.txt
└── README.md

🧠 Core Features
1. Deterministic Salary Prediction

RandomForestRegressor + structured preprocessing

Strict 5-feature schema

Zero randomness during inference

2. Cohort & Market Insights

Salary percentiles

Market band classification

Comparison vs job, education, and gender cohorts

Confidence scoring based on sample size

3. Bias & Fairness Audit

Group means, sample counts

Gap ratio & absolute difference

Parity differences

Dataset-level bias flags

4. Interpretability Module

Normalized feature importances

Contribution breakdown

Top drivers

Narrative explanation

5. Drift Evaluation Harness

Baseline generation

Field-level drift detection

Severity classification

Recommendations

6. Analyst Console UI

Fast, clean, static front-end

Real-time insights

Narrative + structured tables

Debug JSON view

🚀 How to Run the Application
1. Install dependencies
pip install -r requirements.txt

2. Train the model (if needed)

Artifacts are included, but you can retrain:

python app/ml/train_pipeline.py

3. Start the API + UI
uvicorn app.main:app --reload

4. Open the Analyst Console

Go to:

👉 http://127.0.0.1:8000/ui

Use the form to enter candidate information and generate insights.

🧪 Running the Evaluation Harness
Generate a new baseline
python app/insights/eval_harness.py baseline

Compare current engine vs baseline
python app/insights/eval_harness.py compare


Outputs stored in:

tests/baseline_insights.json

tests/drift_report.json

📊 Dataset Notes

The provided dataset contains limited representation across:

job titles

education levels

demographic attributes

Therefore:

Some cohort tables will show “insufficient data”

Fairness metrics may only compute for certain derived age groups

Predictions for sparse groups will produce low confidence

This is expected behavior and documented in the User Guide

📘 Documentation Bundle

This repository includes:

1. Technical Architecture Document

Explains purpose, scope, and full system design.

2. QA Test Plan

Covers scenario testing, regression testing, and edge case coverage.

3. User Guide

Walks HR/analyst users through how to use the UI.

4. README (this file)

GitHub-ready summary for reviewers.

🛑 Scope & Ownership Disclaimer

This solution:

Was built specifically for the PwC challenge

Uses PwC-provided data

Is non-commercial, demonstrative, and not intended for real hiring decisions

Remains entirely the property of PwC with regard to the dataset and challenge context

The system is meant to demonstrate engineering ability, not to serve as a fully validated compensation tool.

🤝 Conclusion

The Recruiting Insight Engine demonstrates:

Strong ML engineering fundamentals

Enterprise-level system design

Deterministic and auditable analytics

Explainability, fairness evaluation, and interpretability

Clean architecture and professional documentation

Focus on reducing cognitive load for HR decision-makers

It is a complete, self-contained technical submission suitable for PwC evaluation.
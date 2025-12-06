⭐ README.md — PwC Recruiting Insight Engine (V2)
Salary Prediction + Insight Generation with Data Integrity Gating
🔷 1. Overview

The PwC Recruiting Insight Engine (V2) is a modular, fully-architected system that:

Predicts candidate salary using ML (Random Forest Regressor)

Generates enterprise-level insights:

Market Band

Drivers (feature contributions)

Cohort comparisons

Fairness overview

Narrative explanation

Enforces data integrity gating using a scientifically structured
Merge Health Index (MHI)
— a formal KPI-driven model designed during research.

The system is built with production-oriented architecture, following:

High cohesion

Low coupling

Cognitive frugality

Explicit module boundaries

Scalable ML design

Full API + UI integration

🔷 2. Architecture Overview
v2_project/
│
├── data_integrity/
│   ├── merge.py
│   ├── kpis.py
│   ├── mhi.py
│   └── diagnostics.py
│
├── model/
│   ├── train.py
│   ├── inference.py
│   ├── artifacts.py
│   └── artifacts/
│       ├── preprocessor.pkl
│       └── model.pkl
│
├── insight_engine/
│   ├── utils.py
│   ├── drivers.py
│   ├── cohorts.py
│   ├── fairness.py
│   ├── narrative.py
│   └── engine.py
│
├── api/
│   ├── server.py
│   └── schemas.py
│
├── ui/
│   ├── ui.html
│   ├── ui.css
│   └── ui.js
│
└── data/
    ├── people.csv
    ├── salary.csv
    ├── descriptions.csv
    └── merged.csv

🔷 3. Key System Concepts
3.1 Merge Health Index (MHI)

A scientifically defined metric that ensures model reliability before training or inference.

It combines:

Gate (schema + deterministic merge)

Core (join survival + completeness)

Refinement (error rates + drift penalties)

If:

RED → training & insights blocked

YELLOW → insights with caution

GREEN → full functionality

This approach mirrors enterprise data engineering quality frameworks.

🔷 4. Installation & Environment Setup
4.1 Install dependencies
pip install fastapi uvicorn pandas scikit-learn joblib

4.2 Ensure Python 3.12 environment

Check interpreter used by the project:

python --version


Make sure it's the same environment used by:

train.py

the UI

FastAPI

🔷 5. Training the Model

Before running the API, you must train and save artifacts.

5.1 Run training

From project root:

python model/train.py


This will:

Load data

Merge tables

Compute KPIs

Compute MHI

Gate training if MHI is RED

Train Random Forest

Save:

preprocessor.pkl

model.pkl

If training succeeds, you will see:

🎉 Training complete!
Artifacts saved to: model/artifacts/

🔷 6. Running the API

Start server:

uvicorn api.server:app --reload


API will be available at:

UI → http://127.0.0.1:8000/ui

Docs → http://127.0.0.1:8000/docs

Health → http://127.0.0.1:8000/health

🔷 7. UI (Analyst Console)

Open:

http://127.0.0.1:8000/ui

Features:

Input panel for candidate attributes

Insight cards:

Prediction

Offer band

Drivers

Cohorts

Fairness overview

Narrative insight

MHI badge

API status indicator

Raw JSON collapsible panel

Design goal:
Professional consulting UI with minimal cognitive load.

🔷 8. API Endpoints
POST /api/insights

Request example:

{
  "Age": 32,
  "Gender": "Male",
  "Education Level": "Bachelor's",
  "Job Title": "Software Engineer",
  "Years of Experience": 5
}


Response example:

{
  "prediction": 95000.0,
  "offer_band": "Market Average",
  "market_band": "Market Average",
  "drivers": [...],
  "cohort": {...},
  "fairness": {...},
  "narrative": "Based on the candidate profile...",
  "mhi": {...},
  "raw_json": {...}
}

🔷 9. Insight Engine Components
drivers.py

Estimates which features influence the prediction.

cohorts.py

Compares candidate to similar profiles.

fairness.py

Checks disparities by gender, education, job title.

narrative.py

Automatically synthesizes a structured explanation.

engine.py

Central orchestrator that:

Loads model

Predicts salary

Computes insights

Bundles final response

🔷 10. Data Integrity Layer
merge.py

Performs a deterministic, schema-controlled 3-table merge.

kpis.py

Computes all nine KPIs required for MHI.

mhi.py

Implements Gate + Core + Refinement + Zone calculation.

diagnostics.py

Utility health checks (missing values, duplicates, preview samples).

🔷 11. Troubleshooting
UI loads but nothing happens when clicking "Generate Insights"

Likely causes:

Incorrect JS path routing

CORS blocked

API returning 400 due to missing fields

JS reading undefined fields

API returns 500

Check server logs:

uvicorn api.server:app --reload

MHI is RED

Check:

Missing fields

Unexpected schema

Merge errors

Drift or mismatch in datasets

🔷 12. Why This Architecture?

Built around:

Cognitive Load Theory → low mental overhead

Bohmian implicate/explicate structure → deep module cohesion

Enterprise ML best practices → reproducibility & clarity

Separation of concerns

Predictive + analytical system integration

This V2 solution demonstrates:

ML engineering

Data quality governance

Insight generation

Backend integration

Frontend console

Scientific model for data integrity

Perfectly aligned with PwC expectations.

🔷 13. Deliverables

Full GitHub repository

This README.md (technical architecture document)

QA validation report

Merge Health Index research paper

User Guide document

Jupyter Notebook

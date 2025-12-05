# ================================
# FILE: api/schemas.py
# ================================
"""
schemas.py

Minimal-depth Pydantic models for request/response validation in the API layer.

These schemas are intentionally:
- simple
- deterministic
- low-cognitive-load
- tightly aligned with the Insight Engine I/O contract

No business logic is included.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Candidate Input Schema
# ------------------------------------------------------------

class CandidateInput(BaseModel):
    Age: Optional[float] = Field(None, description="Age of the candidate")
    Gender: Optional[str] = Field(None, description="Gender identity")
    Education_Level: Optional[str] = Field(None, alias="Education Level")
    Job_Title: Optional[str] = Field(None, alias="Job Title")
    Years_of_Experience: Optional[float] = Field(None, alias="Years of Experience")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "Age": 32,
                "Gender": "Male",
                "Education Level": "Bachelor's",
                "Job Title": "Software Engineer",
                "Years of Experience": 5
            }
        }


# ------------------------------------------------------------
# Insight Response Schema
# ------------------------------------------------------------

class InsightResponse(BaseModel):
    prediction: float
    market_band: str
    mhi: Dict[str, Any]
    gate_mode: str

    drivers: Optional[Any]
    cohorts: Optional[Any]
    fairness: Optional[Any]
    narrative: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "prediction": 92000.0,
                "market_band": "at_market",
                "mhi": {"MHI": 0.96, "zone": "GREEN"},
                "gate_mode": "FULL_INSIGHTS",
                "drivers": [{"feature": "Age", "importance": 0.22}],
                "cohorts": {"job_title": {"mean": 88000, "delta": 4000}},
                "fairness": {"gender": {"mean": 87000, "delta": 5000}},
                "narrative": "The model estimates a salary of $92,000, aligned with market expectations..."
            }
        }


# ------------------------------------------------------------
# Healthcheck Schema
# ------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "Insight Engine API operational"


# ------------------------------------------------------------
# Error Schema (optional but helpful)
# ------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

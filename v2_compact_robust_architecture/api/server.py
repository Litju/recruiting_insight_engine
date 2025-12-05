from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict

from insight_engine.engine import generate_insights

app = FastAPI(
    title="PwC Recruiting Insight Engine API",
    version="2.0",
    description="Salary prediction + insight generation with data integrity gating.",
)

# Mount the UI directory at /ui (for ui.html, ui.css, ui.js, etc.)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


# ============================
# SCHEMAS
# ============================

class CandidateInput(BaseModel):
    Age: int
    Gender: str
    Education_Level: str | None = None
    Education_Level_raw: str | None = None
    Education_Level_display: str | None = None
    Education_Level_alt: str | None = None
    Education_Level_label: str | None = None
    Education_Level_code: str | None = None
    # For compatibility with earlier versions, we keep only a single key:
    # In practice the UI sends "Education Level" as a field in JSON.
    # To avoid complexity, we accept it as "Education_Level" in Pydantic and map manually.

    Job_Title: str | None = None
    Job_Title_raw: str | None = None
    Job_Title_display: str | None = None
    Job_Title_code: str | None = None
    # Again, the UI sends "Job Title", we normalize inside the endpoint.

    Years_of_Experience: float

    # NOTE:
    # We will not rely directly on these field names in the Insight Engine.
    # Instead, we will construct the canonical dict with the expected keys:
    # "Age", "Gender", "Education Level", "Job Title", "Years of Experience"


class InsightResponse(BaseModel):
    prediction: float | None = None
    offer_band: str | None = None
    market_band: str | None = None
    drivers: list[Dict[str, Any]] | None = None
    cohort: Dict[str, Any] | None = None
    fairness: Dict[str, Any] | None = None
    narrative: str | None = None
    mhi: Dict[str, Any] | None = None
    raw_json: Dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    detail: str


# ============================
# ROUTES
# ============================

@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return HealthResponse(status="ok", message="PwC Recruiting Insight Engine API is running.")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", message="Insight Engine operational")


@app.get("/ui", response_class=HTMLResponse)
async def get_ui() -> HTMLResponse:
    """
    Serve the main analyst console HTML.
    Static assets (ui.css, ui.js) are served via the /ui mount.
    """
    with open("ui/ui.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post(
    "/api/insights",
    response_model=InsightResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def api_insights(payload: Dict[str, Any]) -> InsightResponse:
    """
    Main endpoint used by the UI to request insights.
    Accepts a generic dict so we don't fight with key naming,
    then normalizes to the canonical feature names expected by the model.
    """
    try:
        # Normalize keys coming from UI
        # UI sends:
        #   Age
        #   Gender
        #   EducationLevel  (or similar)
        #   JobTitle
        #   YearsOfExperience
        # We map them into the canonical keys used in the model:
        #   "Age", "Gender", "Education Level", "Job Title", "Years of Experience"
        candidate = {
            "Age": payload.get("Age"),
            "Gender": payload.get("Gender"),
            "Education Level": payload.get("Education Level") or payload.get("EducationLevel"),
            "Job Title": payload.get("Job Title") or payload.get("JobTitle"),
            "Years of Experience": payload.get("Years of Experience") or payload.get("YearsOfExperience"),
        }

        # Basic validation
        missing = [k for k, v in candidate.items() if v is None]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing)}",
            )

        insights = generate_insights(candidate)
        return InsightResponse(**insights)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

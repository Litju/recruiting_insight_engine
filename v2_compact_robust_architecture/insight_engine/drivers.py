# ============================
# FILE: insight_engine/drivers.py
# ============================

"""
Drivers module

Computes a simple, interpretable list of "drivers" based on:
- candidate profile
- high-level domain rules

This deliberately uses a heuristic rather than model-specific
feature importances, to keep complexity and cognitive load low.
"""

from typing import Dict, List


FEATURE_ORDER = [
    "Years of Experience",
    "Job Title",
    "Education Level",
    "Age",
    "Gender",
]


def compute_drivers(pipeline, candidate: Dict) -> List[Dict]:
    """
    Compute an ordered list of drivers for the candidate.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Trained pipeline (preprocessor + model). Currently unused,
        but kept in the signature for future extensibility.
    candidate : dict
        Candidate profile:
        {
            "Age": int,
            "Gender": str,
            "Education Level": str,
            "Job Title": str,
            "Years of Experience": float
        }

    Returns
    -------
    List[Dict]:
        [
            {
                "feature": str,
                "contribution": str,   # "high" | "medium" | "low"
                "direction": str,      # "positive" | "neutral" | "uncertain"
                "reason": str
            },
            ...
        ]
    """

    age = candidate.get("Age")
    gender = candidate.get("Gender")
    edu = candidate.get("Education Level")
    job = candidate.get("Job Title")
    years = candidate.get("Years of Experience")

    drivers: List[Dict] = []

    # Years of Experience
    if years is not None:
        if years >= 10:
            contribution = "high"
            direction = "positive"
            reason = "Extensive prior experience tends to support higher salary bands."
        elif years >= 5:
            contribution = "medium"
            direction = "positive"
            reason = "Solid experience level, aligned with standard mid-career roles."
        else:
            contribution = "medium"
            direction = "uncertain"
            reason = "Lower accumulated experience; salary may be more sensitive to role and education."
    else:
        contribution = "low"
        direction = "uncertain"
        reason = "Experience information is missing; contribution is uncertain."
    drivers.append({
        "feature": "Years of Experience",
        "contribution": contribution,
        "direction": direction,
        "reason": reason,
    })

    # Job Title
    if job:
        senior_keywords = ["Senior", "Lead", "Manager", "Director", "Head"]
        is_senior = any(kw.lower() in job.lower() for kw in senior_keywords)
        if is_senior:
            contribution = "high"
            direction = "positive"
            reason = "Job title suggests senior responsibilities, often associated with higher salary ranges."
        else:
            contribution = "medium"
            direction = "neutral"
            reason = "Role is not explicitly senior; salary will depend on market band for this title."
    else:
        contribution = "low"
        direction = "uncertain"
        reason = "Job title is missing; difficult to anchor against market bands."
    drivers.append({
        "feature": "Job Title",
        "contribution": contribution,
        "direction": direction,
        "reason": reason,
    })

    # Education Level
    if edu:
        if edu.lower().startswith("phd"):
            contribution = "medium"
            direction = "positive"
            reason = "Doctoral education can support higher salary expectations in many roles."
        elif edu.lower().startswith("master"):
            contribution = "medium"
            direction = "positive"
            reason = "Postgraduate education is a positive driver for compensation."
        elif edu.lower().startswith("bachelor"):
            contribution = "medium"
            direction = "neutral"
            reason = "Bachelor's level is standard baseline; impact depends on role and experience."
        else:
            contribution = "low"
            direction = "uncertain"
            reason = "Education level does not clearly map to typical compensation tiers."
    else:
        contribution = "low"
        direction = "uncertain"
        reason = "Education information is missing; contribution is uncertain."
    drivers.append({
        "feature": "Education Level",
        "contribution": contribution,
        "direction": direction,
        "reason": reason,
    })

    # Age
    if age is not None:
        if age < 25:
            contribution = "low"
            direction = "neutral"
            reason = "Early-career profile; salary often constrained more by experience than age."
        elif age <= 40:
            contribution = "medium"
            direction = "neutral"
            reason = "Prime working years; age is not a direct driver but contextual."
        else:
            contribution = "low"
            direction = "neutral"
            reason = "Age alone should not drive compensation; experience and role are more central."
    else:
        contribution = "low"
        direction = "uncertain"
        reason = "Age not provided; contribution is neutral."
    drivers.append({
        "feature": "Age",
        "contribution": contribution,
        "direction": direction,
        "reason": reason,
    })

    # Gender
    if gender:
        contribution = "low"
        direction = "neutral"
        reason = (
            "Gender should not directly affect salary; it is monitored only for fairness "
            "and equity, not as a compensable factor."
        )
    else:
        contribution = "low"
        direction = "neutral"
        reason = "Gender not specified; fairness monitoring still applies at a cohort level."
    drivers.append({
        "feature": "Gender",
        "contribution": contribution,
        "direction": direction,
        "reason": reason,
    })

    # Keep order stable
    ordered = [d for name in FEATURE_ORDER for d in drivers if d["feature"] == name]
    return ordered

# ============================
# FILE: insight_engine/narrative.py
# ============================

"""
Narrative module

Generates a concise, deterministic narrative that ties together:
- salary prediction
- drivers
- cohort insights
- fairness summary
"""

from typing import Dict, Any, List


def generate_narrative(
    predicted_salary: float,
    drivers: List[Dict[str, Any]],
    cohort_info: Dict[str, str],
    fairness: Dict[str, Any],
) -> str:
    """
    Build a short narrative explanation.

    Parameters
    ----------
    predicted_salary : float
    drivers : List[dict]
    cohort_info : dict
    fairness : dict

    Returns
    -------
    str
    """
    key_drivers = [d for d in drivers if d.get("contribution") == "high"]
    medium_drivers = [d for d in drivers if d.get("contribution") == "medium"]

    parts = []

    # Core prediction statement
    parts.append(
        f"The model estimates a salary of approximately ${predicted_salary:,.0f} "
        "for this profile."
    )

    # Drivers
    if key_drivers:
        driver_names = ", ".join(d["feature"] for d in key_drivers)
        parts.append(
            f"The strongest drivers influencing this estimate are: {driver_names}."
        )
    elif medium_drivers:
        driver_names = ", ".join(d["feature"] for d in medium_drivers)
        parts.append(
            f"The estimate is primarily shaped by: {driver_names}."
        )
    else:
        parts.append(
            "The model did not identify any single dominant driver for this prediction."
        )

    # Cohort highlights (job + education)
    job_msg = cohort_info.get("job_title")
    edu_msg = cohort_info.get("education")

    if job_msg:
        parts.append(job_msg)
    if edu_msg:
        parts.append(edu_msg)

    # Fairness
    fairness_summary = fairness.get("summary")
    if fairness_summary:
        parts.append(fairness_summary)

    return " ".join(parts)

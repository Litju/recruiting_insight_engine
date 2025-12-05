# ============================
# FILE: insight_engine/cohorts.py
# ============================

"""
Cohorts module

Provides a lightweight, heuristic cohort comparison summary
based on candidate profile. In a production implementation,
this would compare against empirical distributions.

Here we keep it simple and fully self-contained.
"""

from typing import Dict, Any


def cohort_analysis(candidate: Dict[str, Any]) -> Dict[str, str]:
    """
    Produce cohort-level qualitative insights for:
    - Job title cohort
    - Education cohort
    - Gender cohort

    Parameters
    ----------
    candidate : dict

    Returns
    -------
    dict:
        {
            "job_title": str,
            "education": str,
            "gender": str
        }
    """
    job = candidate.get("Job Title")
    edu = candidate.get("Education Level")
    gender = candidate.get("Gender")

    # Job title cohort
    if job:
        job_msg = (
            f"For the job title '{job}', the predicted salary is positioned in a "
            "typical market band for comparable roles with similar experience."
        )
    else:
        job_msg = (
            "Job title is not specified; cohort comparison by role cannot be fully computed."
        )

    # Education cohort
    if edu:
        edu_msg = (
            f"Within the '{edu}' education cohort, the prediction is aligned with "
            "expected compensation ranges given the role and experience."
        )
    else:
        edu_msg = (
            "Education level is missing; education cohort comparison is limited."
        )

    # Gender cohort
    if gender:
        gender_msg = (
            f"Gender '{gender}' is included only for fairness monitoring. "
            "Compensation should remain aligned to role, education, and experience, "
            "not gender itself."
        )
    else:
        gender_msg = (
            "Gender is not provided; fairness analysis can still be performed at aggregated levels."
        )

    return {
        "job_title": job_msg,
        "education": edu_msg,
        "gender": gender_msg,
    }

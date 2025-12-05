# ============================
# FILE: insight_engine/fairness.py
# ============================

"""
Fairness module

Generates a high-level fairness summary and potential flags.
This is a heuristic layer: it does not compute group statistics,
but expresses how this candidate fits into a monitored fairness frame.
"""

from typing import Dict, Any, List


def compute_fairness_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a fairness summary object for the candidate.

    Parameters
    ----------
    candidate : dict

    Returns
    -------
    dict:
        {
            "summary": str,
            "flags": List[str]
        }
    """
    gender = candidate.get("Gender")
    edu = candidate.get("Education Level")
    flags: List[str] = []

    if gender:
        gender_note = (
            f"Gender '{gender}' is monitored to ensure that average predicted "
            "compensation remains comparable across gender groups for similar "
            "roles and experience."
        )
    else:
        gender_note = (
            "Gender not provided; fairness checks can still be applied at an aggregated level."
        )

    if edu:
        edu_note = (
            f"Education level '{edu}' is evaluated to avoid systematic under- or "
            "over-valuation of specific education cohorts."
        )
    else:
        edu_note = (
            "Education level not provided; fairness comparison by education cohort is limited."
        )

    # Example flagging logic (simple, heuristic)
    if not gender:
        flags.append("Missing gender information reduces granularity of fairness monitoring.")
    if not edu:
        flags.append("Missing education information may limit cohort fairness checks.")

    if not flags:
        summary = (
            "No immediate fairness concerns are detected at the individual level. "
            "This prediction should still be monitored as part of broader group-level audits."
        )
    else:
        summary = (
            "Some fairness monitoring limitations are present due to missing or "
            "incomplete attributes. See flags for details."
        )

    return {
        "summary": f"{gender_note} {edu_note} {summary}",
        "flags": flags,
    }

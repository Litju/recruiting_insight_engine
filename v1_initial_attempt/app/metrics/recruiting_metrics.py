from typing import Any, Dict


def compute_recruiting_metrics(candidate_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the 10 locked recruiting-only metrics.

    Phase 1: return minimal, hard-coded stub values with the correct shape.
    Phase 2: implement real formulas based on the candidate profile
             and model outputs (if needed).
    """

    # TODO: implement real metric logic. These are placeholders.
    return {
        "cognitive_ability_proxy": 0.70,
        "experience_relevance_score": 0.75,
        "role_seniority_alignment": 0.80,
        "market_alignment_index": 0.65,
        "candidate_completeness_score": 0.90,
        "interview_readiness_score": 0.85,
        "assessment_completion_indicator": True,
        "early_turnover_risk": 0.20,
        "job_match_index": 0.78,
        "pay_equity_consistency_check": "within_expected_range",
    }

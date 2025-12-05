# ================================
# FILE: insight_engine/utils.py
# ================================
"""
Utility helpers for the Insight Engine.

This module is intentionally minimal to keep cognitive load low.
It exposes only small foundational utilities used across the insight engine layer:
- Formatting helpers
- MHI gating helpers
- Safe numeric operations
- Lightweight data utilities

No business logic, no cross-module dependencies.
"""

from typing import Any, Dict


# ---------------------------------------------------------
# 1. check_mhi_gate
# ---------------------------------------------------------
def check_mhi_gate(mhi_result: Dict[str, Any]) -> str:
    """
    Returns the operational mode permitted by the MHI zone.

    Parameters
    ----------
    mhi_result : dict
        Output from data_integrity.mhi.compute_mhi(), contains:
        - zone ∈ {"RED", "YELLOW", "GREEN"}

    Returns
    -------
    str
        One of:
        - "NO_INSIGHTS" → RED zone (only prediction allowed)
        - "LIMITED_INSIGHTS" → YELLOW zone
        - "FULL_INSIGHTS" → GREEN zone
    """
    zone = mhi_result.get("zone", "RED")

    if zone == "RED":
        return "NO_INSIGHTS"
    elif zone == "YELLOW":
        return "LIMITED_INSIGHTS"
    return "FULL_INSIGHTS"


# ---------------------------------------------------------
# 2. safe_div
# ---------------------------------------------------------
def safe_div(a: float, b: float) -> float:
    """
    Safely divide two numbers. Returns 0.0 if b == 0.

    Reduces cognitive load for fairness, cohorts, and drivers modules.
    """
    if b == 0:
        return 0.0
    return a / b


# ---------------------------------------------------------
# 3. format_currency
# ---------------------------------------------------------
def format_currency(value: float) -> str:
    """
    Utility to format salary predictions and cohort averages.
    """
    try:
        return f"${value:,.0f}"
    except Exception:
        return "$0"


# ---------------------------------------------------------
# 4. clamp
# ---------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    """
    Clamp a numeric value into a given range.
    """
    return max(low, min(high, value))


# ---------------------------------------------------------
# 5. prepare_output
# ---------------------------------------------------------
def prepare_output(label: str, payload: Any) -> Dict[str, Any]:
    """
    Wraps an output payload in a consistent insight-engine JSON structure.
    """
    return {"type": label, "data": payload}

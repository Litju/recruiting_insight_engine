"""
Merge Health Index (MHI) Computation – V2 (Corrected)
-----------------------------------------------------

This module computes:
- Gate score
- Core score
- Refinement score
- Final Merge Health Index (MHI)
- Health Zone (RED/YELLOW/GREEN)

Latency_norm is OPTIONAL and defaults to 0.0.
"""

import math


# ============================================================
# 1. REQUIRED KPIs (Latency_norm REMOVED)
# ============================================================

REQUIRED_KPIS = [
    "KAS",
    "SCS",
    "MDC",
    "JSR",
    "CCR_mean",
    "MER_norm",
    "DFC_norm",
    "MDS"
]


# ============================================================
# 2. INTERNAL VALIDATION
# ============================================================

def _validate_kpis(kpis: dict) -> dict:
    """
    Ensures that required KPIs exist.
    Optional KPIs (Latency_norm) are auto-filled with default.
    """

    if not isinstance(kpis, dict):
        raise ValueError("KPIs must be a dictionary")

    missing = [k for k in REQUIRED_KPIS if k not in kpis]
    if missing:
        raise ValueError(
            f"MHI computation requires KPIs {REQUIRED_KPIS}, "
            f"but missing keys: {missing}"
        )

    # Optional KPI: latency, defaults to 0.0
    latency = kpis.get("Latency_norm", 0.0)

    sanitized = {k: float(kpis[k]) for k in REQUIRED_KPIS}
    sanitized["Latency_norm"] = float(latency)

    return sanitized


# ============================================================
# 3. COMPONENT FUNCTIONS
# ============================================================

def compute_gate(kpis: dict) -> float:
    """
    Gate = 1 if:
        - KAS ≥ 0.9
        - SCS = 1
        - MDC = 1
    else 0
    """
    if kpis["KAS"] >= 0.9 and kpis["SCS"] == 1.0 and kpis["MDC"] == 1:
        return 1.0
    return 0.0


def compute_core(kpis: dict) -> float:
    """
    Core = sqrt(JSR * CCR_mean)
    """
    return math.sqrt(max(0.0, kpis["JSR"] * kpis["CCR_mean"]))


def compute_refinement(kpis: dict) -> float:
    """
    Refinement = exp(-2 * P)

    P = 0.4 * MER_norm
      + 0.3 * DFC_norm
      + 0.3 * (1 - MDS)

    Latency_norm is optional and not included in P by design.
    """
    P = (
        0.4 * kpis["MER_norm"]
        + 0.3 * kpis["DFC_norm"]
        + 0.3 * (1 - kpis["MDS"])
    )
    return math.exp(-2 * P)


# ============================================================
# 4. FINAL MHI + ZONE
# ============================================================

def compute_mhi(kpis: dict) -> dict:
    """
    Computes Gate, Core, Refinement, final MHI, and Zone.

    Returns:
        dict with:
            - Gate
            - Core
            - Refinement
            - MHI
            - zone
    """

    kpis = _validate_kpis(kpis)

    Gate = compute_gate(kpis)
    Core = compute_core(kpis)
    Refinement = compute_refinement(kpis)

    MHI = Gate * Core * Refinement

    # --------------------------------------------------------
    # Zone classification
    # --------------------------------------------------------
    if Gate == 0 or MHI < 0.40:
        zone = "RED"
    elif MHI < 0.75:
        zone = "YELLOW"
    else:
        zone = "GREEN"

    return {
        "Gate": Gate,
        "Core": Core,
        "Refinement": Refinement,
        "MHI": MHI,
        "zone": zone,
        "Latency_norm": kpis["Latency_norm"]
    }

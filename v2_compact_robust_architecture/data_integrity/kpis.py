"""
KPI Computation Module (V2)
Computes all merge health KPIs required for the MHI.
"""

import pandas as pd
import numpy as np

def compute_merge_kpis(people_df: pd.DataFrame,
                       salary_df: pd.DataFrame,
                       desc_df: pd.DataFrame,
                       merged_df: pd.DataFrame) -> dict:
    """
    Computes all KPIs needed for the Merge Health Index.
    
    Args:
        people_df: raw people table
        salary_df: raw salary table
        desc_df: raw descriptions table
        merged_df: merged final table
    
    Returns:
        dict of KPI values, all normalized to [0,1]
    """

    # ------------------------------
    # 1) KEY ALIGNMENT SCORE (KAS)
    # ------------------------------
    people_ids = set(people_df["id"])
    salary_ids = set(salary_df["id"])
    desc_ids = set(desc_df["id"])

    intersection = len(people_ids & salary_ids & desc_ids)
    union = len(people_ids | salary_ids | desc_ids)

    KAS = intersection / union if union > 0 else 0


    # ------------------------------
    # 2) SCHEMA CONFORMITY SCORE (SCS)
    # ------------------------------
    expected_cols = {
        "id",
        "Description",
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience",
        "Salary",
    }

    SCS = 1.0 if set(merged_df.columns) >= expected_cols else 0.0


    # ------------------------------
    # 3) MERGE DETERMINISM CHECK (MDC)
    # ------------------------------
    MDC = 1 if merged_df["id"].is_unique else 0


    # ------------------------------
    # 4) JOIN SURVIVAL RATE (JSR)
    # ------------------------------
    expected_rows = max(len(people_df), len(salary_df), len(desc_df))
    actual_rows = len(merged_df)
    JSR = actual_rows / expected_rows if expected_rows > 0 else 0


    # ------------------------------
    # 5) COMPLETENESS OF CRITICAL FIELDS (CCR_mean)
    # ------------------------------
    critical_fields = [
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience",
        "Salary"
    ]

    completeness_scores = []
    for col in critical_fields:
        if col in merged_df.columns:
            c = merged_df[col].notna().mean()
            completeness_scores.append(c)
        else:
            completeness_scores.append(0.0)

    CCR_mean = float(np.mean(completeness_scores))


    # ------------------------------
    # 6) MERGE ERROR RATE (normalized)
    # ------------------------------
    # Basic: percentage of rows with any missing critical field
    if len(merged_df) > 0:
        missing_any = (~merged_df[critical_fields].notna().all(axis=1)).mean()
        MER_norm = float(missing_any)
    else:
        MER_norm = 1.0


    # ------------------------------
    # 7) DIAGNOSTIC FAILURES COUNT (normalized)
    # ------------------------------
    # We define diagnostic failures as missing columns
    missing_cols = len(expected_cols - set(merged_df.columns))
    DFC_norm = min(1.0, missing_cols / len(expected_cols))


    # ------------------------------
    # 8) MERGE DRIFT SCORE (MDS)
    # ------------------------------
    # Simple drift metric: does merged_df contain expected number of rows?
    MDS = 1 - abs(expected_rows - actual_rows) / expected_rows if expected_rows > 0 else 0


    return {
        "KAS": KAS,
        "SCS": SCS,
        "MDC": MDC,
        "JSR": JSR,
        "CCR_mean": CCR_mean,
        "MER_norm": MER_norm,
        "DFC_norm": DFC_norm,
        "MDS": MDS,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows
    }

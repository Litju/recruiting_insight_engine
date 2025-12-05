"""
Data Integrity Diagnostics (V2)
Provides lightweight structural checks on the merged table.
"""

import pandas as pd

def basic_merge_diagnostics(merged_df: pd.DataFrame) -> dict:
    """
    Compute simple diagnostics on the merged dataset.
    
    Returns:
        dict with:
            - row_count
            - missing_values_total
            - missing_values_by_column
            - duplicate_rows
            - column_count
            - sample_preview (first 3 rows as dict)
    """
    if merged_df is None or len(merged_df) == 0:
        return {
            "row_count": 0,
            "missing_values_total": None,
            "missing_values_by_column": None,
            "duplicate_rows": None,
            "column_count": None,
            "sample_preview": None
        }

    row_count = len(merged_df)
    missing_total = merged_df.isna().sum().sum()
    missing_by_col = merged_df.isna().sum().to_dict()
    duplicates = merged_df.duplicated().sum()
    col_count = len(merged_df.columns)
    sample = merged_df.head(3).to_dict(orient="records")

    return {
        "row_count": row_count,
        "missing_values_total": missing_total,
        "missing_values_by_column": missing_by_col,
        "duplicate_rows": duplicates,
        "column_count": col_count,
        "sample_preview": sample
    }

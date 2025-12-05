"""
Merge Module (V2)
Responsible ONLY for merging the three raw datasets.
Returns a SINGLE DataFrame.
"""

import pandas as pd

def merge_tables(people_df: pd.DataFrame,
                 salary_df: pd.DataFrame,
                 desc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner joins people, salary, and descriptions on 'id'.
    Returns ONLY the merged DataFrame (not a tuple).
    """

    if "id" not in people_df or "id" not in salary_df or "id" not in desc_df:
        raise ValueError("All input tables must contain 'id' column")

    merged = (
        people_df
        .merge(salary_df, on="id", how="inner")
        .merge(desc_df, on="id", how="inner")
    )

    return merged  # <-- IMPORTANT: MUST RETURN ONLY THE DATAFRAME

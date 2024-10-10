"""
# Author = ruben
# Date: 10/10/24
# Project: ml-vit-events
# File: filters.py

Description: Function to filter dataframes
"""
import pandas as pd


def filter_data_by_setting(df_to_filter: pd.DataFrame, setting: str) -> pd.DataFrame:
    """Filter a dataframe by setting."""
    if setting == "NO RD":
        return df_to_filter[df_to_filter["rd"] == 0]
    elif setting == "RD":
        return df_to_filter[df_to_filter["rd"] == 1]
    else:
        return df_to_filter

"""Load, deduplicate and split transactions in chronological order."""

from pathlib import Path

import pandas as pd

TARGET = "Class"


def load_raw(path: str | Path = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Keep the first occurrence of each row and sort by relative second."""
    return (
        pd.read_csv(path)
        .drop_duplicates()
        .sort_values("Time", kind="stable", ignore_index=True)
    )


def time_split(
    df: pd.DataFrame, train: float = 0.70, val: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sorted rows, assigning each boundary second to the later set."""
    if not 0 < train < train + val < 1:
        raise ValueError("Train, validation and test fractions must be positive.")

    times = df["Time"]
    if not times.is_monotonic_increasing:
        raise ValueError("Transactions must be sorted by Time before splitting.")

    n = len(df)
    cut_train = int(n * train)
    cut_val = int(n * (train + val))
    if 0 < cut_train < cut_val < n:
        cut_train = times.searchsorted(times.iloc[cut_train], side="left")
        cut_val = times.searchsorted(times.iloc[cut_val], side="left")

    if not 0 < cut_train < cut_val < n:
        raise ValueError("Three non-empty splits require separate Time ranges.")

    return df.iloc[:cut_train], df.iloc[cut_train:cut_val], df.iloc[cut_val:]


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the fraud label from the input columns."""
    return df.drop(columns=TARGET), df[TARGET]

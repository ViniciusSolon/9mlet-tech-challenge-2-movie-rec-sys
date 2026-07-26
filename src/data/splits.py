"""Train/validation/test split helpers for recommendation data."""

from __future__ import annotations

import pandas as pd


def temporal_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    timestamp_col: str = "timestamp",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions preferring temporal order when timestamps exist.

    Args:
        df: Interaction frame with optional ``timestamp`` column.
        test_ratio: Fraction of rows reserved for test/validation.
        timestamp_col: Column used for chronological ordering.
        seed: RNG seed used only for the random fallback.

    Returns:
        ``(train_df, test_df)`` with no overlapping indices.

    Raises:
        ValueError: If ``test_ratio`` is outside ``(0, 1)`` or ``df`` is empty.
    """
    if df.empty:
        raise ValueError("cannot split an empty DataFrame")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    if timestamp_col in df.columns and df[timestamp_col].notna().any():
        ordered = df.copy()
        ordered["_ts_sort"] = _to_sortable_timestamp(ordered[timestamp_col])
        ordered = ordered.sort_values("_ts_sort", kind="mergesort")
        ordered = ordered.drop(columns=["_ts_sort"])
        cut = max(1, int(len(ordered) * (1.0 - test_ratio)))
        cut = min(cut, len(ordered) - 1)
        train_df = ordered.iloc[:cut].copy()
        test_df = ordered.iloc[cut:].copy()
        return train_df, test_df

    test_n = max(1, int(len(df) * test_ratio))
    test_df = df.sample(n=test_n, random_state=seed)
    train_df = df.drop(test_df.index)
    return train_df, test_df


def _to_sortable_timestamp(series: pd.Series) -> pd.Series:
    """Convert unix ints or datetime strings into a sortable numeric series."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    return converted.astype("int64", errors="ignore")


def assert_temporal_order(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> bool:
    """Return True when max(train timestamp) <= min(test timestamp)."""
    if timestamp_col not in train_df.columns or timestamp_col not in test_df.columns:
        return False
    if train_df.empty or test_df.empty:
        return False
    train_ts = _to_sortable_timestamp(train_df[timestamp_col])
    test_ts = _to_sortable_timestamp(test_df[timestamp_col])
    if train_ts.isna().all() or test_ts.isna().all():
        return False
    return float(train_ts.max()) <= float(test_ts.min())

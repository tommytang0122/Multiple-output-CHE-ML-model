# utils/data_guard.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class DataGuardError(Exception):
    """Base error for data guard checks."""


class MissingColumnsError(DataGuardError, KeyError):
    """Raised when required columns are missing."""


class TooFewRowsError(DataGuardError, ValueError):
    """Raised when dataset has too few rows."""


@dataclass(frozen=True)
class GuardReport:
    path: str
    n_rows_before: int
    n_rows_after: int
    dropped_na_rows: int
    kept_columns: List[str]


def make_xy_columns(n_x: int = 37, n_y: int = 7) -> Tuple[List[str], List[str]]:
    x_cols = [f"x{i}" for i in range(1, n_x + 1)]
    y_cols = [f"y{i}" for i in range(1, n_y + 1)]
    return x_cols, y_cols


def require_columns(df: pd.DataFrame, required: List[str], *, where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnsError(f"Missing required columns in {where}: {missing}")


def load_csv(
    path: str | Path,
    *,
    required_columns: Optional[List[str]] = None,
    keep_columns: Optional[List[str]] = None,
    drop_na: bool = True,
    min_rows: int = 1,
    na_values: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, GuardReport]:
    """
    Minimal, strict loader:
    - checks required columns exist
    - optionally keeps only keep_columns (stable order)
    - coerces ALL kept columns to numeric (errors -> NaN)
    - optionally drops any NaN rows
    - ensures at least min_rows remain
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    na_values = na_values or ["", "NA", "N/A", "null", "None"]

    df = pd.read_csv(p, keep_default_na=True, na_values=na_values)
    n_before = len(df)

    if required_columns is not None:
        require_columns(df, required_columns, where=str(p))

    kept_cols = list(df.columns)
    if keep_columns is not None:
        require_columns(df, keep_columns, where=str(p))
        df = df[keep_columns].copy()
        kept_cols = list(keep_columns)

    # Coerce numeric (strict for modeling)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dropped_na = 0
    if drop_na:
        before = len(df)
        df = df.dropna(axis=0, how="any")
        dropped_na = before - len(df)

    if len(df) < min_rows:
        raise TooFewRowsError(f"Too few rows after cleaning: {len(df)} < {min_rows} (file: {p})")

    report = GuardReport(
        path=str(p),
        n_rows_before=n_before,
        n_rows_after=len(df),
        dropped_na_rows=dropped_na,
        kept_columns=kept_cols,
    )
    return df, report


def load_xy_csv(
    path: str | Path,
    *,
    n_x: int = 37,
    n_y: int = 7,
    drop_na: bool = True,
    min_rows: int = 1,
) -> Tuple[pd.DataFrame, List[str], List[str], GuardReport]:
    """
    Convenience loader for your project format:
      x1..x37, y1..y7
    """
    x_cols, y_cols = make_xy_columns(n_x=n_x, n_y=n_y)
    needed = x_cols + y_cols
    df, rep = load_csv(
        path,
        required_columns=needed,
        keep_columns=needed,
        drop_na=drop_na,
        min_rows=min_rows,
    )
    return df, x_cols, y_cols, rep

# clean_data.py
# Fixed paths:
#   input : data/rawdata.csv
#   output: data/cleandata.csv
#
# Features:
#   Original A..AM (39)
#   - Merge AI + AJ -> AI_AJ
#   - Merge AK + AL -> AK_AL
#   => Final 37 features, renamed to x1..x37
#
# Targets:
#   AN..AT -> y1..y7

import sys
from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/rawdata.csv")
OUTPUT_PATH = Path("data/cleandata.csv")


def excel_col_to_num(col: str) -> int:
    n = 0
    for ch in col.upper():
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def num_to_excel_col(n: int) -> str:
    out = []
    while n:
        n, r = divmod(n - 1, 26)
        out.append(chr(r + ord("A")))
    return "".join(reversed(out))


def excel_col_range(start: str, end: str) -> list[str]:
    s = excel_col_to_num(start)
    e = excel_col_to_num(end)
    return [num_to_excel_col(i) for i in range(s, e + 1)]


def read_csv_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "None"]
    )

    # If header is actually data (all numeric-ish), re-read with header=None
    def is_numericish(x):
        try:
            float(str(x))
            return True
        except Exception:
            return False

    if all(is_numericish(c) for c in df.columns):
        df = pd.read_csv(
            path,
            header=None,
            keep_default_na=True,
            na_values=["", "NA", "N/A", "null", "None"]
        )

    return df


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
        return 2

    df = read_csv_robust(INPUT_PATH)

    if df.shape[1] < 46:
        raise ValueError(f"CSV has {df.shape[1]} columns, but needs at least 46 (A..AT).")

    # Slice by position
    X_raw = df.iloc[:, 0:39].copy()   # A..AM
    Y_raw = df.iloc[:, 39:46].copy()  # AN..AT

    # Temporary names (for merge logic)
    X_raw.columns = excel_col_range("A", "AM")
    Y_raw.columns = excel_col_range("AN", "AT")

    # Ensure numeric for merge sources
    for c in ["AI", "AJ", "AK", "AL"]:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")

    # Merge (SUM)
    X_raw["AI_AJ"] = X_raw["AI"] + X_raw["AJ"]
    X_raw["AK_AL"] = X_raw["AK"] + X_raw["AL"]

    # Drop original columns
    X_clean = X_raw.drop(columns=["AI", "AJ", "AK", "AL"])

    # Rename features to x1..x37 (order preserved)
    X_clean.columns = [f"x{i}" for i in range(1, X_clean.shape[1] + 1)]

    # Rename targets to y1..y7
    Y_clean = Y_raw.rename(
        columns={
            "AN": "y1",
            "AO": "y2",
            "AP": "y3",
            "AQ": "y4",
            "AR": "y5",
            "AS": "y6",
            "AT": "y7",
        }
    )

    # Final dataframe
    clean_df = pd.concat([X_clean, Y_clean], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Features: {X_clean.shape[1]} (x1..x{X_clean.shape[1]})")
    print(f"Targets : {Y_clean.shape[1]} (y1..y7)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

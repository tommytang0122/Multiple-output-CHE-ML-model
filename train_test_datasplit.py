# make_train_test.py
# Purpose:
#   Split data/cleandata.csv into train/test (0.8 / 0.2) once,
#   then you will ONLY use the train set to train:
#     - 7 Ridge (linear) models
#     - 7 RandomForest (non-linear) models
#
# Input:
#   data/cleandata.csv  (columns: x1..x37, y1..y7)
#
# Output:
#   data/train.csv
#   data/test.csv
#
# Run:
#   python make_train_test.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/cleandata.csv")
TRAIN_OUT = Path("data/train.csv")
TEST_OUT = Path("data/test.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    x_cols = [f"x{i}" for i in range(1, 38)]
    y_cols = [f"y{i}" for i in range(1, 8)]
    needed = x_cols + y_cols

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in {DATA_PATH}: {missing}")

    # Keep only needed columns in a stable order and coerce to numeric
    df = df[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop any rows with NaN (you said 0 is valid; NaN is true missing/invalid)
    before = len(df)
    df = df.dropna(axis=0, how="any")
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped rows with NaN: {dropped} (remaining={len(df)})")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("=== Train/Test Split Done ===")
    print(f"Input : {DATA_PATH}  (rows={len(df)}, cols={df.shape[1]})")
    print(f"Train : {TRAIN_OUT}  (rows={len(train_df)})")
    print(f"Test  : {TEST_OUT}  (rows={len(test_df)})")
    print(f"Split : train={1-TEST_SIZE:.1f} / test={TEST_SIZE:.1f} | random_state={RANDOM_STATE}")


if __name__ == "__main__":
    main()

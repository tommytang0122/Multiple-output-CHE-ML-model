# make_train_test.py
# Split data/cleandata.csv into train/test (0.8 / 0.2)
#
# Input : data/cleandata.csv  (x1..x37, y1..y7)
# Output:
#   data/train.csv
#   data/test.csv
#
# Run:
#   python make_train_test.py

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.data_guard import load_xy_csv

DATA_PATH = Path("data/cleandata.csv")
TRAIN_OUT = Path("data/train.csv")
TEST_OUT = Path("data/test.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    df, x_cols, y_cols, rep = load_xy_csv(DATA_PATH, drop_na=True, min_rows=10)

    if rep.dropped_na_rows > 0:
        print(f"[data_guard] Dropped NaN rows: {rep.dropped_na_rows} (kept={rep.n_rows_after})")

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
    print(f"Input : {DATA_PATH} (rows={rep.n_rows_after}, cols={df.shape[1]})")
    print(f"Train : {TRAIN_OUT} (rows={len(train_df)})")
    print(f"Test  : {TEST_OUT} (rows={len(test_df)})")
    print(f"Cols  : {len(x_cols)} features + {len(y_cols)} targets")
    print(f"Split : train={1-TEST_SIZE:.1f} / test={TEST_SIZE:.1f} | random_state={RANDOM_STATE}")


if __name__ == "__main__":
    main()

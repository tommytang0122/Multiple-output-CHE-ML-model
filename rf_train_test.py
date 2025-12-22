# rf_train_test.py
# Train/test split = 0.8 / 0.2
# Model: RandomForestRegressor (multi-output) on data/cleandata.csv
#
# Input : data/cleandata.csv  (expects columns x1..x37 and y1..y7)
# Output:
#   - models/rf_multioutput.joblib
#   - reports/rf_metrics.csv
#
# Run:
#   python rf_train_test.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


DATA_PATH = Path("data/cleandata.csv")
MODEL_PATH = Path("models/rf_multioutput.joblib")
REPORT_PATH = Path("reports/rf_metrics.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# RF hyperparams (safe baseline)
RF_PARAMS = dict(
    n_estimators=600,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)


def rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    return np.sqrt(mse)


def mae_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return mean_absolute_error(y_true, y_pred, multioutput="raw_values")


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return r2_score(y_true, y_pred, multioutput="raw_values")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input CSV: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Expect x1..x37 and y1..y7
    x_cols = [f"x{i}" for i in range(1, 38)]
    y_cols = [f"y{i}" for i in range(1, 8)]

    missing_x = [c for c in x_cols if c not in df.columns]
    missing_y = [c for c in y_cols if c not in df.columns]
    if missing_x or missing_y:
        raise KeyError(f"Missing columns. missing_x={missing_x}, missing_y={missing_y}")

    # Keep only needed columns, coerce to numeric
    df = df[x_cols + y_cols].copy()
    for c in x_cols + y_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with any NaN in X or Y
    before = len(df)
    df = df.dropna(axis=0, how="any")
    after = len(df)
    if after < before:
        print(f"Dropped rows with NaN: {before - after} (remaining={after})")

    X = df[x_cols].values
    Y = df[y_cols].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, Y_train)

    # Predict
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    # Metrics per target
    train_rmse = rmse_per_target(Y_train, Y_pred_train)
    test_rmse = rmse_per_target(Y_test, Y_pred_test)

    train_mae = mae_per_target(Y_train, Y_pred_train)
    test_mae = mae_per_target(Y_test, Y_pred_test)

    train_r2 = r2_per_target(Y_train, Y_pred_train)
    test_r2 = r2_per_target(Y_test, Y_pred_test)

    report = pd.DataFrame(
        {
            "target": y_cols,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }
    )

    # Helpful summaries
    report.loc[len(report)] = {
        "target": "mean",
        "train_rmse": float(np.mean(train_rmse)),
        "test_rmse": float(np.mean(test_rmse)),
        "train_mae": float(np.mean(train_mae)),
        "test_mae": float(np.mean(test_mae)),
        "train_r2": float(np.mean(train_r2)),
        "test_r2": float(np.mean(test_r2)),
    }
    report.loc[len(report)] = {
        "target": "worst(test_rmse)",
        "train_rmse": float(np.max(train_rmse)),
        "test_rmse": float(np.max(test_rmse)),
        "train_mae": float(np.max(train_mae)),
        "test_mae": float(np.max(test_mae)),
        "train_r2": float(np.min(train_r2)),
        "test_r2": float(np.min(test_r2)),
    }

    # Save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "x_cols": x_cols,
            "y_cols": y_cols,
            "rf_params": RF_PARAMS,
            "random_state": RANDOM_STATE,
        },
        MODEL_PATH,
    )
    report.to_csv(REPORT_PATH, index=False)

    # Print
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("\n=== RandomForest multi-output results ===")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Model saved : {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}\n")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

# train_rf_7.py
# Train 7 RandomForestRegressor models (one per target y1..y7) using ONLY data/train.csv
#
# Input : data/train.csv  (x1..x37, y1..y7)
# Output:
#   models/rf_y1.joblib ... models/rf_y7.joblib
#   reports/rf_train_metrics.csv
#
# Run:
#   python train_rf_7.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_guard import load_xy_csv

TRAIN_PATH = Path("data/train.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Baseline RF hyperparams (good default for tabular)
RF_PARAMS = dict(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    df, x_cols, y_cols, rep = load_xy_csv(TRAIN_PATH, drop_na=True, min_rows=10)

    if rep.dropped_na_rows > 0:
        print(f"[data_guard] Dropped NaN rows: {rep.dropped_na_rows} (kept={rep.n_rows_after})")

    X = df[x_cols].values.astype(float)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for y_name in y_cols:
        y = df[y_name].values.astype(float)

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X, y)

        y_pred = model.predict(X)
        m = regression_metrics(y, y_pred)

        model_path = MODELS_DIR / f"rf_{y_name}.joblib"
        joblib.dump(
            {
                "model": model,
                "x_cols": x_cols,
                "y_col": y_name,
                "rf_params": RF_PARAMS,
            },
            model_path,
        )

        rows.append({
            "target": y_name,
            "train_rmse": m["rmse"],
            "train_mae": m["mae"],
            "train_r2": m["r2"],
            "model_path": str(model_path),
        })

        print(f"Saved {y_name} -> {model_path} | train_rmse={m['rmse']:.6f} train_r2={m['r2']:.6f}")

    report = pd.DataFrame(rows)
    report_path = REPORTS_DIR / "rf_train_metrics.csv"
    report.to_csv(report_path, index=False)

    print("\n=== RandomForest training complete (train-only) ===")
    print(f"Report saved: {report_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

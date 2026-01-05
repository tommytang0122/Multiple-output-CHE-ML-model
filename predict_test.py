# predict_test.py
# Predict y1..y7 on data/test.csv using trained models, and export per-row predictions + summary.
#
# Input:
#   data/test.csv (x1..x37, y1..y7)
# Models:
#   models/rf_y1.joblib ... rf_y5.joblib, rf_y7.joblib
#   models/ridge_y6.joblib
# Output:
#   reports/test_predictions.csv
#   reports/test_summary_by_target.csv
#
# Run:
#   python predict_test.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_guard import load_xy_csv

TEST_PATH = Path("data/test.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

OUT_PRED = REPORTS_DIR / "test_predictions.csv"
OUT_SUMM = REPORTS_DIR / "test_summary_by_target.csv"

MODEL_BUNDLE = {
    "y1": MODELS_DIR / "rf_y1.joblib",
    "y2": MODELS_DIR / "rf_y2.joblib",
    "y3": MODELS_DIR / "rf_y3.joblib",
    "y4": MODELS_DIR / "rf_y4.joblib",
    "y5": MODELS_DIR / "rf_y5.joblib",
    "y6": MODELS_DIR / "ridge_y6.joblib",
    "y7": MODELS_DIR / "rf_y7.joblib",
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    art = joblib.load(path)
    if "model" not in art:
        raise KeyError(f"Invalid model artifact (missing 'model') in: {path}")
    return art


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df, x_cols, y_cols, rep = load_xy_csv(TEST_PATH, drop_na=True, min_rows=1)
    X = df[x_cols].values.astype(float)

    # Load models
    artifacts = {yk: load_artifact(p) for yk, p in MODEL_BUNDLE.items()}

    # Predict all y
    pred_cols = []
    err_cols = []
    for yk in y_cols:
        model = artifacts[yk]["model"]
        y_pred = np.asarray(model.predict(X)).reshape(-1)
        df[f"{yk}_pred"] = y_pred
        df[f"{yk}_err"] = df[f"{yk}_pred"] - df[yk]  # pred - true
        pred_cols.append(f"{yk}_pred")
        err_cols.append(f"{yk}_err")

    # Save per-row predictions
    out_df = df[x_cols + y_cols + pred_cols + err_cols].copy()
    out_df.to_csv(OUT_PRED, index=False)

    # Summary metrics
    rows = []
    for yk in y_cols:
        y_true = df[yk].values.astype(float)
        y_pred = df[f"{yk}_pred"].values.astype(float)
        m = regression_metrics(y_true, y_pred)

        abs_err = np.abs(y_pred - y_true)
        rows.append({
            "target": yk,
            "model": "ridge" if yk == "y6" else "rf",
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "abs_err_p50": float(np.percentile(abs_err, 50)),
            "abs_err_p90": float(np.percentile(abs_err, 90)),
            "abs_err_max": float(abs_err.max()),
        })

    summ = pd.DataFrame(rows).sort_values(by="rmse").reset_index(drop=True)
    summ.to_csv(OUT_SUMM, index=False)

    print("\n=== Test prediction done ===")
    print(f"Rows used: {len(df)} (dropped_na={rep.dropped_na_rows})")
    print(f"Saved per-row predictions: {OUT_PRED}")
    print(f"Saved summary:            {OUT_SUMM}\n")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()

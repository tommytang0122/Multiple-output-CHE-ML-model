# rf_y5_improve.py
# Purpose: Improve ONLY y5 with RandomForest (baseline vs log1p transform)
#
# Input : data/cleandata.csv  (expects columns x1..x37 and y1..y7)
# Output:
#   - reports/y5_compare.csv
#   - models/rf_y5_baseline.joblib
#   - models/rf_y5_log1p.joblib
#   - models/rf_y5_best.joblib  (auto-selected by best test_rmse)
#
# Run:
#   python rf_y5_improve.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


DATA_PATH = Path("data/cleandata.csv")
REPORT_PATH = Path("reports/y5_compare.csv")

MODEL_BASELINE_PATH = Path("models/rf_y5_baseline.joblib")
MODEL_LOG1P_PATH = Path("models/rf_y5_log1p.joblib")
MODEL_BEST_PATH = Path("models/rf_y5_best.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Baseline RF params (already decent)
RF_BASELINE = dict(
    n_estimators=800,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)

# A slightly more regularized alternative (often helps generalization)
RF_REG = dict(
    n_estimators=1200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
)


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def fit_and_eval_rf(X_train, X_test, y_train, y_test, rf_params):
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    tr = metrics(y_train, pred_train)
    te = metrics(y_test, pred_test)
    return model, tr, te


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input CSV: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    x_cols = [f"x{i}" for i in range(1, 38)]
    if "y5" not in df.columns:
        raise KeyError("Expected column y5 in data/cleandata.csv")

    # keep only needed, coerce numeric
    df = df[x_cols + ["y5"]].copy()
    for c in x_cols + ["y5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop NaN rows
    before = len(df)
    df = df.dropna(axis=0, how="any")
    after = len(df)
    if after < before:
        print(f"Dropped rows with NaN: {before - after} (remaining={after})")

    X = df[x_cols].values
    y = df["y5"].values

    # same split for all experiments
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rows = []

    # ---- 1) Baseline (raw y5) ----
    model_b1, tr_b1, te_b1 = fit_and_eval_rf(X_train, X_test, y_train, y_test, RF_BASELINE)
    rows.append({
        "variant": "baseline_raw_y5",
        "rf_params": "RF_BASELINE",
        "train_rmse": tr_b1[0], "test_rmse": te_b1[0],
        "train_mae": tr_b1[1],  "test_mae": te_b1[1],
        "train_r2": tr_b1[2],   "test_r2": te_b1[2],
    })

    model_b2, tr_b2, te_b2 = fit_and_eval_rf(X_train, X_test, y_train, y_test, RF_REG)
    rows.append({
        "variant": "baseline_raw_y5",
        "rf_params": "RF_REG",
        "train_rmse": tr_b2[0], "test_rmse": te_b2[0],
        "train_mae": tr_b2[1],  "test_mae": te_b2[1],
        "train_r2": tr_b2[2],   "test_r2": te_b2[2],
    })

    # ---- 2) log1p(y5) ----
    # Guard: log1p requires y >= -1 (your y5 should be >= 0.2, so OK)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    model_l1, tr_l1_log, te_l1_log = fit_and_eval_rf(X_train, X_test, y_train_log, y_test_log, RF_BASELINE)
    # invert to original scale for evaluation
    pred_train = np.expm1(model_l1.predict(X_train))
    pred_test = np.expm1(model_l1.predict(X_test))
    tr_l1 = metrics(y_train, pred_train)
    te_l1 = metrics(y_test, pred_test)
    rows.append({
        "variant": "log1p_y5_then_invert",
        "rf_params": "RF_BASELINE",
        "train_rmse": tr_l1[0], "test_rmse": te_l1[0],
        "train_mae": tr_l1[1],  "test_mae": te_l1[1],
        "train_r2": tr_l1[2],   "test_r2": te_l1[2],
    })

    model_l2, tr_l2_log, te_l2_log = fit_and_eval_rf(X_train, X_test, y_train_log, y_test_log, RF_REG)
    pred_train = np.expm1(model_l2.predict(X_train))
    pred_test = np.expm1(model_l2.predict(X_test))
    tr_l2 = metrics(y_train, pred_train)
    te_l2 = metrics(y_test, pred_test)
    rows.append({
        "variant": "log1p_y5_then_invert",
        "rf_params": "RF_REG",
        "train_rmse": tr_l2[0], "test_rmse": te_l2[0],
        "train_mae": tr_l2[1],  "test_mae": te_l2[1],
        "train_r2": tr_l2[2],   "test_r2": te_l2[2],
    })

    report = pd.DataFrame(rows).sort_values(by=["test_rmse", "test_mae"], ascending=[True, True]).reset_index(drop=True)

    # Save models + pick best by lowest test_rmse
    MODEL_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model_b1, "x_cols": x_cols, "target": "y5", "variant": "baseline_raw_y5", "rf_params": RF_BASELINE}, MODEL_BASELINE_PATH)
    joblib.dump({"model": model_l1, "x_cols": x_cols, "target": "y5", "variant": "log1p_y5_then_invert", "rf_params": RF_BASELINE, "transform": "log1p"}, MODEL_LOG1P_PATH)

    # Best row -> decide which model object to store
    best = report.iloc[0].to_dict()
    if best["variant"] == "baseline_raw_y5" and best["rf_params"] == "RF_BASELINE":
        best_obj = {"model": model_b1, "x_cols": x_cols, "target": "y5", "variant": "baseline_raw_y5", "rf_params": RF_BASELINE}
    elif best["variant"] == "baseline_raw_y5" and best["rf_params"] == "RF_REG":
        best_obj = {"model": model_b2, "x_cols": x_cols, "target": "y5", "variant": "baseline_raw_y5", "rf_params": RF_REG}
    elif best["variant"] == "log1p_y5_then_invert" and best["rf_params"] == "RF_BASELINE":
        best_obj = {"model": model_l1, "x_cols": x_cols, "target": "y5", "variant": "log1p_y5_then_invert", "rf_params": RF_BASELINE, "transform": "log1p"}
    else:
        best_obj = {"model": model_l2, "x_cols": x_cols, "target": "y5", "variant": "log1p_y5_then_invert", "rf_params": RF_REG, "transform": "log1p"}

    joblib.dump(best_obj, MODEL_BEST_PATH)

    report.to_csv(REPORT_PATH, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("\n=== y5 improvement experiments (sorted by best test_rmse) ===")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Report saved: {REPORT_PATH}")
    print(f"Baseline model saved: {MODEL_BASELINE_PATH}")
    print(f"Log1p model saved   : {MODEL_LOG1P_PATH}")
    print(f"Best model saved    : {MODEL_BEST_PATH}\n")
    print(report.to_string(index=False))
    print("\nBest choice:", best)

if __name__ == "__main__":
    main()

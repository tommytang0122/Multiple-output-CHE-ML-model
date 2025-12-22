# eval_models_test.py
# Evaluate trained Ridge (7) and RandomForest (7) models on data/test.csv
#
# Requires:
#   - utils/data_guard.py
#   - data/test.csv (x1..x37, y1..y7)
#   - models/ridge_y1.joblib ... models/ridge_y7.joblib
#   - models/rf_y1.joblib ... models/rf_y7.joblib
#
# Output:
#   - reports/eval_test_ridge_vs_rf_by_target.csv
#   - reports/eval_test_ridge_vs_rf_long.csv
#
# Run:
#   python eval_models_test.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_guard import load_xy_csv

TEST_PATH = Path("data/test.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

OUT_WIDE = REPORTS_DIR / "eval_test_ridge_vs_rf_by_target.csv"
OUT_LONG = REPORTS_DIR / "eval_test_ridge_vs_rf_long.csv"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    obj = joblib.load(path)
    if "model" not in obj:
        raise KeyError(f"Invalid model artifact (missing 'model') in: {path}")
    return obj


def predict_with_artifact(artifact: dict, X: np.ndarray) -> np.ndarray:
    # Ridge artifacts: sklearn Pipeline
    # RF artifacts: sklearn estimator
    model = artifact["model"]
    y_pred = model.predict(X)

    # Ensure 1D
    if isinstance(y_pred, (list, tuple)):
        y_pred = np.asarray(y_pred)
    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)
    return y_pred


def main():
    df, x_cols, y_cols, rep = load_xy_csv(TEST_PATH, drop_na=True, min_rows=10)

    if rep.dropped_na_rows > 0:
        print(f"[data_guard] Dropped NaN rows: {rep.dropped_na_rows} (kept={rep.n_rows_after})")

    X_test = df[x_cols].values.astype(float)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    long_rows = []

    for y_name in y_cols:
        y_true = df[y_name].values.astype(float)

        ridge_path = MODELS_DIR / f"ridge_{y_name}.joblib"
        rf_path = MODELS_DIR / f"rf_{y_name}.joblib"

        ridge_art = load_model(ridge_path)
        rf_art = load_model(rf_path)

        # (Optional safety) ensure model expects same columns
        if ridge_art.get("x_cols") and ridge_art["x_cols"] != x_cols:
            raise ValueError(f"Ridge {y_name}: x_cols mismatch between model and test data.")
        if rf_art.get("x_cols") and rf_art["x_cols"] != x_cols:
            raise ValueError(f"RF {y_name}: x_cols mismatch between model and test data.")

        ridge_pred = predict_with_artifact(ridge_art, X_test)
        rf_pred = predict_with_artifact(rf_art, X_test)

        m_ridge = regression_metrics(y_true, ridge_pred)
        m_rf = regression_metrics(y_true, rf_pred)

        long_rows.append({"target": y_name, "model": "ridge", **m_ridge, "model_path": str(ridge_path)})
        long_rows.append({"target": y_name, "model": "rf", **m_rf, "model_path": str(rf_path)})

    long_df = pd.DataFrame(long_rows)

    # Wide comparison table (one row per target)
    wide = (
        long_df.pivot(index="target", columns="model", values=["rmse", "mae", "r2"])
        .sort_index()
    )
    # flatten multiindex columns: ("rmse","ridge") -> "ridge_rmse"
    wide.columns = [f"{m}_{metric}" for metric, m in wide.columns]
    wide = wide.reset_index()

    # Add "winner" columns (lower RMSE/MAE wins; higher R2 wins)
    wide["winner_rmse"] = np.where(wide["rf_rmse"] < wide["ridge_rmse"], "rf", "ridge")
    wide["winner_mae"] = np.where(wide["rf_mae"] < wide["ridge_mae"], "rf", "ridge")
    wide["winner_r2"] = np.where(wide["rf_r2"] > wide["ridge_r2"], "rf", "ridge")

    # Add deltas (rf - ridge): negative is better for rmse/mae, positive is better for r2
    wide["delta_rmse_rf_minus_ridge"] = wide["rf_rmse"] - wide["ridge_rmse"]
    wide["delta_mae_rf_minus_ridge"] = wide["rf_mae"] - wide["ridge_mae"]
    wide["delta_r2_rf_minus_ridge"] = wide["rf_r2"] - wide["ridge_r2"]

    long_df.to_csv(OUT_LONG, index=False)
    wide.to_csv(OUT_WIDE, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)

    print("\n=== Test Evaluation: Ridge vs RF ===")
    print(f"Test file : {TEST_PATH} (rows={rep.n_rows_after})")
    print(f"Saved long: {OUT_LONG}")
    print(f"Saved wide: {OUT_WIDE}\n")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()

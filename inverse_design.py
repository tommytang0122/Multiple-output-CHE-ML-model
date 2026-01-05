# inverse_design.py (FAST + time budget + progress + batch scoring)
#
# Usage examples:
#   python inverse_design.py --spec specs.json
#   python inverse_design.py --spec specs.json --time_budget 90
#   python inverse_design.py --spec specs.json --maxiter 50 --popsize 10 --samples 8000
#
# Output:
#   reports/inverse_solutions.csv

from __future__ import annotations

from pathlib import Path
import json
import math
import argparse
import time
import numpy as np
import pandas as pd
import joblib

from utils.data_guard import load_xy_csv

# ---------------------------
# Paths
# ---------------------------
TRAIN_PATH = Path("data/train.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
OUT_CSV = REPORTS_DIR / "inverse_solutions.csv"

# ---------------------------
# Model bundle (final decision)
# ---------------------------
MODEL_BUNDLE = {
    "y1": MODELS_DIR / "rf_y1.joblib",
    "y2": MODELS_DIR / "rf_y2.joblib",
    "y3": MODELS_DIR / "rf_y3.joblib",
    "y4": MODELS_DIR / "rf_y4.joblib",
    "y5": MODELS_DIR / "rf_y5.joblib",      # soft by default
    "y6": MODELS_DIR / "ridge_y6.joblib",   # best is ridge on test
    "y7": MODELS_DIR / "rf_y7.joblib",
}

# ---------------------------
# Default specs placeholder (you usually pass --spec)
# ---------------------------
DEFAULT_SPECS = {
    "y1": {"type": "min", "lo": 80.0, "hard": True},
    "y2": {"type": "min", "lo": 80.0, "hard": True},
    "y3": {"type": "range", "lo": 0.2, "hi": 50.0, "hard": True},
    "y4": {"type": "range", "lo": 0.55, "hi": 0.65, "hard": True},
    "y5": {"type": "max", "hi": 10.0, "hard": False},
    "y6": {"type": "range", "lo": 0.2, "hi": 50.0, "hard": True},
}

# Penalty settings
LAMBDA = 50.0
PENALTY_POWER = 2.0
HARD_PENALTY = 1e6

# ---------------------------
# Utilities
# ---------------------------

def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    art = joblib.load(path)
    if "model" not in art:
        raise KeyError(f"Invalid model artifact (missing 'model') in: {path}")
    return art


def load_specs(spec_path: str | None) -> dict:
    if spec_path is None:
        return DEFAULT_SPECS
    p = Path(spec_path)
    if not p.exists():
        raise FileNotFoundError(f"Spec file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        spec = json.load(f)
    # default hard=True except y5
    for k, v in spec.items():
        if "hard" not in v:
            v["hard"] = (k != "y5")
    return spec


def penalty_for_spec(y: np.ndarray, spec: dict) -> np.ndarray:
    """
    Vectorized penalty for array y.
    Returns array >= 0.
    """
    t = spec["type"]
    if t == "range":
        lo, hi = float(spec["lo"]), float(spec["hi"])
        below = np.clip(lo - y, 0.0, None)
        above = np.clip(y - hi, 0.0, None)
        return (below ** PENALTY_POWER) + (above ** PENALTY_POWER)
    elif t == "min":
        lo = float(spec["lo"])
        below = np.clip(lo - y, 0.0, None)
        return below ** PENALTY_POWER
    elif t == "max":
        hi = float(spec["hi"])
        above = np.clip(y - hi, 0.0, None)
        return above ** PENALTY_POWER
    else:
        raise ValueError(f"Unknown spec type: {t}")


def build_bounds_from_train(train_df: pd.DataFrame, x_cols: list[str]) -> list[tuple[float, float]]:
    bounds = []
    for c in x_cols:
        lo = float(train_df[c].min())
        hi = float(train_df[c].max())
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"Non-finite bound for {c}: lo={lo}, hi={hi}")
        if hi <= lo:
            hi = lo + 1e-9
        bounds.append((lo, hi))
    return bounds


def random_uniform_samples(bounds: list[tuple[float, float]], n: int, rng: np.random.Generator) -> np.ndarray:
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    u = rng.random((n, len(bounds)))
    return lows + u * (highs - lows)


def batch_predict(artifact: dict, X: np.ndarray) -> np.ndarray:
    """
    X: (n, d)
    Returns (n,)
    """
    model = artifact["model"]
    y = model.predict(X)
    y = np.asarray(y).reshape(-1)
    return y.astype(float)


def evaluate_pool_batch(pool_x: np.ndarray, artifacts: dict, specs: dict, x_cols: list[str]) -> pd.DataFrame:
    """
    Batch-evaluate a pool of candidate X.
    Much faster than per-row predict loops.
    """
    y_preds = {}
    for yk in ["y1","y2","y3","y4","y5","y6","y7"]:
        y_preds[yk] = batch_predict(artifacts[yk], pool_x)

    # penalties
    total_pen = np.zeros(pool_x.shape[0], dtype=float)
    hard_viol = np.zeros(pool_x.shape[0], dtype=int)

    for yk, spec in specs.items():
        pen = penalty_for_spec(y_preds[yk], spec)
        total_pen += pen
        if spec.get("hard", True):
            hard_viol += (pen > 0.0).astype(int)

    feasible = (hard_viol == 0)

    score = y_preds["y7"] + LAMBDA * total_pen + HARD_PENALTY * hard_viol

    # build dataframe
    df = pd.DataFrame({c: pool_x[:, i] for i, c in enumerate(x_cols)})
    for k, v in y_preds.items():
        df[k] = v
    df["feasible"] = feasible
    df["hard_violations"] = hard_viol
    df["total_penalty"] = total_pen
    df["score"] = score

    df = df.sort_values(by=["feasible", "y7", "score"], ascending=[False, True, True]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, default=None, help="Path to JSON spec file")
    parser.add_argument("--topn", type=int, default=30, help="How many solutions to export")
    parser.add_argument("--time_budget", type=float, default=90.0, help="Seconds. Stop DE when time is up.")
    parser.add_argument("--maxiter", type=int, default=60, help="DE max iterations (generations)")
    parser.add_argument("--popsize", type=int, default=15, help="DE popsize (SciPy uses popsize*dim population)")
    parser.add_argument("--samples", type=int, default=8000, help="Random pool samples for final ranking (batch)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load train for bounds
    train_df, x_cols, _y_cols, rep = load_xy_csv(TRAIN_PATH, drop_na=True, min_rows=10)
    bounds = build_bounds_from_train(train_df, x_cols)

    # Load specs
    specs = load_specs(args.spec)

    # Load model artifacts
    artifacts = {yk: load_artifact(path) for yk, path in MODEL_BUNDLE.items()}

    # Safety: check x_cols consistency
    for yk, art in artifacts.items():
        if "x_cols" in art and art["x_cols"] != x_cols:
            raise ValueError(f"Model {yk} x_cols mismatch vs train.csv x_cols.")

    rng = np.random.default_rng(args.seed)

    # Objective for DE (single-point, not batch)
    # Add progress printing so you KNOW it's running.
    start_time = time.time()
    eval_count = 0
    best_seen = float("inf")

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count, best_seen
        eval_count += 1
        
        if eval_count == 1:
            print("[debug] objective() entered")

        # predict (single row)
        x2 = np.asarray(x, dtype=float).reshape(1, -1)
        y1 = float(batch_predict(artifacts["y1"], x2)[0])
        y2 = float(batch_predict(artifacts["y2"], x2)[0])
        y3 = float(batch_predict(artifacts["y3"], x2)[0])
        y4 = float(batch_predict(artifacts["y4"], x2)[0])
        y5 = float(batch_predict(artifacts["y5"], x2)[0])
        y6 = float(batch_predict(artifacts["y6"], x2)[0])
        y7 = float(batch_predict(artifacts["y7"], x2)[0])

        # penalties
        total_pen = 0.0
        hard_viol = 0

        pred_map = {"y1": y1, "y2": y2, "y3": y3, "y4": y4, "y5": y5, "y6": y6}
        for yk, spec in specs.items():
            # scalar penalty
            pen = float(penalty_for_spec(np.array([pred_map[yk]]), spec)[0])
            total_pen += pen
            if spec.get("hard", True) and pen > 0.0:
                hard_viol += 1

        score = y7 + LAMBDA * total_pen + HARD_PENALTY * hard_viol

        if score < best_seen:
            best_seen = score

        # progress log every 10 evals
        if eval_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[progress] evals={eval_count:,} elapsed={elapsed:.1f}s best_score={best_seen:.6f}")

        return score

    best_x = None
    used_method = "random_only"

    # Differential Evolution with time budget stop
    try:
        from scipy.optimize import differential_evolution  # type: ignore

        def cb(xk, convergence):
            # stop when time budget reached
            return (time.time() - start_time) >= float(args.time_budget)

        print("=== Running Differential Evolution ===")
        print(f"maxiter={args.maxiter}, popsize={args.popsize}, time_budget={args.time_budget}s (will stop when time is up)")
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=int(args.maxiter),
            popsize=int(args.popsize),
            seed=int(args.seed),
            polish=False,              # IMPORTANT: prevents slow local search at end
            updating="deferred",
            workers=1,                 # keep 1 to avoid Windows multiprocess hangs
            callback=cb,
        )
        best_x = np.asarray(result.x, dtype=float)
        used_method = "differential_evolution"
        print(f"[done] DE finished/stopped. evals={eval_count:,} elapsed={time.time()-start_time:.1f}s best_score={float(result.fun):.6f}")
    except Exception as e:
        print(f"[warn] DE unavailable/failed: {e}")
        print("[warn] Using random search only.")

    # Build random pool (batch-evaluated FAST)
    pool = random_uniform_samples(bounds, int(args.samples), rng)
    if best_x is not None:
        pool = np.vstack([best_x.reshape(1, -1), pool])

    print(f"=== Scoring pool (batch) size={len(pool):,} ===")
    out_df = evaluate_pool_batch(pool, artifacts, specs, x_cols)

    topn = int(args.topn)
    out_df.head(topn).to_csv(OUT_CSV, index=False)

    feasible_n = int(out_df["feasible"].sum())
    print("\n=== Inverse Design Results ===")
    print(f"Method used: {used_method}")
    print(f"Exported top-{topn} to: {OUT_CSV}")
    print(f"Feasible in pool: {feasible_n} / {len(out_df)}")

    best_row = out_df.iloc[0].to_dict()
    keys_show = ["feasible", "hard_violations", "total_penalty", "score", "y1","y2","y3","y4","y5","y6","y7"]
    print("\n--- Best candidate (row 1) ---")
    for k in keys_show:
        print(f"{k}: {best_row[k]}")

    if feasible_n == 0:
        print("\n[diagnosis] No feasible solutions found.")
        print("Possible fixes:")
        print("- Increase --samples (e.g. 50000) to explore more, OR")
        print("- Relax some hard constraints (especially y1/y2/y4) or increase buffer, OR")
        print("- Expand x bounds beyond train min/max (careful: model extrapolation risk).")


if __name__ == "__main__":
    main()

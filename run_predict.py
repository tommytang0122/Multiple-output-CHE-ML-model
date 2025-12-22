import joblib
import pandas as pd
import numpy as np

ckpt = joblib.load("models/rf_y5_baseline.joblib")

model = ckpt["model"]
features = ckpt["features"]
use_log = ckpt["transform"] == "log1p_then_invert"

X = pd.read_csv("test_data.csv")
X = X[features]

y_hat = model.predict(X)

if use_log:
    y_hat = np.expm1(y_hat)

print(y_hat[:5])

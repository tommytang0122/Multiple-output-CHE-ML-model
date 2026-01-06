# Multiple-Output CHE ML Model - Project Summary

## Overview

This is a **Chemical Engineering (CHE) Multi-Output Machine Learning** project that predicts 7 target properties from 37 input features and performs inverse design optimization.

## Project Structure

### Data Pipeline

#### 1. cleandata.py
Preprocesses raw data from `data/rawdata.csv`:
- Takes 39 raw features (columns A-AM)
- Merges paired features: AI+AJ → AI_AJ, AK+AL → AK_AL
- Results in 37 features (x1-x37)
- Extracts 7 targets (columns AN-AT) → y1-y7
- Outputs to `data/cleandata.csv`

#### 2. make_train_test.py
Splits cleaned data into training and testing sets:
- 80% training data → `data/train.csv`
- 20% testing data → `data/test.csv`
- Uses random_state=42 for reproducibility

### Model Training

#### 3. train_ridge_7.py
Trains 7 separate Ridge regression models (one per target):
- Pipeline: StandardScaler + Ridge (alpha=1.0)
- Saves models to `models/ridge_y*.joblib`
- Outputs training metrics to `reports/ridge_train_metrics.csv`

#### 4. train_rf_7.py
Trains 7 Random Forest regression models (one per target):
- Configuration: 800 trees, max_depth=None, n_jobs=-1
- Saves models to `models/rf_y*.joblib`
- Outputs training metrics to `reports/rf_train_metrics.csv`

### Model Evaluation

#### 5. eval_models_test.py
Compares Ridge vs Random Forest performance on test data:
- Calculates RMSE, MAE, R² for each target
- Determines which model performs better per target
- Outputs two comparison reports:
  - `reports/eval_test_ridge_vs_rf_by_target.csv` (wide format)
  - `reports/eval_test_ridge_vs_rf_long.csv` (long format)
- Shows winners and deltas for each metric

### Inverse Design (Optimization)

#### 6. inverse_design.py
Finds optimal input values (x1-x37) that meet target specifications:
- **Objective**: Minimize y7 while satisfying constraints on y1-y6
- **Method**: Differential evolution optimization + random sampling
- **Model Bundle**: Uses best-performing models per target (RF for most, Ridge for y6)
- **Constraints** (from `specs.json`):
  - y1: range [81.674, 89.666] (hard)
  - y2: range [94.298, 96.042] (hard)
  - y3: range [2.547, 2.773] (hard)
  - y4: range [0.61434, 0.62966] (hard)
  - y5: max ≤ 30.2 (soft)
  - y6: range [2.364, 2.776] (hard)
- **Output**: Top solutions to `reports/inverse_solutions.csv`

**Usage:**
```bash
python inverse_design.py --spec specs.json
python inverse_design.py --spec specs.json --time_budget 90 --samples 8000
```

### Utilities

#### 7. utils/data_guard.py
Robust CSV loading and validation utilities:
- Column existence checking
- Automatic numeric type conversion
- Missing value handling
- Minimum row requirements
- Consistent error reporting

## Data Schema

**Features**: x1, x2, ..., x37 (37 continuous features)

**Targets**: y1, y2, ..., y7 (7 continuous target properties)

## Key Configuration Files

- **specs.json**: Target specifications for inverse design optimization
- **.gitignore**: Git ignore patterns

## Typical Workflow

1. Clean raw data: `python cleandata.py`
2. Split into train/test: `python make_train_test.py`
3. Train models: `python train_ridge_7.py` and `python train_rf_7.py`
4. Evaluate models: `python eval_models_test.py`
5. Run inverse design: `python inverse_design.py --spec specs.json`

## Output Directories

- `data/`: Raw, cleaned, train, and test datasets
- `models/`: Trained model artifacts (.joblib files)
- `reports/`: Evaluation metrics and inverse design solutions
- `logs/`: Training logs (timestamped)


## Ridge Training Metrics Explanation

  This file shows the training performance of the 7 Ridge regression models. Each row represents one target (y1-y7), and the metrics tell you how well the model fits the training data.

  Columns:

  target: Which output variable (y1-y7) the model predicts

  train_rmse (Root Mean Squared Error): Average prediction error
  - Lower is better
  - Same units as the target
  - Penalizes large errors more heavily

  train_mae (Mean Absolute Error): Average absolute prediction error
  - Lower is better
  - More interpretable than RMSE (average "how far off" in raw units)

  train_r2 (R² Score): How much variance the model explains
  - Range: 0 to 1 (can be negative for very bad models)
  - Higher is better
  - 1.0 = perfect fit, 0.0 = model no better than predicting the mean

  model_path: Location of the saved model file

  Performance Breakdown:

  | Target | RMSE  | MAE   | R²    | Interpretation                             |
  |--------|-------|-------|-------|--------------------------------------------|
  | y1     | 1.256 | 1.012 | 0.794 | Good fit - explains 79.4% of variance      |
  | y2     | 0.251 | 0.202 | 0.849 | Very good fit - explains 84.9% of variance |
  | y3     | 0.043 | 0.035 | 0.898 | Excellent fit - explains 89.8% of variance |
  | y4     | 0.003 | 0.003 | 0.633 | Weakest - only 63.3% variance explained    |
  | y5     | 2.526 | 1.818 | 0.753 | Decent fit - 75.3% variance explained      |
  | y6     | 0.058 | 0.044 | 0.994 | Best - nearly perfect 99.4% fit!           |
  | y7     | 0.009 | 0.007 | 0.751 | Decent fit - 75.1% variance explained      |

  Key Insights:

  1. y6 is the easiest to predict with Ridge (R²=0.994) - almost perfect linear relationship
  2. y4 is the hardest to predict (R²=0.633) - might have nonlinear patterns Ridge can't capture
  3. These are training metrics - they tend to be optimistic. Check eval_test_ridge_vs_rf_by_target.csv to see how Ridge performs on unseen test data
  4. If Random Forest significantly outperforms Ridge on a target, it suggests nonlinear relationships that linear models can't capture


## Random Forest Training Metrics Explanation

  This file shows the training performance of the 7 Random Forest models (800 trees each). Same format as the Ridge metrics, but with much better performance overall.

  Performance Breakdown:

  | Target | RMSE  | MAE   | R²    | Interpretation                       |
  |--------|-------|-------|-------|--------------------------------------|
  | y1     | 0.707 | 0.566 | 0.935 | Excellent - 93.5% variance explained |
  | y2     | 0.138 | 0.113 | 0.954 | Excellent - 95.4% variance explained |
  | y3     | 0.026 | 0.019 | 0.965 | Excellent - 96.5% variance explained |
  | y4     | 0.002 | 0.001 | 0.913 | Excellent - 91.3% variance explained |
  | y5     | 1.496 | 1.113 | 0.913 | Excellent - 91.3% variance explained |
  | y6     | 0.031 | 0.021 | 0.998 | Nearly perfect - 99.8% fit!          |
  | y7     | 0.004 | 0.003 | 0.946 | Excellent - 94.6% variance explained |

  Random Forest vs Ridge Comparison:

  | Target | Ridge R² | RF R² | RF Improvement | Winner      |
  |--------|----------|-------|----------------|-------------|
  | y1     | 0.794    | 0.935 | +0.141         | RF          |
  | y2     | 0.849    | 0.954 | +0.105         | RF          |
  | y3     | 0.898    | 0.965 | +0.067         | RF          |
  | y4     | 0.633    | 0.913 | +0.280 🔥      | RF          |
  | y5     | 0.753    | 0.913 | +0.160         | RF          |
  | y6     | 0.994    | 0.998 | +0.004         | RF (barely) |
  | y7     | 0.751    | 0.946 | +0.195         | RF          |

  Key Insights:

  1. Random Forest dominates on training data - beats Ridge on all 7 targets
  2. y4 shows the biggest improvement (0.633 → 0.913) - Ridge struggled with this, RF handles the nonlinearity
  3. y6 is easy for both models but RF still edges out Ridge (0.998 vs 0.994)
  4. RF error metrics are much lower:
    - y1: RMSE 0.707 (RF) vs 1.256 (Ridge) - 44% reduction
    - y4: RMSE 0.002 (RF) vs 0.003 (Ridge) - 51% reduction

  Important Caveat:

  These are training metrics - RF tends to overfit more than Ridge. The critical question is: Do these gains hold on test data?

  Check eval_test_ridge_vs_rf_by_target.csv to see if RF's superior training performance translates to better test performance, or if it's overfitting.
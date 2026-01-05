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

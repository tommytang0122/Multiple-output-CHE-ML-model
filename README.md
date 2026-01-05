# Multiple-Output CHE ML Model

This repository contains a specialized machine learning pipeline for **Chemical Engineering (CHE)** applications. It leverages multi-output regression to predict seven distinct target properties and features an optimization engine for inverse material design.

## 🛠 Project Structure

The project is organized into a systematic workflow for data processing, model training, and parameter optimization:

- **`data/`**: Storage for raw, cleaned, and split (train/test) datasets.
- **`models/`**: Stores trained `.joblib` artifacts for both Ridge and Random Forest models.
- **`reports/`**: Contains evaluation metrics, comparison CSVs, and optimization results.
- **`utils/`**: Core utilities including `data_guard.py` for robust CSV loading and validation.

## 🚀 Workflow & Pipeline

### 1. Data Processing
- **`cleandata.py`**: Merges paired features (AI+AJ and AK+AL) to create a standardized set of 37 features ($x_1$–$x_37$) and 7 targets ($y_1$–$y_7$).
- **`make_train_test.py`**: Splits the cleaned data into an 80% training set and a 20% testing set.

### 2. Model Training
The pipeline evaluates two distinct approaches for each target:
- **Ridge Regression**: Linear modeling with $L_2$ regularization to handle multicollinearity.
- **Random Forest**: Ensemble modeling with 800 decision trees to capture complex non-linear interactions.

### 3. Evaluation
- **`eval_models_test.py`**: Compares both algorithms across targets. Currently, Random Forest is the preferred model for most targets, while Ridge is used for $y_6$ based on test performance.

### 4. Inverse Design (Optimization)
- **`inverse_design.py`**: Uses **Differential Evolution** and random sampling to identify optimal input values.
- **Objective**: Minimize $y_7$ while satisfying specific range or threshold constraints for $y_1$ through $y_6$ as defined in `specs.json`.

## 📊 Technical Requirements

```bash
pip install pandas numpy scikit-learn joblib scipy
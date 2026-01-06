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


Knowledge to Show Off During Your PresentationTo impress your audience, you can highlight these specific technical concepts:Multi-Output Regression vs. Single-Output: Explain that instead of one giant model, you built seven specialized models to account for the unique behavior of each chemical property. This ensures higher accuracy for each specific target.The Bias-Variance Tradeoff: Discuss why you compared Ridge Regression (high bias, low variance) with Random Forest (low bias, high variance). Mention that Ridge uses $L_2$ Regularization to handle multicollinearity in your 37 features, preventing the model from becoming too sensitive to noise.Ensemble Learning Mechanics: Briefly explain that your Random Forest uses 800 decision trees to "vote" on the final prediction, which captures complex non-linear interactions between chemical inputs that a simple linear model would miss.Inverse Design via Differential Evolution: This is a major "show-off" point. Explain that you aren't just predicting output; you are using an optimization algorithm (Differential Evolution) to work backward from a desired result to find the exact starting "recipe" ($x_1$-$x_{37}$).Data Integrity (Data Guard): Mention your use of a "Data Guard" utility to handle missing values and ensure type consistency. This demonstrates a professional "Production-Ready" mindset rather than just a basic academic approach.Presentation Tip (Traditional Chinese Translation)If presenting in a bilingual environment, use these terms to sound more professional:Parity Plot: 一致性對比圖Generalization: 泛化能力 (the model's ability to handle new data)Multicollinearity: 多重共線性Hold-out set: 留出集 (referring to your test file)


在這個專案中，get_feature_importance 對於 隨機森林 (Random Forest) 和 脊迴歸 (Ridge Regression) 是基於完全不同的數學原理來運作的。以下是這兩種模型比較特徵重要性的核心原理說明：1. 隨機森林 (Random Forest) 的原理隨機森林的特徵重要性通常被稱為 「不純度減少平均值」 (Mean Decrease in Impurity, MDI)。核心概念：隨機森林是由數百棵決策樹組成的。在每一棵樹的每一個節點進行分裂（Split）時，演算法都會尋找一個特徵 ($x_i$) 和一個切分點，目的是讓分裂後的子節點的「不純度」降到最低。在迴歸問題中：所謂的「不純度」通常是指 MSE (均方誤差) 或 變異數 (Variance)。如果某個特徵 ($x_{37}$) 被用來分裂節點後，能大幅度降低該群資料的 MSE（讓預測變得更準確、更集中），那麼這個特徵就被認為很重要。計算方式：模型會計算每個特徵在所有 800 棵樹中，總共貢獻了多少「MSE 的減少量」。減少得越多，重要性分數就越高。優點：能捕捉到非線性關係和特徵間的交互作用。2. 脊迴歸 (Ridge Regression) 的原理脊迴歸的特徵重要性是基於 「迴歸係數的絕對值」 (Absolute Magnitude of Coefficients)。核心概念：脊迴歸是一個線性模型，其公式為：$$y = w_1 \cdot x_1 + w_2 \cdot x_2 + ... + b$$其中 $w$ 就是係數 (Weight/Coefficient)。標準化的關鍵作用：在您的 train_ridge_7.py 中，使用了 StandardScaler。這意味著所有的特徵 ($x_1$ 到 $x_{37}$) 都被縮放到相同的尺度（平均值為 0，標準差為 1）。原理：因為所有特徵的單位都統一了，係數 $w$ 的大小直接代表了該特徵對 $y$ 的敏感度。如果 $|w_1| = 5$，代表 $x_1$ 每變動 1 個標準差，$y$ 就會變動 5 個單位。如果 $|w_2| = 0.1$，代表 $x_2$ 對 $y$ 幾乎沒有影響。計算方式：直接取模型學到的係數 $w$ 的絕對值 ($|w|$) 來排序。絕對值越大，越重要。



模型,重要性判斷依據,物理意義
隨機森林,MSE 減少量 (貢獻度),該特徵在「區分數據」或「降低預測誤差」上的貢獻有多大？ (適合非線性)
脊迴歸,係數絕對值 (權重),該特徵每變動一個單位，目標值會隨之變動多少？ (適合線性，前提是必須標準化)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import joblib

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def get_feature_importance(artifact, model_type):
    """
    從模型 artifact 中提取特徵重要性
    """
    model = artifact["model"]
    feature_names = artifact["x_cols"]
    
    importances = None
    
    if model_type == 'rf':
        # Random Forest: 直接使用 feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
    
    elif model_type == 'ridge':
        # Ridge (Pipeline): 提取係數 (Coefficients)
        # 因為有 StandardScaler，係數的絕對值可視為重要性
        if hasattr(model, 'named_steps') and 'ridge' in model.named_steps:
            ridge_model = model.named_steps['ridge']
            # 取絕對值來比較重要性 (Magnitude)
            importances = np.abs(ridge_model.coef_)
            
            # 若 y 是多維的，coef_ 形狀會不同，但此專案是單一目標 (y1..y7 分開訓練)，coef_ 應為 1D
            if importances.ndim > 1:
                importances = importances.ravel()

    return feature_names, importances

def plot_feature_importance(model_type, target):
    # 1. 建構模型路徑
    # 根據您的檔案結構： models/rf_y1.joblib 或 models/ridge_y1.joblib
    model_filename = f"{model_type}_{target}.joblib"
    model_path = os.path.join('models', model_filename)
    
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型檔案 {model_path}")
        print("請確認 'models' 資料夾中是否有對應的 .joblib 檔案。")
        return

    # 2. 載入模型
    print(f"正在載入模型：{model_path} ...")
    try:
        artifact = joblib.load(model_path)
    except Exception as e:
        print(f"錯誤：無法載入模型。詳細資訊：{e}")
        return

    # 3. 提取特徵重要性
    feature_cols, importances = get_feature_importance(artifact, model_type)

    if importances is None:
        print(f"錯誤：無法從 {model_type} 模型中提取特徵重要性。")
        return

    # 4. 排序並取前 10 名 (Top 10)
    indices = np.argsort(importances)[-10:]
    
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]

    # 5. 繪製圖表
    plt.figure(figsize=(10, 6))
    model_display_name = "Random Forest" if model_type == 'rf' else "Ridge Regression (Abs Coef)"
    plt.title(f'Top 10 Feature Importances for {target} ({model_display_name})')
    
    color = 'mediumseagreen' if model_type == 'rf' else 'cornflowerblue'
    plt.barh(range(len(indices)), top_importances, color=color, align='center')
    plt.yticks(range(len(indices)), top_features)
    
    xlabel_text = 'Relative Importance' if model_type == 'rf' else 'Coefficient Magnitude (Absolute)'
    plt.xlabel(xlabel_text)
    
    # 標示數值
    for i, v in enumerate(top_importances):
        plt.text(v, i, f' {v:.4f}', va='center', fontweight='bold', fontsize=9)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 6. 存檔
    output_filename = f'feature_importance_{model_type}_{target}.png'
    plt.savefig(output_filename)
    print(f"成功！圖表已儲存為：{output_filename}")

if __name__ == "__main__":
    # 使用說明： python importance_plot.py [rf/ridge] [y1-y7]
    if len(sys.argv) < 3:
        print("使用方式錯誤。")
        print("請輸入： python importance_plot_v2.py [model_type] [target]")
        print("範例：   python importance_plot_v2.py rf y1")
        print("範例：   python importance_plot_v2.py ridge y6")
    else:
        m_type = sys.argv[1].lower().strip()
        t_name = sys.argv[2].lower().strip()
        
        if m_type not in ['rf', 'ridge']:
            print(f"錯誤：不支援的模型類型 '{m_type}'。請使用 'rf' 或 'ridge'。")
        else:
            plot_feature_importance(m_type, t_name)
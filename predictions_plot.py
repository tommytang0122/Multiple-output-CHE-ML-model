import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_parity(target):
    # 檔案路徑設定
    csv_path = 'reports/test_predictions.csv'
    
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到檔案 {csv_path}")
        return

    # 讀取數據
    df = pd.read_csv(csv_path)
    
    actual_col = target
    pred_col = f"{target}_pred"
    
    # 檢查欄位是否存在
    if actual_col not in df.columns or pred_col not in df.columns:
        print(f"錯誤：在 CSV 中找不到目標 '{target}'。可用目標：y1-y7")
        return

    actual = df[actual_col]
    predicted = df[pred_col]

    # 繪製圖表
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.6, color='royalblue', label='Data Points')
    
    # 繪製 45 度理想參考線
    lims = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='perfect line (y=x)')
    
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'Parity Plot {target}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 儲存圖片
    output_name = f'parity_plot_{target}.png'
    plt.savefig(output_name)
    print(f"圖表已成功儲存為：{output_name}")

if __name__ == "__main__":
    # 優先檢查命令行參數 (例如: python predictions_plot.py y1)
    if len(sys.argv) > 1:
        target_name = sys.argv[1].strip()
    else:
        # 若無參數則使用標準輸入 (stdin)
        target_name = sys.stdin.read().strip()
    
    if target_name:
        plot_parity(target_name)
    else:
        print("請提供目標名稱（如 y1, y2...）")
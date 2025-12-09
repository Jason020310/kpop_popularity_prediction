# Group 14 Project Contribution Report (Jason's Part)

## 1. 我的主要貢獻 (What I Did)

### A. 救回 92% 的資料 (Data Rescue)
- **問題**：原本的程式因為 `English_percent` 有缺失值，導致 dropna() 刪掉了 1067 筆資料，只剩下 68 筆訓練，模型根本學不到東西。
- **解決**：我寫了程式自動填補缺失值 (Imputation)，把訓練資料量救回 **900多筆**。

### B. 模型擴充 (Model Expansion)
- 除了原本的 Linear Regression 和 Random Forest，我多做了：
    - **XGBoost**
    - **LightGBM**
- 寫了一個腳本 `compare_models.py` 一次跑完這 4 個模型並畫圖比較。

### C. 參數調校 (Optimization)
- 用 `tune_models.py` 幫 Random Forest 做超參數最佳化 (Hyperparameter Tuning)。
- 把 R2 分數從 0.06 提升到 **0.09** (這是純音訊資料的極限)。

### D. 策略轉向 (Pivot to Classification) - **最重要的亮點**
- **發現**：預測「第幾名」太難 (R2=0.09)。
- **創新**：我改成預測「是否為熱門歌曲 (Top 50)」。
- **成果**：寫了 `classification_pivot.py`，準確率達到 **71.4%**。這可以當作我們這組的 "Success Story"。

---

## 2. 如何執行我的程式 (How to Run)

請同學確認有安裝套件：`pip install xgboost lightgbm seaborn`

### 步驟 1: 跑回歸模型比較 (Regression)
執行指令：
```bash
python3 compare_models.py
```
**產出**：
- `model_comparison.png` (看這張圖就知道哪個模型最好)
- `model_comparison_results.csv` (詳細數據)

### 步驟 2: 跑分類模型 (Classification - 我們的亮點)
執行指令：
```bash
python3 classification_pivot.py
```
**產出**：
- Terminal 會顯示 **Accuracy: 0.7143**
- `classification_confusion_matrix.png` (證明我們預測很準的圖)

### 步驟 3: 看特徵分析 (Feature Analysis)
執行指令：
```bash
python3 analyze_features.py
```
**產出**：
- `imputed_feature_importance.png` (告訴我們哪些特徵最重要，例如 duration, acousticness)

---

## 3. 建議上傳檔案 (Files to Commit)

建議把這些檔案 push 上去，因為它們是完整的一套工作流：

1.  `compare_models.py` (核心腳本)
2.  `classification_pivot.py` (分類腳本)
3.  `analyze_features.py` (分析腳本)
4.  `tune_models.py` (調校腳本)
5.  `analyze_data_quality.py` (證明我們有檢查資料)
6.  所有 `.png`圖檔 (報告直接用)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import pandas as pd

# 讀取 CSV
df = pd.read_csv("../dataset/combined_kpop_dataset.csv")  # 替換成你的路徑

# target
df['top35'] = (df['avg_rank'] <= 35).astype(int)

# features
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'time_signature', 'eng_line_ratio',
            'eng_word_ratio', 'company', 'gender']

X = df[features]
y = df['top35']

# categorical encoding
categorical_features = ['company', 'gender']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cat = encoder.fit_transform(X[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))

X = X.drop(columns=categorical_features).reset_index(drop=True)
X = pd.concat([X, encoded_cat_df], axis=1)

# standardize for linear model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# 計算 class imbalance
n_pos = sum(y_train == 1)
n_neg = sum(y_train == 0)
scale_pos_weight = n_neg / n_pos

# 建立模型
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42),
    "LightGBM": lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
}

# 訓練與評估
plt.figure(figsize=(14, 6))
for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]  # positive class probability
    
    print(f"===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, len(models), i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} CM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.subplot(2, len(models), i + len(models))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"{name} ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

plt.tight_layout()
plt.savefig("classifier_comparison.png")
plt.show()

# -- Feature Importance Visualization --
def plot_top_features(importances, features_names, model_name, top_n=10):
    df = pd.DataFrame({'feature': features_names, 'importance': importances})
    df = df.sort_values(by='importance', ascending=False).head(top_n)

    # Print top features to terminal
    print(f"\nTop {top_n} features for {model_name}:")
    print(df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.barh(df['feature'], df['importance'])
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} Top {top_n} Feature Importance")
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.show()

features_names = X.columns

# Logistic Regression (abs(coefficients))
coef_importance = abs(models['Logistic Regression'].coef_[0])
plot_top_features(coef_importance, features_names, "Logistic Regression")

# Random Forest
rf_importance = models['Random Forest'].feature_importances_
plot_top_features(rf_importance, features_names, "Random Forest")

# XGBoost
xgb_importance = models['XGBoost'].feature_importances_
plot_top_features(xgb_importance, features_names, "XGBoost")

# LightGBM
lgb_importance = models['LightGBM'].feature_importances_
plot_top_features(lgb_importance, features_names, "LightGBM")
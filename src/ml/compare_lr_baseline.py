import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/features.csv")
y = df["label"].values
X = df.drop(columns=["label", "source_file"])
X["method"] = X["method"].astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- LightGBM (existing model) ---
lgbm = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    random_state=42, scale_pos_weight=1.5, n_jobs=1, verbose=-1
)
start = time.perf_counter()
lgbm.fit(X_train, y_train)
lgbm_train_time = time.perf_counter() - start
lgbm_preds = lgbm.predict(X_test)

# --- Logistic Regression (linear baseline) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight={0: 1.0, 1: 1.5}, random_state=42)
start = time.perf_counter()
lr.fit(X_train_scaled, y_train)
lr_train_time = time.perf_counter() - start
lr_preds = lr.predict(X_test_scaled)

print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Train time (s)':>16}")
for name, preds, t in [("LightGBM", lgbm_preds, lgbm_train_time), ("Logistic Regression", lr_preds, lr_train_time)]:
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {t:>16.3f}")
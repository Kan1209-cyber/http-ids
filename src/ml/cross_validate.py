import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/features.csv")
y = df["label"].values
X = df.drop(columns=["label", "source_file"])
X["method"] = X["method"].astype("category").cat.codes
X = X.values

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

precisions, recalls, f1s = [], [], []

print(f"Running {N_FOLDS}-fold stratified cross-validation...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, scale_pos_weight=1.5, n_jobs=1, verbose=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

    print(f"Fold {fold}: precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

def mean_ci95(values):
    values = np.array(values)
    mean = values.mean()
    std = values.std(ddof=1)
    # 95% CI using t-distribution critical value for n=5 (df=4) ~= 2.776
    margin = 2.776 * (std / np.sqrt(len(values)))
    return mean, std, margin

print("\n=== Cross-Validation Summary (mean ± 95% CI) ===")
for name, vals in [("Precision", precisions), ("Recall", recalls), ("F1-score", f1s)]:
    mean, std, margin = mean_ci95(vals)
    print(f"{name}: {mean:.4f} ± {margin:.4f}  (std={std:.4f})")
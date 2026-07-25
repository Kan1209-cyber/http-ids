import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/features.csv")
y = df["label"]
X = df.drop(columns=["label", "source_file"])
X["method"] = X["method"].astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weights = [1.0, 1.5, 2.0, 2.5, 2.87]

print(f"{'weight':>8} {'precision':>10} {'recall':>8} {'f1':>8} {'FPR':>8}")
for w in weights:
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, scale_pos_weight=w, verbose=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    tn = ((y_test == 0) & (preds == 0)).sum()
    fp = ((y_test == 0) & (preds == 1)).sum()
    fpr = fp / (fp + tn)

    print(f"{w:>8} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f} {fpr:>8.3f}")
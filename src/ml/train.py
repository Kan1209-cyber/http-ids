import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = "data/processed/features.csv"
MODEL_OUT = "src/ml/lgbm_model.pkl"

CATEGORICAL_COLS = ["method"]
DROP_COLS = ["label", "source_file"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df["label"]
    X = df.drop(columns=DROP_COLS)

    # Encode method (GET/POST/etc.) as categorical codes
    X["method"] = X["method"].astype("category").cat.codes

    return X, y


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=1.5,
        n_jobs=1,  # pin thread count — avoids per-call cpu_count() detection overhead
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=["normal", "malicious"]))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    print("\n=== Feature Importances ===")
    importances = sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda x: -x[1]
    )
    for name, score in importances:
        print(f"  {name}: {score}")

    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

    return model, X_test, y_test


if __name__ == "__main__":
    train()
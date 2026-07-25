import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from src.features.request_parser import parse_requests_from_file
from src.features.extractor import extract_features

FILES = {
    "data/raw/normalTrafficTraining.txt": 0,   # 0 = normal
    "data/raw/normalTrafficTest.txt": 0,
    "data/raw/anomalousTrafficTest.txt": 1,    # 1 = malicious
}

def build():
    rows = []
    for filepath, label in FILES.items():
        requests = parse_requests_from_file(filepath)
        for req in requests:
            feats = extract_features(req)
            feats["label"] = label
            feats["source_file"] = filepath
            rows.append(feats)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df["label"].value_counts())
    print(df.groupby("source_file")["label"].count())

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print("\nSaved to data/processed/features.csv")
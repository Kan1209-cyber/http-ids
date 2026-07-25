import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lgbm_model.pkl")

FEATURE_COLUMNS = [
    "method", "path_length", "query_length", "body_length", "content_length",
    "num_params", "num_headers", "special_char_count", "content_entropy",
    "sqli_keyword_count", "xss_keyword_count", "path_traversal_keyword_count",
    "max_param_length",
]

METHOD_CODES = {"GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "HEAD": 4, "OPTIONS": 5, "PATCH": 6}


class MLModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, feature_dict: dict) -> int:
        """Returns 0 (normal) or 1 (malicious)."""
        row = feature_dict.copy()
        row["method"] = METHOD_CODES.get(row["method"], -1)
        df = pd.DataFrame([row])[FEATURE_COLUMNS]
        return int(self.model.predict(df)[0])

    def predict_proba(self, feature_dict: dict) -> float:
        """Returns probability of malicious class."""
        row = feature_dict.copy()
        row["method"] = METHOD_CODES.get(row["method"], -1)
        arr = np.array([[row[col] for col in FEATURE_COLUMNS]])
        return float(self.model.predict_proba(arr)[0][1])
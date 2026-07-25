import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from src.features.extractor import extract_features
from src.ml.model import MLModel

_ml_model = None

def get_ml_model():
    global _ml_model
    if _ml_model is None:
        _ml_model = MLModel()
    return _ml_model


def run_pure_ml(request: dict) -> dict:
    """
    Baseline: no FSM gate. Every request — even structurally malformed ones —
    goes straight to the ML model. Used to measure the cost of skipping the
    cheap structural pre-filter.
    """
    try:
        features = extract_features(request)
        model = get_ml_model()
        score = model.predict_proba(features)
        prediction = 1 if score >= 0.5 else 0

        return {
            "verdict": "MALICIOUS" if prediction == 1 else "ALLOWED",
            "ml_score": score,
            "error": None,
        }
    except Exception as e:
        # Malformed requests may break feature extraction itself —
        # record this as a failure mode, not silently skip it.
        return {
            "verdict": "ERROR",
            "ml_score": None,
            "error": str(e),
        }
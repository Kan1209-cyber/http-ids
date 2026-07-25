import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from src.fsm.request_fsm import RequestFSM, State
from src.features.extractor import extract_features
from src.ml.model import MLModel

_ml_model = None

def get_ml_model():
    global _ml_model
    if _ml_model is None:
        _ml_model = MLModel()
    return _ml_model


def run_pipeline(request: dict) -> dict:
    """
    request: parsed request dict (from request_parser.py)
    Returns: {
        "verdict": "REJECTED_STRUCTURE" | "MALICIOUS" | "ALLOWED",
        "fsm_state": str,
        "fsm_reject_reason": str or None,
        "ml_score": float or None,
    }
    """
    fsm = RequestFSM()
    state = fsm.process(request)

    if state != State.COMPLETE:
        return {
            "verdict": "REJECTED_STRUCTURE",
            "fsm_state": state.name,
            "fsm_reject_reason": fsm.reject_reason,
            "ml_score": None,
        }

    features = extract_features(request)
    model = get_ml_model()
    score = model.predict_proba(features)
    prediction = 1 if score >= 0.5 else 0

    return {
        "verdict": "MALICIOUS" if prediction == 1 else "ALLOWED",
        "fsm_state": state.name,
        "fsm_reject_reason": None,
        "ml_score": score,
    }
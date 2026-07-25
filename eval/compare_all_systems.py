import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from src.features.request_parser import parse_requests_from_file
from src.pipeline import run_pipeline
from src.baselines.pure_ml import run_pure_ml
from src.baselines.signature_based import run_signature_based

normal = parse_requests_from_file("data/raw/normalTrafficTest.txt")
anomalous = parse_requests_from_file("data/raw/anomalousTrafficTest.txt")

labeled = [(req, 0) for req in normal] + [(req, 1) for req in anomalous]

def evaluate(name, predict_fn, get_pred):
    tp = tn = fp = fn = 0
    for req, true_label in labeled:
        result = predict_fn(req)
        pred_label = get_pred(result)
        if pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 0 and true_label == 0:
            tn += 1
        elif pred_label == 1 and true_label == 0:
            fp += 1
        elif pred_label == 0 and true_label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    print(f"\n=== {name} ===")
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  FPR: {fpr:.3f}")


def fsm_ml_pred(result):
    return 1 if result["verdict"] == "MALICIOUS" else 0

def pure_ml_pred(result):
    return 1 if result["verdict"] == "MALICIOUS" else 0

def sig_pred(result):
    return 1 if result["verdict"] == "MALICIOUS" else 0


print(f"Evaluating on {len(labeled)} requests ({len(normal)} normal, {len(anomalous)} anomalous)")

evaluate("FSM + ML (proposed)", run_pipeline, fsm_ml_pred)
evaluate("Pure ML (baseline)", run_pure_ml, pure_ml_pred)
evaluate("Signature-based (baseline)", run_signature_based, sig_pred)
import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

from src.features.request_parser import parse_requests_from_file
from src.pipeline import run_pipeline
from src.baselines.pure_ml import run_pure_ml

normal = parse_requests_from_file("data/raw/normalTrafficTest.txt")[:4000]

def make_malformed():
    # Structurally broken: bad method, conflicting headers -> FSM rejects instantly
    return {
        "method": "INVALID",
        "path": "/x",
        "http_version": "HTTP/1.1",
        "headers": {"Content-Length": "10", "Transfer-Encoding": "chunked"},
        "body": None,
    }

def build_sample(malformed_pct, total=4000):
    n_malformed = int(total * malformed_pct)
    n_normal = total - n_malformed
    sample = normal[:n_normal] + [make_malformed() for _ in range(n_malformed)]
    random.seed(42)
    random.shuffle(sample)
    return sample

def time_system(fn, sample):
    start = time.perf_counter()
    for req in sample:
        fn(req)
    return time.perf_counter() - start

# Warm up
for req in normal[:200]:
    run_pipeline(req)
    run_pure_ml(req)

print(f"{'malformed%':>12} {'FSM+ML ms/req':>15} {'PureML ms/req':>15} {'FSM+ML faster by':>18}")
for pct in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]:
    sample = build_sample(pct)
    t_fsm = time_system(run_pipeline, sample) / len(sample) * 1000
    t_pure = time_system(run_pure_ml, sample) / len(sample) * 1000
    diff = ((t_pure - t_fsm) / t_pure) * 100
    print(f"{pct*100:>11.0f}% {t_fsm:>15.4f} {t_pure:>15.4f} {diff:>17.1f}%")
import sys, os, time, warnings, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import wilcoxon

from src.features.request_parser import parse_requests_from_file
from src.pipeline import run_pipeline
from src.baselines.pure_ml import run_pure_ml
from src.baselines.signature_based import run_signature_based

normal = parse_requests_from_file("data/raw/normalTrafficTest.txt")
anomalous = parse_requests_from_file("data/raw/anomalousTrafficTest.txt")

SAMPLE_SIZE = 20000
sample = normal[:SAMPLE_SIZE // 2] + anomalous[:SAMPLE_SIZE // 2]

def time_system(fn, sample):
    start = time.perf_counter()
    for req in sample:
        fn(req)
    return time.perf_counter() - start

systems = {
    "FSM + ML (proposed)": run_pipeline,
    "Pure ML": run_pure_ml,
    "Signature-based": run_signature_based,
}

for req in sample[:200]:
    run_pipeline(req)
    run_pure_ml(req)
    run_signature_based(req)

N_ROUNDS = 10  # more rounds gives the statistical test more power
results = {name: [] for name in systems}

print(f"Benchmarking on {len(sample)} requests, {N_ROUNDS} interleaved rounds\n")

for round_num in range(N_ROUNDS):
    order = list(systems.items())
    random.shuffle(order)
    for name, fn in order:
        t = time_system(fn, sample)
        results[name].append(t)
    print(f"Round {round_num + 1}/{N_ROUNDS} done")

print(f"\n{'System':<25} {'Mean ms/req':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
for name, times in results.items():
    times_ms = [(t / len(sample)) * 1000 for t in times]
    mean = np.mean(times_ms)
    std = np.std(times_ms, ddof=1)
    print(f"{name:<25} {mean:>12.4f} {std:>10.4f} {min(times_ms):>10.4f} {max(times_ms):>10.4f}")

# --- Paired Wilcoxon signed-rank test: FSM+ML vs Pure-ML ---
fsm_ml_times = results["FSM + ML (proposed)"]
pure_ml_times = results["Pure ML"]

stat, p_value = wilcoxon(fsm_ml_times, pure_ml_times)

print(f"\n=== Wilcoxon signed-rank test: FSM+ML vs Pure-ML ===")
print(f"Statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: statistically significant difference (p < 0.05)")
else:
    print("Result: NOT statistically significant (p >= 0.05) — cannot reject that they perform the same")
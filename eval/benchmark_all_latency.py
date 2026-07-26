import sys, os, time, warnings, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

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

# Warm-up pass — run all three once before timing anything
for req in sample[:200]:
    run_pipeline(req)
    run_pure_ml(req)
    run_signature_based(req)

N_ROUNDS = 5
results = {name: [] for name in systems}

print(f"Benchmarking on {len(sample)} requests, {N_ROUNDS} interleaved rounds\n")

for round_num in range(N_ROUNDS):
    # Randomize order each round so no system is always first or always last
    order = list(systems.items())
    random.shuffle(order)
    for name, fn in order:
        t = time_system(fn, sample)
        results[name].append(t)

print(f"{'System':<25} {'Avg ms/request':>16} {'Min':>10} {'Max':>10}")
for name, times in results.items():
    avg_ms = (sum(times) / len(times) / len(sample)) * 1000
    min_ms = (min(times) / len(sample)) * 1000
    max_ms = (max(times) / len(sample)) * 1000
    print(f"{name:<25} {avg_ms:>16.4f} {min_ms:>10.4f} {max_ms:>10.4f}")
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.request_parser import parse_requests_from_file
from src.fsm.request_fsm import RequestFSM, State

files = {
    "normal_train": "data/raw/normalTrafficTraining.txt",
    "normal_test": "data/raw/normalTrafficTest.txt",
    "anomalous_test": "data/raw/anomalousTrafficTest.txt",
}

for label, path in files.items():
    requests = parse_requests_from_file(path)
    completed = 0
    rejected = 0
    reject_reasons = {}

    for req in requests:
        fsm = RequestFSM()
        result = fsm.process(req)
        if result == State.COMPLETE:
            completed += 1
        else:
            rejected += 1
            reason = fsm.reject_reason
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    print(f"\n=== {label} ({path}) ===")
    print(f"Total requests: {len(requests)}")
    print(f"COMPLETE: {completed}")
    print(f"REJECTED: {rejected}")
    if reject_reasons:
        print("Reject reasons:")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
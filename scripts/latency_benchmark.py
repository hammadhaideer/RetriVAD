import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.retrivad import RetriVAD
from benchmark import image_files

UNIVAD_LATENCY_S = 900.0


def measure_latency(model, test_paths, n_warmup=2):
    for p in test_paths[:n_warmup]:
        model.predict(p)
    times = []
    for p in test_paths:
        t0 = time.time()
        model.predict(p)
        times.append(time.time() - t0)
    return np.array(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--category",  default="pushpins")
    p.add_argument("--max_ref",   type=int, default=69)
    p.add_argument("--n_test",    type=int, default=50)
    p.add_argument("--device",    default="cpu")
    args = p.parse_args()

    root  = Path(args.data_root) / args.category
    train = image_files(root / "train" / "good")
    test  = image_files(root / "test"  / "good")[:args.n_test]

    if not train or not test:
        print(f"No images found in {root}")
        return

    model = RetriVAD(k=1, device=args.device)
    model.build_memory_bank(train, max_ref=args.max_ref)

    print(f"\nMeasuring latency on {len(test)} images ...")
    times   = measure_latency(model, test)
    mean_t  = np.mean(times)
    std_t   = np.std(times)
    speedup = UNIVAD_LATENCY_S / mean_t

    print("\n" + "=" * 50)
    print(f"  Mean latency : {mean_t:.3f}s ± {std_t:.3f}s per image")
    print(f"  UniVAD       : {UNIVAD_LATENCY_S:.0f}s per image")
    print(f"  Speedup      : {speedup:.0f}×")
    print(f"  Device       : {args.device}")
    print("=" * 50)

    result = {
        "category":         args.category,
        "mean_latency_s":   round(float(mean_t), 4),
        "std_latency_s":    round(float(std_t), 4),
        "univad_latency_s": UNIVAD_LATENCY_S,
        "speedup":          round(speedup, 0),
        "device":           args.device,
        "n_test_images":    len(test),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/latency_benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n  Saved: results/latency_benchmark.json")


if __name__ == "__main__":
    main()

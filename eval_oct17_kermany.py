import argparse
import json
import sys
import time
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from models.retrivad import RetriVAD

UNIVAD          = 82.1
ANOMALY_CLASSES = ["CNV", "DME", "DRUSEN"]


def image_files(folder):
    exts = {".png", ".jpg", ".jpeg"}
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--max_ref", type=int, default=69)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.data_root)

    train_normal = image_files(root / "train" / "NORMAL")
    if not train_normal:
        print(f"[ERROR] No training normals found at {root / 'train' / 'NORMAL'}")
        sys.exit(1)

    test_normal  = image_files(root / "test" / "NORMAL")
    test_anomaly = []
    for cls in ANOMALY_CLASSES:
        test_anomaly.extend(image_files(root / "test" / cls))

    if not test_normal:
        print(f"[ERROR] No test normals at {root / 'test' / 'NORMAL'}")
        sys.exit(1)
    if not test_anomaly:
        print(f"[ERROR] No anomaly images found.")
        sys.exit(1)

    print(f"Train normals : {len(train_normal)}")
    print(f"Test normals  : {len(test_normal)}")
    print(f"Test anomalies: {len(test_anomaly)}")

    model = RetriVAD(k=1, device=args.device)
    model.build_memory_bank(train_normal, max_ref=args.max_ref)

    scores, labels = [], []
    t0 = time.time()
    for p in test_normal:
        scores.append(model.predict(p)); labels.append(0)
    for p in test_anomaly:
        scores.append(model.predict(p)); labels.append(1)

    elapsed  = time.time() - t0
    latency  = elapsed / len(scores)
    auc      = roc_auc_score(labels, scores) * 100

    print(f"\nN={labels.count(0)}  A={labels.count(1)}")
    print(f"OCT17 img-AUC = {auc:.2f}%")
    print(f"UniVAD        = {UNIVAD}%")
    print(f"Gap           = {auc - UNIVAD:+.2f}%")
    print(f"Latency       = {latency:.3f}s/image")

    result = {
        "dataset":       "OCT17_Kermany",
        "image_auroc":   round(auc, 2),
        "univad_auroc":  UNIVAD,
        "gap":           round(auc - UNIVAD, 2),
        "latency_s":     round(latency, 4),
        "n_normal_test": len(test_normal),
        "n_anomaly":     len(test_anomaly),
    }
    out = Path("results/oct17_kermany_retrivad.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

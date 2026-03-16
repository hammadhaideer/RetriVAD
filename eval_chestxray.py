import argparse
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from models.retrivad import RetriVAD



def image_files(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=r"C:\Users\HP\Downloads\temp_mvtec\chest_xray",
    )
    parser.add_argument("--max_ref", type=int, default=69)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.data_root)

    train_normal = image_files(root / "train" / "NORMAL")
    test_normal  = image_files(root / "test"  / "NORMAL")
    test_anomaly = image_files(root / "test"  / "PNEUMONIA")

    print(f"Train normals : {len(train_normal)}")
    print(f"Test normals  : {len(test_normal)}")
    print(f"Test anomalies: {len(test_anomaly)}")

    model = RetriVAD(k=1, device=args.device)
    model.build_memory_bank(train_normal, max_ref=args.max_ref)

    scores, labels = [], []
    for p in test_normal:
        scores.append(model.predict(p)); labels.append(0)
    for p in test_anomaly:
        scores.append(model.predict(p)); labels.append(1)

    auc = roc_auc_score(labels, scores) * 100
    print(f"\nN={labels.count(0)}  A={labels.count(1)}")
    print(f"ChestXray img-AUC = {auc:.2f}%")



if __name__ == "__main__":
    main()

import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIM   = 768

def preprocess(img, size=224):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def load_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
    return model

def extract(model, img_path, mode):
    img = Image.open(img_path)
    x = preprocess(img)
    with torch.no_grad():
        out = model.forward_features(x)
        cls_token   = out["x_norm_clstoken"][0]        # (768,)
        patch_tokens = out["x_norm_patchtokens"][0]    # (256, 768)
        pooled = patch_tokens.mean(0)                  # (768,)

    if mode == "cls":
        feat = cls_token
    elif mode == "patch":
        feat = pooled
    else:  # combined (default)
        feat = (cls_token + pooled) / 2.0

    v = feat.cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def imgs(folder):
    exts = {'.png', '.jpg', '.jpeg'}
    f = Path(folder)
    return sorted(p for p in f.iterdir() if p.suffix.lower() in exts) if f.exists() else []

def run(model, train_normals, test_normals, test_anomalies, mode):
    # Build memory bank
    feats = []
    for p in tqdm(train_normals[:69], desc=f"  Bank [{mode}]", leave=False):
        try: feats.append(extract(model, p, mode))
        except: pass
    feats = np.stack(feats).astype(np.float32)
    faiss.normalize_L2(feats)
    index = faiss.IndexFlatL2(DIM)
    index.add(feats)

    scores, labels = [], []
    for p in test_normals:
        try:
            v = extract(model, p, mode).reshape(1, -1)
            faiss.normalize_L2(v)
            d, _ = index.search(v, 1)
            scores.append(float(d[0][0])); labels.append(0)
        except: pass
    for p in test_anomalies:
        try:
            v = extract(model, p, mode).reshape(1, -1)
            faiss.normalize_L2(v)
            d, _ = index.search(v, 1)
            scores.append(float(d[0][0])); labels.append(1)
        except: pass

    auc = roc_auc_score(labels, scores) * 100
    return round(auc, 1)

root = Path(r'C:\Users\HP\Downloads\mvtec_loco_anomaly_detection\pushpins')
train_normals  = imgs(root / 'train' / 'good')
test_normals   = imgs(root / 'test' / 'good')
test_anomalies = []
for sub in (root / 'test').iterdir():
    if sub.is_dir() and sub.name != 'good':
        test_anomalies.extend(imgs(sub))

print("Pushpins: N_train={} N_test={} A_test={}".format(
    len(train_normals), len(test_normals), len(test_anomalies)))

print("Loading DINOv2...")
model = load_model()

print("\nRunning ablation...")
for mode in ["cls", "patch", "combined"]:
    auc = run(model, train_normals, test_normals, test_anomalies, mode)
    label = {
        "cls":      "CLS token only      ",
        "patch":    "Mean patch only     ",
        "combined": "CLS + mean (default)"
    }[mode]
    print(f"  {label}  AUROC = {auc:.1f}%")
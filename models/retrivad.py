from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIM   = 768


def _preprocess(img, size=224):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class DINOv2Encoder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            .eval().to(device)
        )

    @torch.no_grad()
    def encode(self, img):
        x    = _preprocess(img).to(self.device)
        out  = self.model(x)
        feat = out[0] if out.ndim == 2 else (out[0, 0] + out[0, 1:].mean(0)) / 2.0
        v    = feat.cpu().numpy().astype(np.float32)
        v   /= (np.linalg.norm(v) + 1e-8)
        return v

    @torch.no_grad()
    def encode_paths(self, paths, desc="Encoding", max_n=None):
        paths = list(paths)
        if max_n:
            paths = paths[:max_n]
        feats = []
        for p in tqdm(paths, desc=desc, leave=False):
            try:
                feats.append(self.encode(Image.open(p)))
            except Exception as e:
                print(f"  [warn] {p}: {e}")
        return (
            np.stack(feats).astype(np.float32)
            if feats else np.empty((0, DIM), np.float32)
        )


class RetriVAD:
    def __init__(self, k=1, device="cpu"):
        self.k       = k
        self.device  = device
        self.encoder = DINOv2Encoder(device)
        self.index   = None

    def build_memory_bank(self, normal_paths, max_ref=69):
        feats = self.encoder.encode_paths(
            normal_paths, desc="Memory bank", max_n=max_ref
        )
        if len(feats) == 0:
            raise ValueError("No features extracted.")
        faiss.normalize_L2(feats)
        self.index = faiss.IndexFlatL2(DIM)
        self.index.add(feats)
        print(f"  Memory bank: {self.index.ntotal} refs")

    def save_index(self, path):
        faiss.write_index(self.index, str(path))

    def load_index(self, path):
        self.index = faiss.read_index(str(path))

    def predict(self, img_or_path):
        if isinstance(img_or_path, (str, Path)):
            img_or_path = Image.open(img_or_path)
        v = self.encoder.encode(img_or_path).reshape(1, -1)
        faiss.normalize_L2(v)
        d, _ = self.index.search(v, self.k)
        return float(np.mean(d[0]))

    def anomaly_map(self, img_or_path, patch_grid=14):
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert("RGB")
        else:
            img = img_or_path.convert("RGB")
        img  = img.resize((224, 224), Image.BILINEAR)
        ps   = 224 // patch_grid
        amap = np.zeros((patch_grid, patch_grid), dtype=np.float32)
        for i in range(patch_grid):
            for j in range(patch_grid):
                crop = img.crop((j*ps, i*ps, (j+1)*ps, (i+1)*ps))
                v    = self.encoder.encode(crop).reshape(1, -1)
                faiss.normalize_L2(v)
                d, _ = self.index.search(v, self.k)
                amap[i, j] = float(np.mean(d[0]))
        return amap

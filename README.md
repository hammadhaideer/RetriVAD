# RetriVAD: A Retrieval-Based Approach to Unified Visual Anomaly Detection

RetriVAD is a training-free visual anomaly detection framework that replaces complex multi-model pipelines with a single frozen **DINOv2** encoder and a **FAISS** nearest-neighbour index.

**Key idea:** A retrieval baseline built on DINOv2 global features matches or exceeds the three-model UniVAD (CVPR 2025) pipeline at **921× lower inference latency** on multiple benchmark datasets.

---

## How It Works

RetriVAD operates in three training-free steps:

```
Input Image
     │
     ▼
┌─────────────────────────────┐
│  DINOv2 ViT-B/14 (frozen)   │
│  CLS token + mean(patches)  │
│  → 768-d L2-normalised vec  │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  FAISS IndexFlatL2          │
│  Memory bank: ≤69 normals   │
│  k-NN distance → score      │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  14×14 patch map            │
│  Bilinear upsample          │
│  → Pixel-level heatmap      │
└─────────────────────────────┘
```

Higher distance = more anomalous. No training. No fine-tuning. No GPU required.

---

## Installation

```bash
git clone https://github.com/hammadhaideer/RetriVAD.git
cd RetriVAD

conda create -n retrivad python=3.10 -y
conda activate retrivad

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu numpy Pillow scikit-learn scipy tqdm
```

---

## Usage

### MVTec LOCO

```bash
python benchmark.py --dataset loco \
    --data_root /path/to/mvtec_loco_anomaly_detection
```

### MVTec-AD

```bash
python benchmark.py --dataset mvtec \
    --data_root /path/to/mvtec
```

### VisA

```bash
python benchmark.py --dataset visa \
    --data_root /path/to/VisA_20220922
```

### Medical Datasets

```bash
python eval_brainmri.py
python eval_chestxray.py
python eval_resc.py
python eval_oct17_kermany.py --data_root /path/to/OCT2017
```

---

## Use in Your Own Code

```python
from models.retrivad import RetriVAD
from pathlib import Path

model = RetriVAD(k=1, device='cpu')

normal_paths = list(Path('data/mvtec/bottle/train/good').glob('*.png'))
model.build_memory_bank(normal_paths, max_ref=69)

# Higher score = more anomalous
score = model.predict('data/mvtec/bottle/test/broken_large/000.png')
print(f'Anomaly score: {score:.4f}')

# 14x14 patch-level heatmap
heatmap = model.anomaly_map('data/mvtec/bottle/test/broken_large/000.png')
```

---

## Repository Structure

```
RetriVAD/
├── models/
│   └── retrivad.py              # Core model — DINOv2 + FAISS
├── utils/
│   └── metrics.py               # AUROC metrics helpers
├── scripts/
│   ├── eval_pixel_auroc.py      # Pixel-level AUROC evaluation
│   └── latency_benchmark.py     # Latency timing
├── benchmark.py                 # Unified eval: MVTec-AD, VisA, LOCO
├── eval_brainmri.py
├── eval_chestxray.py
├── eval_resc.py
├── eval_oct17_kermany.py
└── results/
```

---

## Dataset Preparation

| Dataset | Source |
|---------|--------|
| MVTec-AD | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| MVTec LOCO | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-loco) |
| VisA | [GitHub](https://github.com/amazon-science/spot-diff) |
| BrainMRI | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| ChestXray | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| RESC | [GitHub](https://github.com/SuhaoLiu/RESC) |
| OCT17 | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |

---

## Latency Comparison

| Method | Latency / image | Speedup vs UniVAD |
|--------|----------------|-------------------|
| UniVAD | ~900s | 1× |
| WinCLIP | ~60s | 15× |
| PatchCore | ~5s | 180× |
| **RetriVAD** | **~1s** | **921×** |

Measurements on CPU (Intel Core i7).

---

## Acknowledgements

- [UniVAD](https://github.com/FantasticGNU/UniVAD) — primary baseline
- [DINOv2](https://github.com/facebookresearch/dinov2) — frozen backbone
- [FAISS](https://github.com/facebookresearch/faiss) — similarity search

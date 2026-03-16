# RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval

Official implementation of the paper **RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval** (Submitted to ACM Multimedia 2026).

## Introduction

RetriVAD is a training-free unified visual anomaly detection method capable of detecting anomalies across industrial, logical, and medical imaging domains using a single frozen encoder — without any domain-specific training, fine-tuning, or GPU.

Given a category, RetriVAD builds a memory bank offline by encoding normal reference images into feature vectors stored in a FAISS index. At test time, the anomaly score for a query image is its nearest-neighbour distance to the memory bank. No thresholds are tuned and no labels are used. The same index generalises across industrial products, brain MRI, chest X-rays, and retinal images.

## Results

### Image-Level AUROC (1-shot setting)

| Dataset | PatchCore | WinCLIP | AnomalyGPT | UniAD | MedCLIP | UniVAD | **RetriVAD** |
|---------|-----------|---------|------------|-------|---------|--------|------------|
| MVTec-AD | 84.0 | 93.1 | 94.1 | 70.3 | 75.2 | 97.8 | **89.6** |
| VisA | 74.8 | 83.8 | 87.4 | 61.3 | 69.0 | 93.5 | **79.1** |
| MVTec LOCO | 62.0 | 58.0 | 60.4 | 50.9 | 54.9 | 71.0 | **72.1** |
| BrainMRI | 73.2 | 55.4 | 73.1 | 50.0 | 69.7 | 80.2 | **96.16** |
| RESC | 56.3 | 72.9 | 82.4 | 53.5 | 66.9 | 85.5 | **76.53** |
| ChestXray | 66.4 | 70.2 | 68.5 | 60.6 | 71.4 | 72.2 | **86.79** |
| OCT17 Kermany | 59.9 | 79.7 | 77.5 | 44.4 | 64.6 | 82.1 | **74.67** |
| **Mean** | 68.1 | 73.3 | 77.6 | 55.9 | 70.3 | 81.9 | **82.1** |

> VisA results use the official CSV split protocol.

### Pixel-Level AUROC

| Dataset | Pixel-AUROC |
|---------|-------------|
| VisA (12 categories) | 91.92 |
| MVTec-AD (15 categories) | 71.27 |
| MVTec LOCO (5 categories) | 56.21 |

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/hammadhaideer/RetriVAD.git
cd RetriVAD
```

Create and activate the conda environment:

```bash
conda create -n retrivad python=3.10
conda activate retrivad
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The core dependencies are:

```
torch==2.10.0+cpu
faiss-cpu==1.13.2
scikit-learn==1.7.2
numpy==2.2.6
scipy
Pillow
tqdm
```

## Data Preparation

Download each dataset and place it at the path shown. Datasets are not included in this repository.

#### MVTec-AD

Download from [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract to:

```
Downloads/mvtec/
├── bottle/
├── cable/
├── ...
```

#### VisA

Download from [https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) and extract to:

```
Downloads/VisA_20220922/
├── candle/
├── capsules/
├── ...
```

#### MVTec LOCO

Download from [https://www.mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco) and extract to:

```
Downloads/mvtec_loco_anomaly_detection/
├── breakfast_box/
├── pushpins/
├── ...
```

#### Medical Datasets

| Dataset | Source | Path |
|---------|--------|------|
| BrainMRI | [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) | `Downloads/medical_anomaly/brain_mri` |
| RESC | [GitHub](https://github.com/CharlesKing/RESC) | `Downloads/resc/RESC` |
| ChestXray | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | `Downloads/temp_mvtec/chest_xray` |
| OCT17 Kermany | [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3) | `Downloads/archive/OCT2017_` |

## Running Evaluation

### Image-Level Evaluation

```bash
python benchmark.py
```

Runtime: approximately 30–60 minutes on CPU. All scores should reproduce the table above within ±0.5%.

### Pixel-Level Evaluation

```bash
# MVTec-AD and MVTec LOCO
python scripts/eval_pixel_auroc.py

# VisA
python eval_pixel_auroc_visa.py

# BrainMRI
python eval_pixel_auroc_brainmri.py

# RESC
python eval_pixel_auroc_resc.py
```

**Note:** Pixel-level evaluation is computationally intensive on CPU-only hardware. All results are pre-computed and saved in `results/`. Re-running is only necessary if the model code changes.

### Latency Benchmark

```bash
python scripts/latency_benchmark.py --data_root /path/to/mvtec --category bottle --n_test 20
```

### Qualitative Figure

```bash
python generate_heatmaps.py
```

Output is saved to `figures/fig_qualitative.png`.

## Pre-computed Results

All confirmed results are available in `results/`:

| File | Contents |
|------|----------|
| `confirmed_results_10mar2026.json` | Image-level AUROC across all 7 datasets |
| `pixel_auroc_final.json` | Pixel-level AUROC for MVTec-AD and MVTec LOCO |
| `visa_pixel_auroc.json` | Pixel-level AUROC for VisA (12 categories) |
| `retrivad_latency_confirmed.json` | Per-image latency measured on CPU |

## Citation

If you find RetriVAD useful in your research, please cite:

```bibtex
@inproceedings{haider2026retrivad,
  title={RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval},
  author={Haider, Hammad Ali and Zheng, Panpan},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```

---

**Hammad Ali Haider, Panpan Zheng**  
School of Computer Science and Technology, Xinjiang University, Ürümqi, China  
📧 hammadhaideerr@gmail.com

> Submitted to **ACM Multimedia 2026** — Melbourne, Australia, October 2026

---

## Overview

RetriVAD is a **training-free** unified visual anomaly detection method that uses a single frozen DINOv2 encoder combined with nearest-neighbour retrieval over a small memory bank of normal reference images. It requires **no training, no fine-tuning, no domain adaptation, and no GPU**, yet matches or exceeds the accuracy of multi-model pipelines across seven benchmark datasets spanning industrial, medical, and retinal imaging domains.

**Key claim:** RetriVAD processes one image in **0.76 seconds** on a standard dual-core CPU — a fraction of the cost of pipeline-based unified methods.

---

## Key Results

### Image-Level AUROC (7 Datasets)

| Dataset | RetriVAD | UniVAD (CVPR 2025) | Gap |
|---|---|---|---|
| MVTec-AD | **89.6%** | 97.8% | −8.2% |
| VisA | **79.1%** | 93.5% | −14.4% |
| MVTec LOCO | **72.1%** | 71.0% | **+1.1%** |
| BrainMRI | **96.16%** | 80.2% | **+15.96%** |
| RESC | **76.53%** | 85.5% | −8.97% |
| ChestXray | **86.79%** | 72.2% | **+14.59%** |
| OCT17 Kermany | **74.67%** | 82.1% | −7.43% |
| **Mean (7 datasets)** | **82.1%** | **81.9%** | **+0.2%** |

> VisA result uses the official CSV split protocol.

### Pixel-Level AUROC

| Dataset | Mean Pixel-AUROC |
|---|---|
| VisA (12 categories) | **91.92%** |
| MVTec-AD (15 categories) | **71.27%** |
| MVTec LOCO (5 categories) | **56.21%** |

### Latency

| Method | Hardware | Per-Image Latency |
|---|---|---|
| RetriVAD (image-level) | Intel Core i7, CPU-only | **0.76s** |
| RetriVAD (pixel-level) | Intel Core i7, CPU-only | ~180s (256 crops) |

---

## Method

RetriVAD uses **DINOv2 ViT-B/14** without modification. For each category:

1. **Offline:** Encode up to 69 normal reference images into 768-dimensional feature vectors and store in a FAISS index.
2. **Online:** For a query image, run one DINOv2 forward pass and compute the nearest-neighbour distance to the memory bank. This distance is the anomaly score.

No thresholds are tuned. No labels are used. The same index works for industrial products, MRI scans, chest X-rays, and retinal images.

> **Pixel-level localisation** requires 256 sequential DINOv2 forward passes — one per 14×14-pixel crop from the 16×16 spatial grid — producing a 16×16 distance map which is bilinearly upsampled to the original image resolution.

---

## Installation

```bash
# Create environment
conda create -n retrivad python=3.10
conda activate retrivad

# Install dependencies
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu==1.13.2
pip install scikit-learn==1.7.2 scipy numpy==2.2.6 Pillow tqdm
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch, faiss, numpy; print('torch:', torch.__version__); print('faiss: OK'); print('numpy:', numpy.__version__)"
```

---

## Dataset Setup

Download each dataset and place it at the path shown below. **Datasets are not included in this repository.**

| Dataset | Download | Expected Path |
|---|---|---|
| MVTec-AD | [mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad) | `C:/Users/HP/Downloads/mvtec` |
| VisA | [kaggle.com/datasets/amarbagul/visapcb](https://www.kaggle.com/datasets/amarbagul/visapcb) | `C:/Users/HP/Downloads/VisA_20220922` |
| MVTec LOCO | [mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco) | `C:/Users/HP/Downloads/mvtec_loco_anomaly_detection` |
| BrainMRI | [kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) | `C:/Users/HP/Downloads/medical_anomaly/brain_mri` |
| RESC | [github.com/CharlesKing/RESC](https://github.com/CharlesKing/RESC) | `C:/Users/HP/Downloads/resc/RESC` |
| ChestXray | [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | `C:/Users/HP/Downloads/temp_mvtec/chest_xray` |
| OCT17 Kermany | [data.mendeley.com/datasets/rscbjbr9sj/3](https://data.mendeley.com/datasets/rscbjbr9sj/3) | `C:/Users/HP/Downloads/archive/OCT2017_` |

---

## Running Evaluation

### Image-Level Evaluation (All 7 Datasets)
```bash
conda activate retrivad
cd RetriVAD
python benchmark.py
```
Expected runtime: 30–60 minutes on CPU. All scores should match the table above within ±0.5%.

### Pixel-Level Evaluation
```bash
# MVTec-AD + MVTec LOCO
python scripts/eval_pixel_auroc.py

# VisA
python eval_pixel_auroc_visa.py

# BrainMRI
python eval_pixel_auroc_brainmri.py

# RESC
python eval_pixel_auroc_resc.py
```
> ⚠️ **Warning:** Pixel-level evaluation is extremely slow on CPU. MVTec-AD took approximately 3 days; MVTec LOCO took approximately 4 days. All results are already saved in `results/`. Re-run only if the model code changes.

### Latency Benchmark
```bash
python scripts/latency_benchmark.py --data_root C:/Users/HP/Downloads/mvtec --category bottle --n_test 20
```

### Qualitative Heatmap Figure
```bash
python generate_heatmaps.py
# Output saved to figures/fig_qualitative.png
```

---

## Pre-computed Results

All confirmed results are saved in `results/`:

| File | Contents |
|---|---|
| `confirmed_results_10mar2026.json` | Image-level AUROC — all 7 datasets |
| `pixel_auroc_final.json` | Pixel-level AUROC — MVTec-AD and LOCO |
| `visa_pixel_auroc.json` | Pixel-level AUROC — VisA (12 categories) |
| `retrivad_latency_confirmed.json` | Measured latency — 5 categories, CPU |

---

## Repository Structure

```
RetriVAD/
├── benchmark.py                  # Main evaluation script (all 7 datasets)
├── generate_heatmaps.py          # Qualitative figure generation
├── ablation_features.py          # Ablation study
├── eval_brainmri.py              # BrainMRI standalone eval
├── eval_chestxray.py             # ChestXray standalone eval
├── eval_resc.py                  # RESC standalone eval
├── eval_oct17_kermany.py         # OCT17 standalone eval
├── eval_pixel_auroc_visa.py      # VisA pixel-level eval
├── eval_pixel_auroc_brainmri.py  # BrainMRI pixel-level eval
├── eval_pixel_auroc_resc.py      # RESC pixel-level eval
├── models/
│   └── retrivad.py               # Core model — DINOv2 + FAISS retrieval
├── utils/
│   └── metrics.py                # AUROC and evaluation utilities
├── scripts/
│   ├── eval_pixel_auroc.py       # MVTec-AD + LOCO pixel eval
│   └── latency_benchmark.py      # Per-image latency measurement
├── results/                      # Pre-computed results (JSON)
└── figures/                      # Generated figures
```

---

## Citation

If you use RetriVAD in your research, please cite:

```bibtex
@inproceedings{haider2026retrivad,
  title     = {RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval},
  author    = {Haider, Hammad Ali and Zheng, Panpan},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  address   = {Melbourne, Australia}
}
```

---

## License

This project is released under the MIT License.

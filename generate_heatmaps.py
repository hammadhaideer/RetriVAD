"""
RetriVAD qualitative figure generator.

Produces a 4-row x 4-column grid:
    Columns: Normal reference | Test image | Ground truth | Anomaly heatmap
    Rows:    MVTec-AD carpet | MVTec-AD bottle | BrainMRI | VisA PCB1

Usage:
    python generate_heatmaps.py \
        --carpet_normal  PATH/mvtec/carpet/train/good/000.png \
        --carpet_test    PATH/mvtec/carpet/test/color/000.png \
        --carpet_gt      PATH/mvtec/carpet/ground_truth/color/000_mask.png \
        --bottle_normal  PATH/mvtec/bottle/train/good/000.png \
        --bottle_test    PATH/mvtec/bottle/test/broken_large/000.png \
        --bottle_gt      PATH/mvtec/bottle/ground_truth/broken_large/000_mask.png \
        --brain_normal   PATH/brain_mri/Training/no_tumor/image(1).jpg \
        --brain_test     PATH/brain_mri/Testing/glioma/Te-gl_0010.jpg \
        --visa_normal    PATH/VisA_20220922/pcb1/Data/Images/Normal/0000.JPG \
        --visa_test      PATH/VisA_20220922/pcb1/Data/Images/Anomaly/000.JPG \
        --visa_gt        PATH/VisA_20220922/pcb1/Data/Masks/Anomaly/000.png \
        --output         figures/fig_qualitative.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_dinov2():
    import torch
    print("Loading DINOv2 ViT-B/14 ...")
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14",
        pretrained=True, verbose=False,
    )
    model.eval()
    return model, torch


def encode(model, torch, path):
    import torch.nn.functional as F
    img = Image.open(path).convert("RGB")
    t   = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out   = model.forward_features(t)
        patch = out["x_norm_patchtokens"].squeeze(0)
        patch = F.normalize(patch, dim=1)
    return patch


def compute_heatmap(model, torch, normal_path, test_path, plo=5, phi=95):
    import torch.nn.functional as F
    nf       = encode(model, torch, normal_path)
    tf       = encode(model, torch, test_path)
    dists    = torch.cdist(tf, nf)
    min_dist = dists.min(dim=1).values.reshape(16, 16).cpu().numpy()
    lo, hi   = np.percentile(min_dist, plo), np.percentile(min_dist, phi)
    if hi - lo < 1e-8:
        return np.zeros((224, 224), np.float32)
    normed = np.clip((min_dist - lo) / (hi - lo), 0, 1)
    t  = torch.from_numpy(normed).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, (224, 224), mode="bilinear",
                       align_corners=False).squeeze().numpy()
    return up.astype(np.float32)


def load_image(path, gray=False, size=224):
    mode = "L" if gray else "RGB"
    img  = Image.open(path).convert(mode).resize((size, size), Image.LANCZOS)
    arr  = np.array(img)
    if gray:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def load_mask(path, size=224):
    if not path or not os.path.exists(path):
        return None
    m   = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = (np.array(m) > 0).astype(np.uint8)
    return arr if arr.sum() > 0 else None


def make_ground_truth(test_img, mask):
    if mask is not None:
        gt = test_img.copy().astype(float)
        m  = mask.astype(bool)
        gt[m, 0] = np.clip(gt[m, 0] * 0.15 + 235, 0, 255)
        gt[m, 1] = np.clip(gt[m, 1] * 0.15 + 10,  0, 255)
        gt[m, 2] = np.clip(gt[m, 2] * 0.15 + 10,  0, 255)
        return gt.astype(np.uint8), True
    faded = (test_img.astype(float) * 0.35 + 155).clip(0, 255).astype(np.uint8)
    return faded, False


def render(rows, out_path, dpi=300):
    col_titles = [
        "Normal reference", "Test image", "Ground truth", "Anomaly heatmap",
    ]
    n = len(rows)

    fig = plt.figure(
        figsize=(4 * 2.6 + 0.9, n * 2.6 + 0.4),
        dpi=dpi,
        facecolor="white",
    )
    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[4 * 2.6, 0.55],
        wspace=0.02,
    )
    gs = gridspec.GridSpecFromSubplotSpec(
        n, 4, subplot_spec=outer[0],
        hspace=0.04, wspace=0.04,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n)]

    for c, title in enumerate(col_titles):
        axes[0][c].set_title(
            title, fontsize=9, fontweight="bold",
            pad=5, fontfamily="sans-serif",
        )

    for r, row in enumerate(rows):
        axes[r][0].imshow(row["normal"])
        axes[r][1].imshow(row["test"])
        axes[r][2].imshow(row["gt"])
        if not row["has_mask"]:
            axes[r][2].text(
                112, 185, "Image-level\nlabel only",
                ha="center", va="center", fontsize=7.5,
                color="#333", style="italic",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          alpha=0.80, ec="#bbb", lw=0.5),
            )
        axes[r][3].imshow(
            row["heatmap"], cmap="jet",
            vmin=0, vmax=1, interpolation="bilinear",
        )
        for c in range(4):
            axes[r][c].axis("off")
        axes[r][0].set_ylabel(
            row["label"], fontsize=9, fontweight="bold",
            labelpad=8, rotation=90, va="center",
            fontfamily="sans-serif",
        )
        axes[r][0].yaxis.set_label_position("left")
        axes[r][0].yaxis.label.set_visible(True)

    cbar_ax = fig.add_subplot(outer[1])
    sm = plt.cm.ScalarMappable(
        cmap="jet", norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Anomaly score", fontsize=8, labelpad=6)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"], fontsize=7.5)

    plt.subplots_adjust(left=0.10, right=0.89, top=0.94, bottom=0.01)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}  ({dpi} DPI)")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--carpet_normal",  required=True)
    p.add_argument("--carpet_test",    required=True)
    p.add_argument("--carpet_gt",      default=None)
    p.add_argument("--bottle_normal",  required=True)
    p.add_argument("--bottle_test",    required=True)
    p.add_argument("--bottle_gt",      default=None)
    p.add_argument("--brain_normal",   required=True)
    p.add_argument("--brain_test",     required=True)
    p.add_argument("--visa_normal",    required=True)
    p.add_argument("--visa_test",      required=True)
    p.add_argument("--visa_gt",        default=None)
    p.add_argument("--output", default="figures/fig_qualitative.png")
    p.add_argument("--dpi",    type=int, default=300)
    return p.parse_args()


def main():
    args         = parse()
    model, torch = load_dinov2()

    configs = [
        dict(label="MVTec-AD\n(carpet)",
             normal=args.carpet_normal, test=args.carpet_test,
             gt=args.carpet_gt,         gray=False),
        dict(label="MVTec-AD\n(bottle)",
             normal=args.bottle_normal, test=args.bottle_test,
             gt=args.bottle_gt,         gray=False),
        dict(label="BrainMRI\n(Medical)",
             normal=args.brain_normal,  test=args.brain_test,
             gt=None,                   gray=True),
        dict(label="VisA\n(PCB1)",
             normal=args.visa_normal,   test=args.visa_test,
             gt=args.visa_gt,           gray=False),
    ]

    rows = []
    for cfg in configs:
        tag = cfg["label"].replace("\n", " ")
        print(f"Processing {tag} ...")
        normal  = load_image(cfg["normal"], cfg["gray"])
        test    = load_image(cfg["test"],   cfg["gray"])
        mask    = load_mask(cfg["gt"])
        heatmap = compute_heatmap(model, torch, cfg["normal"], cfg["test"])
        gt, has = make_ground_truth(test, mask)
        rows.append(dict(
            label=cfg["label"], normal=normal, test=test,
            gt=gt, has_mask=has, heatmap=heatmap,
        ))

    print("Rendering ...")
    render(rows, args.output, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()

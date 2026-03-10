import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.retrivad import RetriVAD
from utils.metrics import pixel_auroc, upsample_map, combine_masks, load_mask
from benchmark import image_files, LOCO_CATEGORIES, MVTEC_CATEGORIES


def eval_pixel_loco(data_root, category, max_ref=69, device="cpu"):
    root  = Path(data_root) / category
    train = image_files(root / "train" / "good")

    model = RetriVAD(k=1, device=device)
    model.build_memory_bank(train, max_ref=max_ref)

    pred_maps, gt_maps = [], []

    for atype in ["logical_anomalies", "structural_anomalies"]:
        adir   = root / "test"         / atype
        gt_dir = root / "ground_truth" / atype
        if not adir.exists():
            continue
        for img_path in tqdm(image_files(adir), desc=f"{category}/{atype}", leave=False):
            mask_dir = gt_dir / img_path.stem
            if not mask_dir.exists():
                continue
            orig     = Image.open(img_path)
            oh, ow   = orig.size[1], orig.size[0]
            amap     = model.anomaly_map(img_path)
            amap     = upsample_map(amap, oh, ow)
            mask     = combine_masks(str(mask_dir))
            if mask is not None and mask.shape == amap.shape:
                pred_maps.append(amap)
                gt_maps.append(mask)

    if not pred_maps:
        print(f"  [!] No valid mask pairs for {category}")
        return None

    pix_auc = pixel_auroc(gt_maps, pred_maps)
    print(f"  {category:30s}  pixel-AUC = {pix_auc*100:.2f}%  ({len(pred_maps)} images)")
    return pix_auc * 100


def eval_pixel_mvtec(data_root, category, max_ref=69, device="cpu"):
    root  = Path(data_root) / category
    train = image_files(root / "train" / "good")

    model = RetriVAD(k=1, device=device)
    model.build_memory_bank(train, max_ref=max_ref)

    pred_maps, gt_maps = [], []
    test_dir = root / "test"
    gt_dir   = root / "ground_truth"

    for defect in test_dir.iterdir():
        if not defect.is_dir() or defect.name == "good":
            continue
        for img_path in image_files(defect):
            mask_path = gt_dir / defect.name / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                continue
            orig   = Image.open(img_path)
            oh, ow = orig.size[1], orig.size[0]
            amap   = model.anomaly_map(img_path)
            amap   = upsample_map(amap, oh, ow)
            mask   = load_mask(mask_path)
            if mask.shape == amap.shape:
                pred_maps.append(amap)
                gt_maps.append(mask)

    if not pred_maps:
        return None

    pix_auc = pixel_auroc(gt_maps, pred_maps)
    print(f"  {category:30s}  pixel-AUC = {pix_auc*100:.2f}%")
    return pix_auc * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True, choices=["loco", "mvtec"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--category",  default="all")
    p.add_argument("--max_ref",   type=int, default=69)
    p.add_argument("--device",    default="cpu")
    args = p.parse_args()

    cats = (
        (LOCO_CATEGORIES if args.dataset == "loco" else MVTEC_CATEGORIES)
        if args.category == "all" else [args.category]
    )
    fn   = eval_pixel_loco if args.dataset == "loco" else eval_pixel_mvtec

    print(f"\n=== Pixel-AUROC: {args.dataset} ===")
    aucs = []
    for cat in cats:
        auc = fn(args.data_root, cat, args.max_ref, args.device)
        if auc is not None:
            aucs.append(auc)

    if aucs:
        print(f"\n  Mean pixel-AUC : {np.mean(aucs):.2f}%")
        print(f"  UniVAD         : 75.1% (LOCO) / 96.5% (MVTec)")


if __name__ == "__main__":
    main()

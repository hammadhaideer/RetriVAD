import os

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def pixel_auroc(gt_masks, pred_maps):
    all_gt   = np.concatenate([m.flatten() for m in gt_masks])
    all_pred = np.concatenate([m.flatten() for m in pred_maps])
    if all_gt.sum() == 0:
        return float("nan")
    return roc_auc_score(all_gt, all_pred)


def upsample_map(amap, target_h, target_w):
    zh = target_h / amap.shape[0]
    zw = target_w / amap.shape[1]
    return zoom(amap, (zh, zw), order=1)


def load_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 127).astype(np.uint8)


def combine_masks(mask_dir):
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if not files:
        return None
    masks = [np.array(Image.open(os.path.join(mask_dir, f)).convert("L")) for f in files]
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        combined = np.logical_or(combined, m > 127).astype(np.uint8)
    return combined

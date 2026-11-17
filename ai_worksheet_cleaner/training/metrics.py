"""Segmentation + image-quality metrics (implemented here so results are reproducible)."""
from __future__ import annotations

import cv2
import numpy as np

from training.classes import IGNORE_INDEX


def confusion(pred: np.ndarray, gt: np.ndarray, n: int) -> np.ndarray:
    m = gt != IGNORE_INDEX
    idx = gt[m].astype(np.int64) * n + pred[m].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)       # rows = truth, cols = prediction


def per_class(cm: np.ndarray) -> dict:
    out = {}
    for c in range(cm.shape[0]):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue                                              # class absent from truth AND prediction
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": float(p), "recall": float(r), "f1": float(2 * p * r / (p + r)) if p + r else 0.0,
                  "iou": float(tp / (tp + fp + fn)), "support_px": int(cm[c, :].sum())}
    return out


def binary_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM (Gaussian 11x11, sigma 1.5) over channels."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    vals = []
    for ch in range(a.shape[2] if a.ndim == 3 else 1):
        x = a[..., ch] if a.ndim == 3 else a
        y = b[..., ch] if b.ndim == 3 else b
        mx, my = g(x), g(y)
        sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(s.mean())
    return float(np.mean(vals))

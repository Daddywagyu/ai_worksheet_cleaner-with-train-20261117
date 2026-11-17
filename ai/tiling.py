"""Tiled inference shared by evaluation (PyTorch) and runtime (OpenVINO).
`infer` maps float32 RGB batch (B,3,T,T) in [0,1] -> logits (B,C,T,T)."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


def _window(t: int) -> np.ndarray:
    w = np.hanning(t).astype(np.float32)
    return np.outer(w, w) * 0.98 + 0.02


def predict_probs(bgr: np.ndarray, infer: Callable[[np.ndarray], np.ndarray], tile: int = 256,
                  stride: int = 192, batch: int = 4, n_classes: int = 7) -> np.ndarray:
    """Returns softmax probabilities (C,H,W) float32."""
    h, w = bgr.shape[:2]
    ph, pw = max(tile, h), max(tile, w)
    padded = cv2.copyMakeBorder(bgr, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101) if (ph, pw) != (h, w) else bgr
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ys = list(range(0, ph - tile + 1, stride)); xs = list(range(0, pw - tile + 1, stride))
    if ys[-1] != ph - tile:
        ys.append(ph - tile)
    if xs[-1] != pw - tile:
        xs.append(pw - tile)
    acc = np.zeros((n_classes, ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    win = _window(tile)
    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = np.stack([rgb[y:y + tile, xx:xx + tile].transpose(2, 0, 1) for y, xx in chunk]).astype(np.float32)
        logits = np.asarray(infer(x), np.float32)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        for (y, xx), pi in zip(chunk, p):
            acc[:, y:y + tile, xx:xx + tile] += pi * win
            wsum[y:y + tile, xx:xx + tile] += win
    return (acc / np.maximum(wsum, 1e-6))[:, :h, :w]

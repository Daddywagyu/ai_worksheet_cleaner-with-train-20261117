"""Pre-processing helpers.  Deliberately minimal: no global denoise / threshold / blur."""
from __future__ import annotations

import cv2
import numpy as np


def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def estimate_paper_color(bgr: np.ndarray) -> tuple[int, int, int]:
    """Median colour of the brightest 30% pixels (paper)."""
    step = max(1, int(np.sqrt(bgr.shape[0] * bgr.shape[1] / 300_000)))
    px = bgr[::step, ::step].reshape(-1, 3)
    lum = px.mean(axis=1)
    sel = px[lum >= np.percentile(lum, 70)]
    return tuple(int(v) for v in np.median(sel, axis=0))

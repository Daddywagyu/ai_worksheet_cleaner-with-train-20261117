"""Pencil / graphite detection (Phase 3).

* PencilDetector  - OpenVINO model (models/pencil_segmentation/pencil_unet.xml), if present.
* classical_pencil_mask - conservative non-AI fallback.  It NEVER marks pixels that are part of
  the protection mask or that hug printed content; such "pencil-like but unsafe" regions are
  returned separately so the QA stage can flag them for manual review.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ai.ink_detector import InkDetector
from core import color_detector, stroke_analyzer

PENCIL = 4


@dataclass
class PencilResult:
    mask: np.ndarray            # 0/255 pencil pixels judged safe to remove
    uncertain: np.ndarray       # pencil-like but unsafe / low confidence -> review, do not remove
    source: str                 # "ai" | "classical"
    device: list | None = None
    strict: np.ndarray | None = None   # AI pixels with p >= 0.9 (only used for the optional print-override)


class PencilDetector(InkDetector):
    KEY = "pencil"

    def predict_pencil(self, bgr: np.ndarray, dpi: float, conf: float = 0.7) -> PencilResult:
        probs = self._probs(bgr, dpi)
        h, w = bgr.shape[:2]
        p = cv2.resize(probs[PENCIL], (w, h), interpolation=cv2.INTER_LINEAR)
        L, _, _, chroma, _ = color_detector.lab_chroma(bgr)
        step = max(1, int(np.sqrt(L.size / 300_000)))
        paper_l = float(np.percentile(L[::step, ::step], 85))
        graphite = ((paper_l - L.astype(np.float32)) > 8) & (chroma < 20)     # snap to real, achromatic dark pixels
        sure, unsure = (p >= conf) & graphite, (p >= 0.3) & (p < conf) & graphite
        to = lambda m: m.astype(np.uint8) * 255
        return PencilResult(to(sure), to(unsure), "ai", list(self.engine.actual), to((p >= 0.9) & graphite))


def classical_pencil_mask(bgr: np.ndarray, dpi: float, analysis, conservative: bool = True) -> PencilResult:
    """analysis: core.document_analyzer.StructureAnalysis of the same image."""
    k = dpi / 200.0
    L8, _, _, chroma, _ = color_detector.lab_chroma(bgr)
    L = L8.astype(np.float32)
    r = max(3, int(round(10 * k)))
    bg = cv2.morphologyEx(L8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, 3 * k))
    diff = bg - L                                                      # how much darker than local paper
    hi_t, lo_t = (16.0, 8.0) if conservative else (12.0, 6.0)
    min_len, min_area = (45, 150) if conservative else (30, 80)     # short blobs are indistinguishable from gray print
    cand = ((diff >= hi_t) & (chroma < 14)).astype(np.uint8) * 255
    low = ((diff >= lo_t) & (chroma < 18)).astype(np.uint8) * 255

    printed = cv2.bitwise_or(cv2.bitwise_or(analysis.printed_dark, analysis.lines), cv2.bitwise_or(analysis.dotted, analysis.illustration))
    g = max(2, int(round(3 * k)))
    near_print = cv2.dilate(printed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    safe_c = cv2.bitwise_and(cand, cv2.bitwise_not(near_print))
    safe_c = cv2.morphologyEx(safe_c, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, lab, st, _ = cv2.connectedComponentsWithStats(safe_c, connectivity=8)
    text_like = stroke_analyzer._text_like(st, k)
    keep = np.zeros(n, bool)
    dt = cv2.distanceTransform(safe_c, cv2.DIST_L2, 3)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area * k * k or max(w, h) < min_len * k or text_like[i]:
            continue
        sub = lab[y:y + h, x:x + w] == i
        thick = 2.0 * float(dt[y:y + h, x:x + w][sub].max())
        fill = a / float(w * h)
        if thick > 14 * k or (fill > 0.85 and max(w, h) < 3.5 * min(w, h)):
            continue                                                   # solid gray block / graphic, not a stroke
        if float(diff[y:y + h, x:x + w][sub].mean()) < hi_t:
            continue
        keep[i] = True
    lut = np.zeros(n, np.uint8); lut[keep] = 255; lut[0] = 0
    mask = lut[lab]
    if mask.any():                                                      # grow into faint edges of confirmed strokes
        halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(4 * k)) + 1,) * 2))
        lc = cv2.bitwise_and(cv2.bitwise_and(low, halo), cv2.bitwise_not(near_print))
        n2, l2 = cv2.connectedComponents(lc, connectivity=8)
        sel = np.zeros(n2, bool); sel[np.unique(l2[mask > 0])] = True; sel[0] = False
        lut2 = np.zeros(n2, np.uint8); lut2[sel] = 255
        mask = cv2.bitwise_or(mask, lut2[l2])
    # drop word-like blobs (gray printed text that merged into words): aligned, similar-height neighbours
    if mask.any():
        n3, l3, s3, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        drop = np.zeros(n3, bool)
        for i in range(1, n3):
            x, y, w, h, a = s3[i]
            if not (6 * k <= h <= 30 * k and w <= 150 * k):
                continue
            cy, cx = y + h / 2, x + w / 2
            near = [j for j in range(1, n3) if j != i and abs(s3[j][1] + s3[j][3] / 2 - cy) < 0.6 * h
                    and abs(s3[j][0] + s3[j][2] / 2 - cx) < 14 * h and abs(s3[j][3] - h) < 0.6 * h + 3]
            fill = a / float(w * h)
            if len(near) >= 1 or (fill < 0.45 and w > 2.2 * h and h > 9 * k):
                drop[i] = True
        if drop.any():
            lut3 = np.full(n3, 255, np.uint8); lut3[drop] = 0; lut3[0] = 0
            mask = lut3[l3]
    # pencil-like regions that hug printed content: not removed automatically, flagged for review
    unsafe = cv2.bitwise_and(cand, near_print)
    unsafe = cv2.bitwise_and(unsafe, cv2.bitwise_not(cv2.dilate(printed, np.ones((3, 3), np.uint8))))   # exclude print itself
    unsafe = cv2.morphologyEx(unsafe, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return PencilResult(mask, unsafe, "classical", None)

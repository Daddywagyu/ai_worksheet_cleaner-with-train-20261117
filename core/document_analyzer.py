"""Structure analysis of the ORIGINAL worksheet (before removal).

Finds printed content that must be protected: dark achromatic print (text, numbers,
symbols), solid lines / table rules, dotted tracing lines and large coloured
illustrations.  OCR is not bundled in Phase 1 (`ocr_used=False` is reported).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core import color_detector, stroke_analyzer


@dataclass
class StructureAnalysis:
    printed_dark: np.ndarray            # near-black achromatic print (text, symbols)
    lines: np.ndarray                   # long horizontal / vertical rules
    dotted: np.ndarray                  # dotted tracing guides
    illustration: np.ndarray            # thick coloured artwork + its surroundings
    ocr_used: bool = False
    stats: dict = field(default_factory=dict)


def _dotted_lines(gray_mask: np.ndarray, k: float) -> np.ndarray:
    n, lab, st, cen = cv2.connectedComponentsWithStats(gray_mask, connectivity=8)
    out = np.zeros_like(gray_mask)
    w, h, a = st[:, 2], st[:, 3], st[:, 4]
    cand = np.where((np.arange(n) > 0) & (w >= 2 * k) & (h >= 2 * k) & (w <= 14 * k) & (h <= 14 * k)
                    & (a >= 3 * k * k))[0]
    if len(cand) < 5 or len(cand) > 8000:
        return out
    cx, cy = cen[cand, 0], cen[cand, 1]
    keep = np.zeros(len(cand), bool)
    for j in range(len(cand)):
        d_row = (np.abs(cy - cy[j]) < 2 * k) & (np.abs(cx - cx[j]) > 4 * k) & (np.abs(cx - cx[j]) < 45 * k)
        d_col = (np.abs(cx - cx[j]) < 2 * k) & (np.abs(cy - cy[j]) > 4 * k) & (np.abs(cy - cy[j]) < 45 * k)
        keep[j] = d_row.sum() >= 3 or d_col.sum() >= 3
    lut = np.zeros(n, np.uint8)
    lut[cand[keep]] = 255
    return lut[lab]


def _crisp_rules_only(cand, L, k):
    """Printed rules are thin and uniformly dark; hand-drawn (pencil) lines are thicker and grainy.
    Components that are too thick or whose darkness varies too much are NOT treated as printed rules."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    dt = cv2.distanceTransform(cand, cv2.DIST_L2, 3)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        sub = lab[y:y + h, x:x + w] == i
        core = sub & (dt[y:y + h, x:x + w] >= 1.0)
        vals = L[y:y + h, x:x + w][core if core.sum() > 20 else sub].astype(np.float32)
        thick = 2.0 * float(np.median(dt[y:y + h, x:x + w][sub]))        # typical stroke width (robust at grid crossings)
        keep[i] = thick <= max(8.0 * k, 6.0)   # thin = printed rule (blur widens rules, so no tighter limit); only clearly thick strokes are not rules
    lut = np.zeros(n, np.uint8)
    lut[keep] = 255
    lut[0] = 0
    return lut[lab]


def analyze(bgr: np.ndarray, dpi: float = 200.0) -> StructureAnalysis:
    k = dpi / 200.0
    L, _, _, chroma, _ = color_detector.lab_chroma(bgr)
    achromatic = chroma < 18
    printed = ((L < 100) & achromatic).astype(np.uint8) * 255

    gray_ink = ((L < 190) & achromatic).astype(np.uint8) * 255
    ln = max(15, int(40 * k))
    horiz = cv2.morphologyEx(gray_ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ln, 1)))
    vert = cv2.morphologyEx(gray_ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ln)))
    lines = cv2.bitwise_or(horiz, vert)
    # faint printed rules (light-gray grid lines): contrast against LOCAL paper instead of absolute gray
    rr = max(3, int(round(10 * k)))
    bg = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1)))
    faint = (((bg.astype(np.int16) - L.astype(np.int16)) > 10) & achromatic).astype(np.uint8) * 255
    lf = max(15, int(60 * k))
    faint_lines = cv2.bitwise_or(
        cv2.morphologyEx(faint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (lf, 1))),
        cv2.morphologyEx(faint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, lf))))
    lines = _crisp_rules_only(cv2.bitwise_or(lines, faint_lines), L, k)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    dotted = _dotted_lines(gray_ink, k)

    colour = ((chroma > 25)).astype(np.uint8) * 255                 # any hue: cartoons, logos
    body = stroke_analyzer._photo_body(colour, k)
    if body.any():
        r = int(round(6 * k))
        body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return StructureAnalysis(printed, lines, dotted, body, False, {
        "printed_px": int((printed > 0).sum()), "line_px": int((lines > 0).sum()),
        "dotted_px": int((dotted > 0).sum()), "illustration_px": int((body > 0).sum())})

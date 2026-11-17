"""Post-processing inspection.  Reports measurements and review regions;
it never claims a perfect restoration."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from config import Settings
from core import color_detector
from core.document_analyzer import StructureAnalysis


@dataclass
class QAReport:
    changed_outside_px: int = 0
    printed_preserved: float = 1.0
    residual_ratio: float = 0.0
    review_regions: list = field(default_factory=list)     # [x, y, w, h, reason]
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def check(original: np.ndarray, restored: np.ndarray, final_mask: np.ndarray,
          analysis: StructureAnalysis | None, settings: Settings, methods: dict | None = None,
          uncertain: np.ndarray | None = None) -> QAReport:
    k = settings.scale
    rep = QAReport()
    if uncertain is not None and uncertain.any():              # AI saw ink-like pixels it is not sure about
        un = cv2.bitwise_and(uncertain, cv2.bitwise_not(cv2.dilate(final_mask, np.ones((5, 5), np.uint8))))
        n, _, st, _ = cv2.connectedComponentsWithStats((un > 0).astype(np.uint8))
        for i in range(1, n):
            if st[i][4] >= 80 * k * k:
                rep.review_regions.append([int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3]),
                                           "AI ไม่มั่นใจว่าเป็นลายมือ — ไม่ได้ลบ ควรตรวจด้วยตา"])
    if not final_mask.any():
        rep.notes.append("ไม่พบรอยที่ต้องลบในหน้านี้")
        return rep
    r = max(1, int(round(5 * k)))
    near = cv2.dilate(final_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    diff = np.abs(restored.astype(np.int16) - original.astype(np.int16)).max(axis=2)
    rep.changed_outside_px = int(((diff > 0) & (near == 0)).sum())
    if rep.changed_outside_px:
        rep.notes.append(f"พบพิกเซล {rep.changed_outside_px} จุดเปลี่ยนนอกบริเวณ Mask — ควรตรวจสอบ")
    if analysis is not None:
        printed = cv2.bitwise_or(cv2.bitwise_or(analysis.printed_dark, analysis.lines), analysis.dotted)
        keep = (printed > 0) & (cv2.dilate(final_mask, np.ones((3, 3), np.uint8)) == 0)
        if keep.any():
            rep.printed_preserved = float((diff[keep] < 40).mean())
        # pen that overlapped printed content -> cannot be recovered with certainty
        touch = cv2.dilate(printed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(4 * k)) + 1,) * 2))
        n, lab, st, _ = cv2.connectedComponentsWithStats((final_mask > 0).astype(np.uint8))
        for i in range(1, n):
            x, y, w, h, _a = st[i]
            sub = lab[y:y + h, x:x + w] == i
            if (touch[y:y + h, x:x + w][sub] > 0).any():
                rep.review_regions.append([int(x), int(y), int(w), int(h),
                                           "รอยทับเนื้อหาพิมพ์ — กู้คืนไม่แน่ชัด ควรตรวจด้วยตา"])
    high, _ = color_detector.detect_pen_pixels(restored, settings)
    rep.residual_ratio = float(((high > 0) & (near > 0)).sum()) / max(1, int((final_mask > 0).sum()))
    if rep.residual_ratio > 0.02:
        rep.notes.append(f"ยังเหลือสีคล้ายรอยปากกา {rep.residual_ratio * 100:.1f}% — ลองเพิ่ม Mask Expansion/ความไว")
    if rep.review_regions:
        rep.notes.append(f"มี {len(rep.review_regions)} บริเวณที่รอยปากกาทับเนื้อหาพิมพ์: "
                         "เนื้อหาที่ถูกบังกู้คืนไม่ได้แน่ชัด (ไม่ได้สร้างเนื้อหาใหม่)")
    return rep

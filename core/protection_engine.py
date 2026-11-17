"""Protection mask: pixels that must NOT be removed automatically (conservative logic)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core import document_analyzer


@dataclass
class Protection:
    mask: np.ndarray                  # union of everything protected
    analysis: document_analyzer.StructureAnalysis


def build_protection(bgr: np.ndarray, dpi: float = 200.0, strength: float = 0.5) -> Protection:
    """`strength` 0..1 grows the protected area by up to 3 px (at 200 DPI)."""
    a = document_analyzer.analyze(bgr, dpi)
    m = cv2.bitwise_or(cv2.bitwise_or(a.printed_dark, a.lines), cv2.bitwise_or(a.dotted, a.illustration))
    r = int(round(3 * float(np.clip(strength, 0, 1)) * dpi / 200.0))
    if r > 0:
        printed = cv2.bitwise_or(cv2.bitwise_or(a.printed_dark, a.lines), a.dotted)
        m = cv2.bitwise_or(m, cv2.dilate(printed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))))
    return Protection(m, a)

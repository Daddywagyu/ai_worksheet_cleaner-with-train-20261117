"""Detection mask  -  Protection mask  =  Final removal mask; then restoration + QA."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from config import Settings
from core import color_detector, protection_engine, quality_checker, restoration_engine, stroke_analyzer
from core.mask_editor import MaskEditor


@dataclass
class MaskSet:
    detection: np.ndarray
    protection: np.ndarray
    final: np.ndarray
    analysis: object = None
    uncertain: np.ndarray | None = None
    ai_device: list | None = None
    pencil: np.ndarray | None = None
    pencil_source: str | None = None


@dataclass
class PageResult:
    masks: MaskSet
    final_mask: np.ndarray            # after manual edits
    restored: np.ndarray | None
    residual_ratio: float = 0.0
    methods: dict = field(default_factory=dict)
    qa: quality_checker.QAReport | None = None

    @property
    def auto_mask(self):
        return self.masks.final


def generate_masks(bgr: np.ndarray, settings: Settings, detector=None, pencil_detector=None) -> MaskSet:
    """settings.ai_mode: off = classical colour | union = colour + AI | ai_only.  `detector` is an
    ai.ink_detector.InkDetector (created once by the caller).  AI output is still filtered by the
    protection mask, so it can never delete protected printed content automatically."""
    from config import AI_CONF
    uncertain, dev = None, None
    if settings.ai_mode == "off":
        det = color_detector.detect_mask(bgr, settings)
    else:
        if detector is None:
            from ai.openvino_engine import ModelNotAvailable
            raise ModelNotAvailable("เลือกโหมด AI แต่ไม่มีตัวตรวจจับ/โมเดล")
        res = detector.predict(bgr, settings.process_dpi, AI_CONF.get(settings.mode, 0.6))
        ai = np.zeros_like(res.red)
        if settings.detect_red:
            ai = cv2.bitwise_or(ai, res.red)
        if settings.detect_blue:
            ai = cv2.bitwise_or(ai, res.blue)
        if settings.detect_handwriting:
            ai = cv2.bitwise_or(ai, res.other)
        det = stroke_analyzer.expand(ai, settings)
        if settings.ai_mode == "union":
            det = cv2.bitwise_or(det, color_detector.detect_mask(bgr, settings))
        uncertain, dev = res.uncertain, res.device
    prot = protection_engine.build_protection(bgr, settings.process_dpi, settings.preservation)
    pencil, psrc, pencil_strict = None, None, None
    if settings.detect_pencil:
        from config import PENCIL_CONF
        from ai import pencil_detector as pd
        if pencil_detector is not None:
            pr = pencil_detector.predict_pencil(bgr, settings.process_dpi, PENCIL_CONF.get(settings.mode, 0.7))
        else:
            pr = pd.classical_pencil_mask(bgr, settings.process_dpi, prot.analysis, settings.mode == "conservative")
        pencil, psrc = stroke_analyzer.expand(pr.mask, settings), pr.source
        pencil_strict = pencil if pr.source != "ai" else stroke_analyzer.expand(pr.strict, settings)
        det = cv2.bitwise_or(det, pencil)
        uncertain = pr.uncertain if uncertain is None else cv2.bitwise_or(uncertain, pr.uncertain)
        dev = pr.device or dev
    final = cv2.bitwise_and(det, cv2.bitwise_not(prot.mask))
    if pencil is not None and settings.pencil_override_print and psrc == "ai":
        # dark graphite looks like dark print; a very confident AI pencil mask may override ONLY the
        # dark-print part of the protection (rules, dotted guides and illustrations stay protected).
        a = prot.analysis
        hard = cv2.bitwise_or(cv2.bitwise_or(a.lines, a.dotted), a.illustration)
        strict = cv2.bitwise_and(pencil_strict, cv2.bitwise_not(hard))
        final = cv2.bitwise_or(final, strict)
    return MaskSet(det, prot.mask, final, prot.analysis, uncertain, dev, pencil, psrc)


def process_page(bgr: np.ndarray, settings: Settings, editor: MaskEditor | None = None,
                 masks: MaskSet | None = None, cancel: Callable[[], bool] | None = None,
                 detector=None, pencil_detector=None) -> PageResult:
    masks = masks or generate_masks(bgr, settings, detector, pencil_detector)
    final = editor.final_mask(masks.final) if editor is not None and editor.shape == masks.final.shape else masks.final
    res = restoration_engine.restore(bgr, final, settings, cancel)
    qa = quality_checker.check(bgr, res.image, final, masks.analysis, settings, res.methods, masks.uncertain)
    return PageResult(masks, final, res.image, res.residual_ratio, res.methods, qa)

"""AI ink segmentation (Phase 2) via OpenVINO.  Requires models/ink_segmentation/ink_unet.xml.
The shipped model (if any) is a PROTOTYPE trained on synthetic data - see ink_unet.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai import device_manager, model_manager
from ai.openvino_engine import ModelNotAvailable, OpenVinoEngine
from ai.tiling import predict_probs
from core import color_detector

RED, BLUE, OTHER = 2, 3, 5


@dataclass
class InkResult:
    red: np.ndarray            # uint8 0/255 full-resolution masks
    blue: np.ndarray
    other: np.ndarray
    uncertain: np.ndarray      # ink-like but below the confidence threshold
    device: list
    seconds: float = 0.0


class InkDetector:
    KEY = "ink"

    def __init__(self, device: str = "AUTO", model_path: str | Path | None = None):
        spec = next(m for m in model_manager.REGISTRY if m.key == self.KEY)
        path = Path(model_path) if model_path else spec.path
        if not path.is_file():
            raise ModelNotAvailable(f"ยังไม่มีโมเดล {self.KEY} ({path}) — ใช้ตัวตรวจจับแบบดั้งเดิมแทน")
        meta_p = path.with_suffix(".json")
        self.meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        self.tile = int(self.meta.get("tile", 256))
        self.train_dpi = int((self.meta.get("meta") or {}).get("train_dpi", 100))
        rep = device_manager.probe()
        chosen, why = device_manager.choose_device(device, rep)
        order = [chosen] + [d for d in device_manager.PRIORITY if d != chosen]
        self.engine, self.fallback_notes = None, []
        for d in order:                                   # never claim a device that failed to compile
            if d not in {x.split(".")[0] for x in rep.devices}:
                continue
            try:
                self.engine = OpenVinoEngine().load(path, d)
                break
            except Exception as e:                        # noqa: BLE001
                self.fallback_notes.append(f"คอมไพล์บน {d} ไม่สำเร็จ: {e}")
        if self.engine is None:
            raise ModelNotAvailable("คอมไพล์โมเดลบนอุปกรณ์ใดๆ ไม่สำเร็จ: " + "; ".join(self.fallback_notes))
        self.requested, self.reason = device, why

    @classmethod
    def available(cls) -> bool:
        return next(m for m in model_manager.REGISTRY if m.key == cls.KEY).available

    @property
    def actual_device(self) -> list:
        return self.engine.actual

    def _infer(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([self.engine.predict(x[i:i + 1]) for i in range(len(x))])

    def _probs(self, bgr: np.ndarray, dpi: float) -> np.ndarray:
        s = self.train_dpi / float(dpi)
        small = bgr if abs(s - 1) < 0.02 else cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        return predict_probs(small, self._infer, tile=self.tile, stride=self.tile * 3 // 4, batch=4)

    def predict(self, bgr: np.ndarray, dpi: float, conf: float = 0.6) -> InkResult:
        import time
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        probs = self._probs(bgr, dpi)
        ink = probs[[RED, BLUE, OTHER]]
        p_ink = cv2.resize(ink.sum(0), (w, h), interpolation=cv2.INTER_LINEAR)
        cls = cv2.resize(ink.argmax(0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        # snap the (low-resolution) AI region to real ink pixels so mask edges are pixel-accurate
        L, _, _, chroma, _ = color_detector.lab_chroma(bgr)
        step = max(1, int(np.sqrt(L.size / 300_000)))
        paper_l = float(np.percentile(L[::step, ::step], 85))
        inkish = ((paper_l - L.astype(np.float32)) > 18) | (chroma > 14)
        sure = (p_ink >= conf) & inkish
        unsure = (p_ink >= 0.3) & (p_ink < conf) & inkish
        to = lambda m: m.astype(np.uint8) * 255
        return InkResult(to(sure & (cls == 0)), to(sure & (cls == 1)), to(sure & (cls == 2)), to(unsure),
                         list(self.engine.actual), time.perf_counter() - t0)

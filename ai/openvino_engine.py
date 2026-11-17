"""Thin OpenVINO runtime wrapper.  Records the device that actually executes the model."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ModelNotAvailable(FileNotFoundError):
    pass


class OpenVinoEngine:
    def __init__(self):
        self.compiled = None
        self.requested = None
        self.actual: list[str] = []

    def load(self, model_path: str | Path, device: str = "CPU") -> "OpenVinoEngine":
        import openvino as ov
        p = Path(model_path)
        if not p.is_file():
            raise ModelNotAvailable(f"ไม่พบไฟล์โมเดล: {p}")
        core = ov.Core()
        model = core.read_model(str(p))
        self.compiled = core.compile_model(model, device)
        self.requested = device
        try:
            self.actual = list(self.compiled.get_property("EXECUTION_DEVICES"))
        except Exception:                                       # noqa: BLE001
            self.actual = [device]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.compiled is None:
            raise RuntimeError("ยังไม่ได้โหลดโมเดล")
        return self.compiled(x)[0]

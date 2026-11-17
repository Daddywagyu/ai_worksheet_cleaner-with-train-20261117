"""Registry of model files.  Phase 1 ships NO trained models; this reports what is missing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sys as _sys

# Frozen (PyInstaller) builds keep data next to the bundled modules (sys._MEIPASS).
_BASE = Path(getattr(_sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
MODELS_DIR = _BASE / "models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    rel_path: str
    phase: int
    purpose: str

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.rel_path

    @property
    def available(self) -> bool:
        return self.path.is_file()


REGISTRY = (
    ModelSpec("ink", "ink_segmentation/ink_unet.xml", 2, "7-class ink segmentation (U-Net/MobileNetV3)"),
    ModelSpec("pencil", "pencil_segmentation/pencil_unet.xml", 3, "pencil / graphite segmentation"),
    ModelSpec("restoration", "restoration/restore_net.xml", 4, "learned restoration"),
)


def status() -> list[dict]:
    return [{"key": m.key, "path": str(m.path), "available": m.available, "phase": m.phase,
             "purpose": m.purpose} for m in REGISTRY]


def missing() -> list[ModelSpec]:
    return [m for m in REGISTRY if not m.available]

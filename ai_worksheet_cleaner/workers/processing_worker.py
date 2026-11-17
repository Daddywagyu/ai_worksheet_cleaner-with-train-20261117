"""Background QThreads (UI never blocks).  Each worker opens its own DocumentLoader and
supports safe cancel."""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from config import Settings
from core import mask_generator as mg
from core.document_loader import DocumentLoader
from core.export_manager import ExportError, ExportOptions, export_document
from core.mask_editor import MaskEditor
from ai.openvino_engine import ModelNotAvailable


class ProcessingWorker(QThread):
    progress = Signal(int, int, str)
    pageDone = Signal(int, str, object, dict)   # index, cleaned png ('' for detect), MaskSet, info
    failed = Signal(str)
    cancelled = Signal()
    completed = Signal()

    def __init__(self, path: str, password, indices, settings: Settings, edits: dict, tmp_dir: str,
                 detect_only: bool = False, parent=None):
        super().__init__(parent)
        self.path, self.password = path, password
        self.indices, self.settings = list(indices), Settings(**settings.to_dict())
        self.edits, self.tmp_dir, self.detect_only = edits, tmp_dir, detect_only
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self):
        try:
            detector = None
            if self.settings.ai_mode != "off":
                from ai.ink_detector import InkDetector
                detector = InkDetector(self.settings.ai_device)      # raises ModelNotAvailable if no model
                self.progress.emit(0, 1, f"โหลดโมเดล AI บน {detector.actual_device}")
            pencil_det = None
            if self.settings.detect_pencil and self.settings.ai_mode != "off":
                from ai.pencil_detector import PencilDetector
                if PencilDetector.available():
                    pencil_det = PencilDetector(self.settings.ai_device)
                    self.progress.emit(0, 1, f"โหลดโมเดลดินสอบน {pencil_det.actual_device}")
            with DocumentLoader(self.path, self.password) as loader:
                total = len(self.indices)
                for n, idx in enumerate(self.indices):
                    if self.is_cancelled():
                        self.cancelled.emit()
                        return
                    t0 = time.perf_counter()
                    self.progress.emit(n, total, f"หน้า {idx + 1}/{loader.page_count}: โหลดภาพ")
                    img, used = loader.render(idx, self.settings.process_dpi)
                    s = Settings(**self.settings.to_dict())
                    s.process_dpi = used
                    self.progress.emit(n, total, f"หน้า {idx + 1}: วิเคราะห์โครงสร้าง/ตรวจจับรอย")
                    masks = mg.generate_masks(img, s, detector, pencil_det)
                    if self.detect_only:
                        self.pageDone.emit(idx, "", masks, {"dpi": used, "seconds": time.perf_counter() - t0})
                    else:
                        editor = None
                        if idx in self.edits:
                            editor = MaskEditor(img.shape[:2])
                            editor.set_layers(*self.edits[idx])
                        self.progress.emit(n, total, f"หน้า {idx + 1}: ฟื้นฟูพื้นหลัง + ตรวจคุณภาพ")
                        res = mg.process_page(img, s, editor, masks, cancel=self.is_cancelled, detector=detector, pencil_detector=pencil_det)
                        if self.is_cancelled():
                            self.cancelled.emit()
                            return
                        path = str(Path(self.tmp_dir) / f"page_{idx}.png")
                        mpath = str(Path(self.tmp_dir) / f"mask_{idx}.png")
                        if not (cv2.imwrite(path, res.restored) and cv2.imwrite(mpath, res.final_mask)):
                            raise IOError("เขียนไฟล์ชั่วคราวไม่สำเร็จ")
                        info = {"dpi": used, "mask_px": int((res.final_mask > 0).sum()),
                                "methods": res.methods, "mask_path": mpath,
                                "seconds": round(time.perf_counter() - t0, 3),
                                "inference_device": (f"AI บน {', '.join(detector.actual_device)}" + (" (สำรองจาก " + detector.requested + ")" if detector.requested not in ("AUTO",) and detector.requested not in detector.actual_device else "")
                                             if detector else "CPU (OpenCV, ไม่ใช้โมเดล AI)"),
                                "pencil_source": masks.pencil_source,
                                "qa": res.qa.to_dict() if res.qa else None}
                        self.pageDone.emit(idx, path, masks, info)
                    del img
                    gc.collect()
                    self.progress.emit(n + 1, total, f"หน้า {idx + 1}: เสร็จ")
            self.completed.emit()
        except ModelNotAvailable as e:
            self.failed.emit(f"ไม่พบโมเดล AI: {e}")
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class ExportWorker(QThread):
    progress = Signal(int, int)
    failed = Signal(str)
    cancelled = Signal()
    done = Signal(object)

    def __init__(self, src, pages, opts: ExportOptions, settings, password=None, parent=None):
        super().__init__(parent)
        self.args = (src, dict(pages), opts, Settings(**settings.to_dict()), password)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        src, pages, opts, settings, pw = self.args
        try:
            res = export_document(src, pages, opts, settings, pw,
                                  progress=lambda a, b: self.progress.emit(a, b), cancel=self._cancel.is_set)
            self.done.emit(res)
        except ExportError as e:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(e))
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")

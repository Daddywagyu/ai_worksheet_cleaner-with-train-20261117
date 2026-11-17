"""Import JPG/PNG/TIFF/BMP and PDF.  Original pixel resolution is never altered."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

from core.pdf_loader import PdfLoader, PdfOpenError, PdfPasswordRequired

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}
DEFAULT_IMAGE_DPI = 200          # used only when the file carries no DPI metadata
DocumentError = PdfOpenError
PasswordRequired = PdfPasswordRequired


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTS


def list_supported(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and is_supported(p))


class DocumentLoader:
    """Uniform page API for PDFs and images (same interface as PdfLoader)."""

    def __init__(self, path: str | Path, password: str | None = None):
        self.path = str(path)
        self.password = password
        self.ext = Path(path).suffix.lower()
        self.is_pdf = self.ext == ".pdf"
        self._pdf: PdfLoader | None = None
        self._sizes: list[tuple[int, int]] = []
        self._dpi: list[int] = []
        if not Path(path).is_file():
            raise DocumentError(f"ไม่พบไฟล์: {path}")
        if self.is_pdf:
            self._pdf = PdfLoader(path, password)
        elif self.ext in IMAGE_EXTS:
            self._index_image()
        else:
            raise DocumentError(f"ไม่รองรับไฟล์ชนิดนี้: {self.ext}")

    # -- images -----------------------------------------------------
    def _index_image(self):
        try:
            with Image.open(self.path) as im:
                n = getattr(im, "n_frames", 1)
                for i in range(n):
                    im.seek(i)
                    fr = ImageOps.exif_transpose(im)               # orientation from EXIF only
                    self._sizes.append(fr.size)
                    dpi = fr.info.get("dpi", (0, 0))[0] if isinstance(fr.info.get("dpi"), tuple) else 0
                    self._dpi.append(int(dpi) if dpi and 50 <= dpi <= 1200 else DEFAULT_IMAGE_DPI)
        except Exception as e:                                      # noqa: BLE001
            raise DocumentError(f"เปิดไฟล์ภาพไม่ได้: {e}") from e

    def _read_frame(self, i: int) -> np.ndarray:
        with Image.open(self.path) as im:
            im.seek(i)
            fr = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR)

    # -- common API ---------------------------------------------------
    @property
    def page_count(self) -> int:
        return self._pdf.page_count if self._pdf else len(self._sizes)

    def page_size(self, i: int) -> tuple[float, float]:
        if self._pdf:
            return self._pdf.page_size(i)
        w, h = self._sizes[i]
        return w * 72.0 / self._dpi[i], h * 72.0 / self._dpi[i]

    def native_dpi(self, i: int) -> int | None:
        return None if self._pdf else self._dpi[i]

    def has_text(self, i: int) -> bool:
        return self._pdf.has_text(i) if self._pdf else False

    def render(self, i: int, dpi: int) -> tuple[np.ndarray, int]:
        """PDF: render at `dpi`.  Image: native pixels (returns its nominal DPI)."""
        if self._pdf:
            return self._pdf.render(i, dpi)
        return self._read_frame(i), self._dpi[i]

    def thumbnail(self, i: int, max_side: int = 140) -> np.ndarray:
        if self._pdf:
            return self._pdf.thumbnail(i, max_side)
        img = self._read_frame(i)
        s = max_side / max(img.shape[:2])
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._pdf:
            self._pdf.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

"""Export: PNG / JPG / TIFF / PDF + optional original, comparison, mask, report.
Rules: never overwrite the source; never overwrite an existing output unless allowed;
preserve original pixel dimensions (images) / page size (PDF)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from config import Settings
from core import pdf_exporter
from core.document_loader import DocumentLoader

FORMATS = ("png", "jpg", "tiff", "pdf")


@dataclass
class ExportOptions:
    fmt: str = "png"
    out_dir: str = ""
    save_original: bool = False
    save_comparison: bool = False
    save_mask: bool = False
    save_report: bool = True
    jpeg_quality: int = 95
    allow_overwrite: bool = False


@dataclass
class PageExport:
    image_path: str                   # cleaned PNG (lossless working copy)
    mask_path: str | None = None
    info: dict = field(default_factory=dict)   # dpi, qa, methods ...


@dataclass
class ExportResult:
    files: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    verified: bool = True


class ExportError(Exception):
    pass


def _free_path(p: Path, allow_overwrite: bool, source: Path) -> Path:
    if p.resolve() == source.resolve():
        raise ExportError("ห้ามเขียนทับไฟล์ต้นฉบับ")
    if allow_overwrite or not p.exists():
        return p
    i = 1
    while True:
        q = p.with_name(f"{p.stem}_{i}{p.suffix}")
        if not q.exists():
            return q
        i += 1


def _save_image(path: Path, img: np.ndarray, fmt: str, quality: int):
    ext = {"png": ".png", "jpg": ".jpg", "tiff": ".tif"}[fmt]
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ExportError(f"บันทึกไม่สำเร็จ: {path}")
    path.write_bytes(buf.tobytes())            # unicode-safe path


def export_document(src_path: str | Path, pages: dict[int, PageExport], opts: ExportOptions,
                    settings: Settings | None = None, password: str | None = None,
                    progress=None, cancel=None) -> ExportResult:
    settings = settings or Settings()
    src = Path(src_path)
    if opts.fmt not in FORMATS:
        raise ExportError(f"รูปแบบไม่รองรับ: {opts.fmt}")
    out_dir = Path(opts.out_dir or src.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = ExportResult()
    stem = src.stem
    multi = len(pages) > 1 or (src.suffix.lower() == ".pdf")

    def name(suffix: str, ext: str, idx: int) -> Path:
        tag = f"_p{idx + 1}" if multi else ""
        return _free_path(out_dir / f"{stem}{suffix}{tag}{ext}", opts.allow_overwrite, src)

    with DocumentLoader(src, password) as loader:
        # 1) main output
        if opts.fmt == "pdf":
            out = _free_path(out_dir / f"{stem}_cleaned.pdf", opts.allow_overwrite, src)
            if loader.is_pdf:
                rep = pdf_exporter.export_pdf(src, out, {i: p.image_path for i, p in pages.items()},
                                              settings, password, progress, cancel)
                res.warnings += rep.warnings
                res.verified = rep.verified
            else:
                doc = pymupdf.open()
                for n in range(loader.page_count):
                    if n in pages:
                        img = cv2.imread(pages[n].image_path)
                        dpi = pages[n].info.get("dpi") or loader.native_dpi(n) or 200
                    else:
                        img, dpi = loader.render(n, 200)
                    h, w = img.shape[:2]
                    pg = doc.new_page(width=w * 72.0 / dpi, height=h * 72.0 / dpi)
                    ok, buf = cv2.imencode(".png", img)
                    pg.insert_image(pg.rect, stream=buf.tobytes())
                doc.save(str(out), garbage=3, deflate=True)
                doc.close()
                chk = pymupdf.open(str(out))
                res.verified = chk.page_count == loader.page_count
                chk.close()
            res.files.append(str(out))
        else:
            for k, (idx, pe) in enumerate(sorted(pages.items())):
                if cancel and cancel():
                    raise ExportError("ยกเลิกการส่งออก")
                img = cv2.imread(pe.image_path)
                out = name("_cleaned", {"png": ".png", "jpg": ".jpg", "tiff": ".tif"}[opts.fmt], idx)
                _save_image(out, img, opts.fmt, opts.jpeg_quality)
                res.files.append(str(out))
                if progress:
                    progress(k + 1, len(pages))
        # 2) optional extras (processed pages only)
        for idx, pe in sorted(pages.items()):
            clean = cv2.imread(pe.image_path)
            need_orig = opts.save_original or opts.save_comparison
            orig = loader.render(idx, pe.info.get("dpi") or settings.process_dpi)[0] if need_orig else None
            if opts.save_original:
                p = name("_original", ".png", idx)
                _save_image(p, orig, "png", 0)
                res.files.append(str(p))
            if opts.save_comparison:
                p = name("_comparison", ".png", idx)
                _save_image(p, np.hstack([orig, clean]), "png", 0)
                res.files.append(str(p))
            if opts.save_mask and pe.mask_path:
                p = name("_mask", ".png", idx)
                p.write_bytes(Path(pe.mask_path).read_bytes())
                res.files.append(str(p))
        if opts.save_report:
            p = _free_path(out_dir / f"{stem}_report.json", opts.allow_overwrite, src)
            rep = {"source": str(src), "format": opts.fmt, "settings": settings.to_dict(),
                   "processed_pages": {str(i): pe.info for i, pe in sorted(pages.items())},
                   "note": "รายงานนี้เป็นค่าวัดจริงของการประมวลผล ไม่ใช่การรับรองความสมบูรณ์ของการกู้คืน"}
            p.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            res.files.append(str(p))
    return res

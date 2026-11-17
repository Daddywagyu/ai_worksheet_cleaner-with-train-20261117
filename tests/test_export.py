"""Tests 6 & 9 + export formats."""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from config import Settings
from core import mask_generator as mg
from core.export_manager import ExportError, ExportOptions, PageExport, export_document


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_page(tmp, img, name="p0"):
    r = mg.process_page(img, Settings())
    ip, mp = str(tmp / f"{name}.png"), str(tmp / f"{name}_m.png")
    cv2.imwrite(ip, r.restored); cv2.imwrite(mp, r.final_mask)
    return PageExport(ip, mp, {"dpi": 200, "qa": r.qa.to_dict()})


@pytest.fixture()
def image_src(tmp_path, pair):
    p = tmp_path / "scan.jpg"
    cv2.imwrite(str(p), pair[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return p


@pytest.mark.parametrize("fmt", ["png", "jpg", "tiff"])
def test_image_formats_keep_dimensions_and_never_touch_source(tmp_path, image_src, fmt):
    before = sha(image_src)
    src_img = cv2.imread(str(image_src))
    out = tmp_path / "out"
    res = export_document(image_src, {0: make_page(tmp_path, src_img)}, ExportOptions(fmt=fmt, out_dir=str(out)))
    main = [f for f in res.files if "_cleaned" in f][0]
    assert cv2.imread(main).shape == src_img.shape
    assert Path(main).stem.startswith("scan_cleaned")              # original filename retained
    assert sha(image_src) == before


def test9_never_overwrites_source_or_existing_output(tmp_path, image_src):
    src_img = cv2.imread(str(image_src))
    page = make_page(tmp_path, src_img)
    out = tmp_path / "out"
    r1 = export_document(image_src, {0: page}, ExportOptions(fmt="png", out_dir=str(out)))
    first = sha(r1.files[0])
    r2 = export_document(image_src, {0: page}, ExportOptions(fmt="png", out_dir=str(out)))
    assert r2.files[0] != r1.files[0] and sha(r1.files[0]) == first      # numbered, not overwritten
    # writing into the source folder with a format that would collide with the source name
    same = export_document(image_src, {0: page}, ExportOptions(fmt="jpg", out_dir=str(image_src.parent)))
    assert Path(same.files[0]).name != image_src.name
    with pytest.raises(ExportError):
        export_document(image_src, {0: page}, ExportOptions(fmt="bogus"))


def test_extras_report_mask_comparison_original(tmp_path, image_src):
    src_img = cv2.imread(str(image_src))
    res = export_document(image_src, {0: make_page(tmp_path, src_img)},
                          ExportOptions(fmt="png", out_dir=str(tmp_path / "o"), save_original=True,
                                        save_comparison=True, save_mask=True, save_report=True))
    names = " ".join(Path(f).name for f in res.files)
    for tag in ("_cleaned", "_original", "_comparison", "_mask", "_report.json"):
        assert tag in names
    cmp_ = [f for f in res.files if "_comparison" in f][0]
    assert cv2.imread(cmp_).shape[1] == 2 * src_img.shape[1]


def test_pdf_source_to_pdf_keeps_pages_and_size(tmp_path, pair):
    pdf = pair[0]
    page = make_page(tmp_path, pair[2])
    res = export_document(pdf, {0: page}, ExportOptions(fmt="pdf", out_dir=str(tmp_path / "o"), save_report=False))
    d = pymupdf.open(res.files[0])
    assert res.verified and d.page_count == 1 and abs(d[0].rect.width - 595) < 0.5
    assert Path(res.files[0]).name == "ws_cleaned.pdf"


def test_image_to_pdf(tmp_path, image_src):
    src_img = cv2.imread(str(image_src))
    res = export_document(image_src, {0: make_page(tmp_path, src_img)}, ExportOptions(fmt="pdf", out_dir=str(tmp_path / "o")))
    d = pymupdf.open([f for f in res.files if f.endswith(".pdf")][0])
    assert d.page_count == 1 and res.verified
    w = d[0].get_images(full=True)[0][2]
    assert w == src_img.shape[1]                                   # full-resolution pixels embedded

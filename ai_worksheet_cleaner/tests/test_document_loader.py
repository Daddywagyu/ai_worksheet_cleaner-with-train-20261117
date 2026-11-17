import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

from core.document_loader import DocumentError, DocumentLoader, PasswordRequired, is_supported, list_supported


def test_supported_extensions(tmp_path):
    for n in ("a.jpg", "b.JPEG", "c.png", "d.tif", "e.tiff", "f.bmp", "g.pdf"):
        assert is_supported(n)
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "h.txt").write_bytes(b"x")
    assert len(list_supported(tmp_path)) == 7


@pytest.mark.parametrize("ext", [".png", ".jpg", ".bmp", ".tif"])
def test_image_native_resolution_preserved(tmp_path, ext):
    img = np.random.default_rng(1).integers(0, 255, (321, 457, 3), dtype=np.uint8)
    p = tmp_path / f"scan{ext}"
    cv2.imwrite(str(p), img)
    with DocumentLoader(p) as d:
        out, dpi = d.render(0, 150)                 # requested DPI must not resample images
        assert out.shape == (321, 457, 3) and d.page_count == 1 and dpi >= 50


def test_multipage_tiff_and_dpi_metadata(tmp_path):
    frames = [Image.fromarray(np.full((100, 80, 3), v, np.uint8)) for v in (10, 120, 240)]
    p = tmp_path / "m.tiff"
    frames[0].save(p, save_all=True, append_images=frames[1:], dpi=(300, 300))
    with DocumentLoader(p) as d:
        assert d.page_count == 3 and d.native_dpi(0) == 300
        assert int(d.render(1, 0)[0].mean()) == 120


def test_exif_orientation_applied(tmp_path):
    im = Image.fromarray(np.zeros((40, 80, 3), np.uint8))
    ex = im.getexif(); ex[0x0112] = 6                # rotate 90
    p = tmp_path / "o.jpg"; im.save(p, exif=ex)
    with DocumentLoader(p) as d:
        assert d.render(0, 0)[0].shape[:2] == (80, 40)


def test_pdf_pages_dpi_and_errors(tmp_path, pair):
    pdf = pair[0]
    with DocumentLoader(pdf) as d:
        for dpi in (150, 300, 600):
            img, used = d.render(0, dpi)
            assert used == dpi and abs(img.shape[1] - 595 * dpi / 72) < 2
    with pytest.raises(DocumentError):
        DocumentLoader(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"; bad.write_bytes(b"not an image")
    with pytest.raises(DocumentError):
        DocumentLoader(bad)
    enc = tmp_path / "enc.pdf"
    doc = pymupdf.open(); doc.new_page()
    doc.save(str(enc), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="p", owner_pw="o")
    with pytest.raises(PasswordRequired):
        DocumentLoader(enc)

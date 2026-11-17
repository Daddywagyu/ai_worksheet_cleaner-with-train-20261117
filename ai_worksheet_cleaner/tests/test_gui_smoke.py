import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import cv2
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QMessageBox


def wait_job(app, win, timeout=120):
    t = time.time()
    while win.worker is not None and win.worker.isRunning() and time.time() - t < timeout:
        app.processEvents(); time.sleep(0.02)
    for _ in range(20):
        app.processEvents()


def test_gui_flow_image_source(tmp_path, monkeypatch, pair):
    for n in ("information", "critical", "warning"):
        monkeypatch.setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    from app.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), pair[2])
    win = MainWindow(); win.show(); win.open_pdf(str(src))
    assert win.loader.page_count == 1 and win.orig_img.shape == pair[2].shape
    assert "ไม่ใช่ NPU" in win.panel.device_lbl.text() or "NPU" in win.panel.device_lbl.text()

    win.run_preview(); wait_job(app, win)
    st = win.pages[0]
    assert st.detection is not None and st.protection is not None and st.auto_mask.any()
    for k in range(3):                                         # detection / protection / final views
        win.preview.mask_view.setCurrentIndex(k)
    win.preview.mask_view.setCurrentIndex(0)

    win.preview.left.set_tool("add")
    win.on_brush_start(); win.on_brush_move(QPointF(10, 10), QPointF(60, 10)); win.on_brush_end()
    assert st.editor.has_edits()
    win.undo(); assert not st.editor.has_edits()
    win.run_process([0]); wait_job(app, win)
    assert st.processed_path and st.info["qa"] is not None
    assert "ไม่ใช้โมเดล AI" in st.info["inference_device"]
    for k in range(4):
        win.preview.mode.setCurrentIndex(k)
    from core.export_manager import ExportOptions, PageExport, export_document
    res = export_document(src, {0: PageExport(st.processed_path, st.info["mask_path"], st.info)},
                          ExportOptions(fmt="png", out_dir=str(tmp_path / "o")))
    assert (tmp_path / "sheet.png").exists() and res.files
    win.close()

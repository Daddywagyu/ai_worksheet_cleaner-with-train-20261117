"""Settings panel (right side) and export dialog."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from ai.ink_detector import InkDetector
from config import MODE_LABELS, Settings
from core.export_manager import FORMATS, ExportOptions


def _slider(lo, hi, val):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


class SettingsPanel(QWidget):
    changed = Signal()
    previewClicked = Signal()
    processCurrentClicked = Signal()
    processAllClicked = Signal()
    exportClicked = Signal()
    cancelClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        d = Settings()
        lay = QVBoxLayout(self)

        g = QGroupBox("โหมดการประมวลผล")
        f = QFormLayout(g)
        self.mode = QComboBox()
        for k, label in MODE_LABELS.items():
            self.mode.addItem(label, k)
        f.addRow(self.mode)
        self.preserve = _slider(0, 100, int(d.preservation * 100))
        f.addRow("Preservation Strength", self.preserve)
        lay.addWidget(g)

        g = QGroupBox("ชนิดรอยที่ลบ")
        f = QFormLayout(g)
        self.red = QCheckBox("ปากกาแดง"); self.red.setChecked(True)
        self.blue = QCheckBox("ปากกาน้ำเงิน"); self.blue.setChecked(True)
        self.sens_red, self.sens_blue = _slider(0, 100, int(d.sens_red * 100)), _slider(0, 100, int(d.sens_blue * 100))
        self.pencil = QCheckBox("ดินสอ (ทดลอง — แม่นยำต่ำ ตรวจผลทุกครั้ง)")
        self.pencil.setToolTip("ใช้โมเดลดินสอ (ต้นแบบ) เมื่อเลือกโหมด AI และมีไฟล์โมเดล; ไม่เช่นนั้นใช้ตัวตรวจจับแบบดั้งเดิมที่ระมัดระวังมาก\n"
                               "ดินสอที่ทับเนื้อหาพิมพ์จะไม่ถูกลบอัตโนมัติ แต่ถูกติดธงให้ตรวจด้วยตา")
        self.hand = QCheckBox("ลายมืออื่น ๆ (ต้องใช้โมเดล AI)")
        has_ai = InkDetector.available()
        self.hand.setEnabled(has_ai)
        if not has_ai:
            self.hand.setToolTip("ไม่พบ models/ink_segmentation/ink_unet.xml")
        f.addRow(self.red, self.sens_red)
        f.addRow(self.blue, self.sens_blue)
        f.addRow(self.pencil)
        f.addRow(self.hand)
        lay.addWidget(g)

        g = QGroupBox("การตรวจจับ (AI)")
        f = QFormLayout(g)
        self.ai_mode = QComboBox()
        self.ai_mode.addItem("Classical (สี) — ค่าเริ่มต้น", "off")
        self.ai_mode.addItem("AI + สี (ต้นแบบ)", "union")
        self.ai_mode.addItem("AI อย่างเดียว (ต้นแบบ)", "ai_only")
        self.ai_dev = QComboBox()
        for dev_name in ("AUTO", "NPU", "GPU", "CPU"):
            self.ai_dev.addItem(dev_name, dev_name)
        if not has_ai:
            for i in (1, 2):
                self.ai_mode.model().item(i).setEnabled(False)
            self.ai_mode.setToolTip("ยังไม่มีไฟล์โมเดล — ใช้โหมด Classical")
        else:
            self.ai_mode.setToolTip("โมเดลต้นแบบเทรนจากข้อมูลสังเคราะห์ ยังไม่ได้พิสูจน์กับใบงานจริง")
        f.addRow(self.ai_mode)
        f.addRow("อุปกรณ์ AI", self.ai_dev)
        lay.addWidget(g)

        g = QGroupBox("Mask / Restoration")
        f = QFormLayout(g)
        self.expand = _slider(0, 10, d.mask_expand)
        self.radius = _slider(1, 15, d.inpaint_radius)
        self.method = QComboBox()
        for label, key in (("อัตโนมัติ", "auto"), ("Telea", "telea"), ("Navier-Stokes", "ns"),
                           ("ประมาณพื้นหลัง", "background")):
            self.method.addItem(label, key)
        self.dpi = QComboBox()
        for v in (150, 300, 600):
            self.dpi.addItem(f"{v} DPI (เฉพาะ PDF)", v)
        self.dpi.setCurrentIndex(1)
        f.addRow("Mask Expansion", self.expand)
        f.addRow("Inpainting Radius", self.radius)
        f.addRow("วิธีฟื้นฟู", self.method)
        f.addRow("DPI เรนเดอร์", self.dpi)
        lay.addWidget(g)

        self.device_lbl = QLabel("")
        self.device_lbl.setWordWrap(True)
        self.device_lbl.setStyleSheet("color:#666;")
        lay.addWidget(self.device_lbl)

        self.b_prev = QPushButton("Preview (แสดง Mask)")
        self.b_cur = QPushButton("Process Current Page")
        self.b_all = QPushButton("Process All Pages")
        self.b_exp = QPushButton("Export…")
        self.b_cancel = QPushButton("ยกเลิกงาน")
        self.b_cancel.setEnabled(False)
        for b, sig in ((self.b_prev, self.previewClicked), (self.b_cur, self.processCurrentClicked),
                       (self.b_all, self.processAllClicked), (self.b_exp, self.exportClicked),
                       (self.b_cancel, self.cancelClicked)):
            b.clicked.connect(sig)
            lay.addWidget(b)
        lay.addStretch(1)
        for w in (self.mode, self.method, self.dpi, self.ai_mode, self.ai_dev):
            w.currentIndexChanged.connect(lambda _=0: self.changed.emit())
        for w in (self.red, self.blue, self.hand, self.pencil):
            w.toggled.connect(lambda _=False: self.changed.emit())
        for w in (self.sens_red, self.sens_blue, self.expand, self.radius, self.preserve):
            w.valueChanged.connect(lambda _=0, w=w: (not w.isSliderDown()) and self.changed.emit())
            w.sliderReleased.connect(self.changed)

    def settings(self) -> Settings:
        return Settings(mode=self.mode.currentData(), detect_red=self.red.isChecked(),
                        detect_blue=self.blue.isChecked(), sens_red=self.sens_red.value() / 100.0,
                        sens_blue=self.sens_blue.value() / 100.0, mask_expand=self.expand.value(),
                        inpaint_radius=self.radius.value(), restore_method=self.method.currentData(),
                        process_dpi=self.dpi.currentData(), preservation=self.preserve.value() / 100.0,
                        ai_mode=self.ai_mode.currentData(), ai_device=self.ai_dev.currentData(),
                        detect_handwriting=self.hand.isChecked(), detect_pencil=self.pencil.isChecked())

    def set_busy(self, busy: bool):
        for b in (self.b_prev, self.b_cur, self.b_all, self.b_exp):
            b.setEnabled(not busy)
        self.b_cancel.setEnabled(busy)


class ExportDialog(QDialog):
    def __init__(self, default_dir: str, is_pdf: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ส่งออก")
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.fmt = QComboBox()
        for x in FORMATS:
            self.fmt.addItem(x.upper(), x)
        if is_pdf:
            self.fmt.setCurrentIndex(FORMATS.index("pdf"))
        self.folder = QLineEdit(default_dir)
        b = QPushButton("เลือก…")
        b.clicked.connect(lambda: self.folder.setText(QFileDialog.getExistingDirectory(self, "โฟลเดอร์ปลายทาง", self.folder.text()) or self.folder.text()))
        row = QHBoxLayout(); row.addWidget(self.folder); row.addWidget(b)
        f.addRow("รูปแบบ", self.fmt)
        f.addRow("โฟลเดอร์", row)
        lay.addLayout(f)
        self.orig = QCheckBox("ส่งออกไฟล์ต้นฉบับ (หน้าที่ประมวลผล)")
        self.comp = QCheckBox("ส่งออกภาพเปรียบเทียบ Side-by-side")
        self.mask = QCheckBox("ส่งออก Mask ที่ลบ")
        self.report = QCheckBox("ส่งออกรายงาน (JSON)"); self.report.setChecked(True)
        for w in (self.orig, self.comp, self.mask, self.report):
            lay.addWidget(w)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def options(self) -> ExportOptions:
        return ExportOptions(fmt=self.fmt.currentData(), out_dir=self.folder.text(), save_original=self.orig.isChecked(),
                             save_comparison=self.comp.isChecked(), save_mask=self.mask.isChecked(),
                             save_report=self.report.isChecked())

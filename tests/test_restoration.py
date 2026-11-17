import hashlib

import numpy as np

from config import Settings
from core import mask_generator as mg
from core.mask_editor import MaskEditor


def test6_resolution_preserved(pair):
    _, _, dirty, _ = pair
    assert mg.process_page(dirty, Settings()).restored.shape == dirty.shape


def test10_reproducible(pair):
    _, _, dirty, _ = pair
    h = [hashlib.sha256(mg.process_page(dirty, Settings(mode="advanced")).restored.tobytes()).hexdigest() for _ in range(2)]
    assert h[0] == h[1]


def test_manual_edit_can_restore_and_add(pair):
    _, _, dirty, gt = pair
    e = MaskEditor(dirty.shape[:2])
    e.begin_stroke(); e.removed[:] = 255
    r = mg.process_page(dirty, Settings(), e)
    assert not r.final_mask.any() and np.array_equal(r.restored, dirty)
    e2 = MaskEditor(dirty.shape[:2]); e2.begin_stroke(); e2.paint((100, 100), (200, 100), 8, "add")
    assert mg.process_page(dirty, Settings(), e2).final_mask[100, 150] == 255


def test_pen_on_blank_paper_restored_to_paper():
    import cv2
    img = np.full((600, 800, 3), 245, np.uint8)
    cv2.line(img, (100, 300), (700, 320), (40, 40, 200), 11, cv2.LINE_AA)
    r = mg.process_page(img, Settings())
    assert np.abs(r.restored.astype(int) - 245).max() <= 12


def test_settings_yaml_default_is_safe_mode():
    from pathlib import Path
    from config import load_settings
    s = load_settings(Path(__file__).resolve().parent.parent / "config" / "settings.yaml")
    assert s.mode == "conservative" and s.process_dpi == 300

"""Acceptance tests 1, 2, 5 + protection behaviour."""
import cv2
import numpy as np

from config import Settings
from core import mask_generator as mg


def far_from(gt, px=31):
    return cv2.dilate(gt, np.ones((px, px), np.uint8)) == 0


def test1_red_marks_removed_printed_text_intact(pair):
    _, clean, dirty, gt = pair
    r = mg.process_page(dirty, Settings())
    assert (r.final_mask[gt > 0] > 0).mean() > 0.9
    dark = (clean.max(axis=2) < 90) & far_from(gt)
    assert (np.abs(r.restored.astype(int) - clean.astype(int)).max(axis=2)[dark] < 40).all()


def test2_blue_printed_illustration_untouched(pair):
    _, clean, dirty, gt = pair
    for mode in ("conservative", "balanced", "advanced"):
        r = mg.process_page(dirty, Settings(mode=mode))
        art = np.zeros(gt.shape, bool)
        art[int(115 * 200 / 72):int(245 * 200 / 72), int(395 * 200 / 72):int(525 * 200 / 72)] = True   # blue character
        art[int(415 * 200 / 72):int(535 * 200 / 72), int(405 * 200 / 72):int(505 * 200 / 72)] = True   # red apple
        assert int(((r.final_mask > 0) & art).sum()) == 0, mode
        assert np.array_equal(r.restored[art], dirty[art]), mode


def test5_nothing_changes_far_from_marks(pair):
    _, clean, dirty, gt = pair
    r = mg.process_page(dirty, Settings())
    far = far_from(gt)
    assert int((np.abs(r.restored.astype(int) - dirty.astype(int)).max(axis=2)[far] > 0).sum()) == 0


def test_dotted_tracing_lines_and_table_survive(pair):
    _, clean, dirty, gt = pair
    r = mg.process_page(dirty, Settings(mode="advanced"))
    a = r.masks.analysis
    assert a.stats["dotted_px"] > 1000 and a.stats["line_px"] > 1000     # analyzer sees them
    prot = (a.dotted > 0) | (a.lines > 0)
    assert int(((r.final_mask > 0) & prot).sum()) == 0                  # and never removes them


def test_clean_page_yields_empty_mask(pair):
    _, clean, _, _ = pair
    for mode in ("conservative", "balanced", "advanced"):
        assert int((mg.generate_masks(clean, Settings(mode=mode)).final > 0).sum()) < 50


def test_protection_mask_overrides_detection(pair):
    _, clean, dirty, gt = pair
    m = mg.generate_masks(dirty, Settings())
    assert int(((m.final > 0) & (m.protection > 0)).sum()) == 0
    assert (m.detection > 0).sum() >= (m.final > 0).sum()


def test_qa_flags_pen_over_print(pair):
    _, _, dirty, _ = pair
    r = mg.process_page(dirty, Settings())
    assert r.qa.review_regions and any("ไม่แน่ชัด" in x[4] for x in r.qa.review_regions)
    assert r.qa.changed_outside_px == 0 and r.qa.printed_preserved > 0.99

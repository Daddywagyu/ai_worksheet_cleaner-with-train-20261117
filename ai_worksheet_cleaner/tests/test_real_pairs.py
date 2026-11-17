"""Registration + weak-label derivation for REAL scan pairs (tested here on synthetic stand-ins with known truth)."""
import cv2
import numpy as np

from tests import worksheet as ws
from training import real_pdf_pairs as rp
from training.classes import BLUE, PENCIL, RED


def _scan_shift(img, angle=0.6, dx=14, dy=-9, scale=1.004):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale); M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245), flags=cv2.INTER_LINEAR), M


def test_registration_recovers_scan_offset(pair):
    _, clean, dirty, gt = pair
    moved, M = _scan_shift(dirty)
    warped, _ = rp.register(clean, moved)
    assert rp.alignment_score(warped, moved) > 0.95
    assert rp.alignment_score(clean, moved) < rp.alignment_score(warped, moved)       # registration really helped


def test_derived_labels_find_added_ink_and_not_clean_print(pair):
    _, clean, dirty, gt = pair
    moved, M = _scan_shift(dirty)
    gt_m = cv2.warpAffine(gt, M, (gt.shape[1], gt.shape[0]), flags=cv2.INTER_NEAREST)
    warped, _ = rp.register(clean, moved)
    lab, diag = rp.derive_labels(moved, warped)
    ink = np.isin(lab, (RED, BLUE, PENCIL))
    assert (ink & (gt_m > 0)).sum() / (gt_m > 0).sum() > 0.6                           # marks found
    far = cv2.dilate(gt_m, np.ones((41, 41), np.uint8)) == 0
    assert int((ink & far).sum()) < 0.0005 * ink.size                                  # print is not mistaken for ink
    red = (lab == RED)
    assert red.sum() > 0 and (red & (gt_m > 0)).sum() / red.sum() > 0.8                # red strokes classified as red


def test_clean_vs_shifted_clean_has_almost_no_ink(pair):
    _, clean, _, _ = pair
    moved, _ = _scan_shift(clean, angle=-0.4, dx=-11, dy=7)
    warped, _ = rp.register(clean, moved)
    lab, _ = rp.derive_labels(moved, warped)
    assert int(np.isin(lab, (RED, BLUE, PENCIL)).sum()) < 0.0008 * lab.size

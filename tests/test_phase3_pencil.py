"""Phase 3: pencil data, classical fallback, AI model, and acceptance tests 3 & 4."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ai import model_manager
from ai.pencil_detector import PencilDetector, classical_pencil_mask
from config import Settings
from core import mask_generator as mg
from tests import worksheet as ws
from training import synthetic_data as sd
from training.classes import PENCIL

PENCIL_ONLY = dict(detect_pencil=True, detect_red=False, detect_blue=False)


# ----------------------------------------------------------------- synthetic pencil data
def test_pencil_samples_are_grainy_gray_and_labelled():
    s = sd.make_sample(21, 0, pencil=True, n_strokes=0)
    g = s.labels == PENCIL
    assert g.sum() > 500
    px = s.dirty[g].astype(int)
    chroma = np.abs(px[:, 2] - px[:, 0]).mean()
    assert chroma < 40                                          # graphite is near-achromatic (tint + paper colour only)
    assert px.mean(axis=1).std() > 8                            # grain: uneven darkness along the stroke
    far = cv2.dilate(g.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    assert np.abs(s.dirty.astype(int) - s.target.astype(int)).max(axis=2)[far].max() == 0
    assert sd.make_sample(21, 0, pencil=True, n_strokes=0).dirty.tobytes() == s.dirty.tobytes()   # deterministic


def test_gray_printed_content_is_never_labelled_pencil():
    s = sd.make_sample(22, 0, pencil=True, n_strokes=0)
    clean_dark = s.target.min(axis=2) < 235
    printed_only = clean_dark & ~cv2.dilate((s.labels == PENCIL).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert (s.labels[printed_only] == PENCIL).sum() == 0
    assert (s.labels == 1).sum() > 5000                          # includes the gray hard negatives


def test_phase2_generator_unchanged_when_pencil_off():
    a, b = sd.make_sample(5, 0, n_strokes=4), sd.make_sample(5, 0, n_strokes=4, pencil=False)
    assert np.array_equal(a.dirty, b.dirty) and 4 not in np.unique(a.labels)


# ----------------------------------------------------------------- classical fallback
def test_classical_pencil_never_touches_clean_worksheet_structure(pair):
    """Dotted tracing guides, table rules, text and cartoons of a CLEAN page stay out of the mask."""
    _, clean, _, _ = pair
    for mode in ("conservative", "balanced", "advanced"):
        m = mg.generate_masks(clean, Settings(mode=mode, **PENCIL_ONLY))
        assert int((m.final > 0).sum()) < 50, mode


def test_classical_pencil_flags_instead_of_removing_near_print():
    rng = np.random.default_rng(1)
    img = np.full((400, 600, 3), 245, np.uint8)
    cv2.putText(img, "WORKSHEET", (60, 200), cv2.FONT_HERSHEY_DUPLEX, 2.0, (10, 10, 10), 3, cv2.LINE_AA)
    base = img.copy()
    a = np.zeros(img.shape[:2], np.float32)
    cv2.line(a, (50, 215), (550, 215), 1.0, 5, cv2.LINE_AA)              # graphite line hugging the print
    a = cv2.GaussianBlur(a, (0, 0), 0.8) * rng.uniform(0.5, 0.9, a.shape).astype(np.float32)
    img = (img * (1 - a[..., None]) + 90 * a[..., None]).astype(np.uint8)
    m = mg.generate_masks(img, Settings(mode="balanced", **PENCIL_ONLY))
    dark = base.max(axis=2) < 60
    assert int(((m.final > 0) & dark).sum()) == 0                          # printed letters never removed


# ----------------------------------------------------------------- acceptance test 3
def _wavy(a, x0, x1, y0, amp, period, thick):
    xs = np.arange(x0, x1, 3)
    pts = np.stack([xs, y0 + amp * np.sin(xs / period)], 1).astype(np.int32)
    cv2.polylines(a, [pts], False, 1.0, thick, cv2.LINE_AA)


def test3_pencil_cleaned_without_destroying_dotted_tracing_lines(pair):
    """Hand-drawn (wavy) graphite across the dotted tracing rows and in blank space.
    NOTE: ruler-straight thin pencil lines are indistinguishable from printed rules and stay protected."""
    _, clean, _, _ = pair
    h, w = clean.shape[:2]
    a = np.zeros((h, w), np.float32)
    rng = np.random.default_rng(3)
    for (x0, x1, y0) in ((120, 900, 694), (200, 780, 805), (300, 800, 1580), (250, 900, 1700)):
        _wavy(a, x0, x1, y0, 22, 18, 7)
    a = cv2.GaussianBlur(a, (0, 0), 0.8) * np.clip(rng.normal(0.72, 0.15, a.shape), 0.3, 1).astype(np.float32)
    dirty = np.clip(clean.astype(np.float32) * (1 - a[..., None]) + 85 * a[..., None], 0, 255).astype(np.uint8)
    r = mg.process_page(dirty, Settings(mode="balanced", **PENCIL_ONLY))
    an = r.masks.analysis
    assert an.stats["dotted_px"] > 1000
    dotted = an.dotted > 0
    assert int(((r.final_mask > 0) & dotted).sum()) == 0                  # dotted guides never in the removal mask
    removed_blank = (r.final_mask > 0) & (a > 0.25) & ~dotted
    assert removed_blank.sum() > 500                                       # and pencil in blank space IS removed
    far = cv2.dilate(((a > 0.05)).astype(np.uint8), np.ones((31, 31), np.uint8)) == 0
    assert int((np.abs(r.restored.astype(int) - dirty.astype(int)).max(axis=2)[far] > 0).sum()) == 0


# ----------------------------------------------------------------- acceptance test 4
def test4_pencil_over_printed_letters_is_flagged_not_silently_erased():
    img = np.full((300, 700, 3), 245, np.uint8)
    cv2.putText(img, "Trace the letters", (40, 150), cv2.FONT_HERSHEY_DUPLEX, 1.8, (12, 12, 12), 3, cv2.LINE_AA)
    base = img.copy()
    rng = np.random.default_rng(2)
    a = np.zeros(img.shape[:2], np.float32)
    _wavy(a, 30, 660, 135, 14, 22, 8)                                      # wavy graphite across the letters
    a = cv2.GaussianBlur(a, (0, 0), 1.0) * np.clip(rng.normal(0.6, 0.12, a.shape), 0.2, 0.9).astype(np.float32)
    img = np.clip(img.astype(np.float32) * (1 - a[..., None]) + 100 * a[..., None], 0, 255).astype(np.uint8)
    r = mg.process_page(img, Settings(mode="balanced", **PENCIL_ONLY))
    printed = base.max(axis=2) < 60
    assert int(((r.final_mask > 0) & printed).sum()) == 0                  # letters preserved
    assert any("ตรวจ" in x[4] for x in r.qa.review_regions)                # and the overlap is flagged for a human


# ----------------------------------------------------------------- AI pencil model (prototype)
spec = next(m for m in model_manager.REGISTRY if m.key == "pencil")


@pytest.mark.skipif(not spec.available, reason="pencil model not present")
def test_pencil_model_is_declared_prototype_and_runs_on_real_device():
    meta = json.loads(spec.path.with_suffix(".json").read_text(encoding="utf-8"))
    assert "synthetic" in json.dumps(meta).lower() and "ต้นแบบ" in meta["note"]
    det = PencilDetector("NPU")                                           # no NPU here -> honest fallback
    assert det.actual_device in (["CPU"], ["GPU"], ["NPU"]) and det.requested == "NPU"
    s = sd.make_sample(30, 0, pencil=True, n_strokes=0)
    r = det.predict_pencil(s.dirty, 100, 0.7)
    assert r.mask.shape == s.dirty.shape[:2] and r.source == "ai"
    m = mg.generate_masks(s.dirty, Settings(mode="balanced", process_dpi=100, **PENCIL_ONLY), None, det)
    assert m.pencil_source == "ai" and int(((m.final > 0) & (m.protection > 0)).sum()) == 0


def test_shipped_eval_reports_are_honest():
    root = Path(spec.path).parent.parent
    rep = json.loads((root / "eval_real_heldout_sets_1_4.json").read_text(encoding="utf-8"))
    assert rep["weak_labels"] is True and "not measured" in rep["ocr_text_preservation"]
    assert rep["n_pages"] == 36 and "ไม่มีชุดทดสอบแยก" in rep["note"]
    r = rep["summary_all"]["ai/red/thr0.85"]
    assert 0 <= r["recall"] <= 1 and 0 <= r["precision_tol2px"] <= 1
    assert (root / "archive_synthetic_models_v1" / "README.txt").is_file()


def test_missing_pencil_model_raises_cleanly(tmp_path):
    from ai.openvino_engine import ModelNotAvailable
    with pytest.raises(ModelNotAvailable):
        PencilDetector("CPU", tmp_path / "nope.xml")

"""Phase 2 runtime: ONNX/OpenVINO export fidelity, tiled inference, AI + protection integration."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openvino")

from ai.ink_detector import InkDetector
from ai.openvino_engine import ModelNotAvailable
from ai.tiling import predict_probs
from config import Settings
from core import mask_generator as mg
from training import synthetic_data as sd
from training.export_onnx import export


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    return d, export(None, str(d), fp16=False, name="ink_unet")


def test_openvino_matches_pytorch_fp32(exported):
    _, rep = exported
    assert rep["max_abs_logit_diff"] < 1e-2 and rep["argmax_agreement"] > 0.99


def test_fp16_export_agreement(tmp_path):
    rep = export(None, str(tmp_path), fp16=True, name="m16")
    assert rep["precision"] == "fp16" and rep["argmax_agreement"] > 0.97


def test_tiling_shapes_and_constant_prediction():
    def fake(x):                                       # always "class 3"
        out = np.zeros((len(x), 7, x.shape[2], x.shape[3]), np.float32)
        out[:, 3] = 10
        return out
    for shape in ((100, 90), (256, 256), (300, 500), (600, 333)):      # smaller / non-multiple sizes
        img = np.zeros((*shape, 3), np.uint8)
        p = predict_probs(img, fake)
        assert p.shape == (7, *shape) and np.allclose(p.sum(0), 1, atol=1e-4) and (p.argmax(0) == 3).all()


def test_detector_runs_and_reports_real_device(exported):
    d, _ = exported
    det = InkDetector("AUTO", d / "ink_unet.xml")
    assert det.actual_device == ["CPU"]                # this machine has no NPU/GPU -> must say CPU
    s = sd.make_sample(2, 0, n_strokes=5)
    r = det.predict(s.dirty, 100, 0.6)
    assert r.red.shape == s.dirty.shape[:2] and r.red.dtype == np.uint8 and r.device == ["CPU"]
    forced = InkDetector("NPU", d / "ink_unet.xml")    # NPU requested but absent -> honest fallback
    assert forced.actual_device == ["CPU"] and forced.requested == "NPU"


def test_ai_mode_never_removes_protected_content(exported, pair):
    d, _ = exported
    det = InkDetector("CPU", d / "ink_unet.xml")
    _, _, dirty, _ = pair
    s = Settings(ai_mode="ai_only", detect_handwriting=True, mode="advanced")
    m = mg.generate_masks(dirty, s, det)
    assert int(((m.final > 0) & (m.protection > 0)).sum()) == 0
    assert m.ai_device == ["CPU"]
    with pytest.raises(ModelNotAvailable):
        mg.generate_masks(dirty, s, None)              # AI requested without a model -> explicit error
    off = mg.generate_masks(dirty, Settings(), None)   # default path is untouched
    assert off.final.any()

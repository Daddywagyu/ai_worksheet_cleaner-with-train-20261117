"""Test 8: the app must report the device that REALLY executes inference."""
import numpy as np
import pytest

ov = pytest.importorskip("openvino")

from ai import device_manager as dm, model_manager
from ai.ink_detector import InkDetector
from ai.openvino_engine import ModelNotAvailable, OpenVinoEngine


def test_probe_matches_openvino():
    rep = dm.probe()
    assert rep.devices == list(ov.Core().available_devices)
    assert rep.npu_available == any(d.startswith("NPU") for d in rep.devices)


def test_choose_device_never_invents_npu():
    rep = dm.DeviceReport(openvino_available=True, devices=["CPU"])
    assert dm.choose_device("AUTO", rep)[0] == "CPU"
    dev, why = dm.choose_device("NPU", rep)
    assert dev == "CPU" and "สำรอง" in why
    rep2 = dm.DeviceReport(openvino_available=True, devices=["CPU", "GPU", "NPU"], npu_available=True)
    assert dm.choose_device("AUTO", rep2)[0] == "NPU"
    rep3 = dm.DeviceReport(openvino_available=True, devices=["CPU", "GPU"])
    assert dm.choose_device("AUTO", rep3)[0] == "GPU"


def test_self_test_reports_actual_execution_device():
    r = dm.self_test("CPU")
    assert r["ok"] and r["actual"] == ["CPU"]
    rep = dm.probe()
    if not rep.npu_available:
        with pytest.raises(Exception):               # no NPU -> compile must fail, never silently pass
            dm.self_test("NPU")


def test_engine_runs_real_model_and_missing_model_is_reported(tmp_path):
    import openvino.opset13 as ops
    p = ops.parameter([1, 4], dtype=np.float32, name="x")
    xml = tmp_path / "m.xml"
    ov.save_model(ov.Model([ops.relu(p)], [p], "m"), str(xml))
    eng = OpenVinoEngine().load(xml, "CPU")
    assert eng.actual == ["CPU"]
    assert (eng.predict(np.array([[-1, 2, -3, 4]], np.float32)) == [[0, 2, 0, 4]]).all()
    with pytest.raises(ModelNotAvailable):
        OpenVinoEngine().load(tmp_path / "nope.xml")


def test_models_are_honest_about_origin():
    """A shipped model must declare that it is a synthetic-data prototype; a missing one must raise."""
    import json
    spec = next(m for m in model_manager.REGISTRY if m.key == "ink")
    if InkDetector.available():
        meta = json.loads(spec.path.with_suffix(".json").read_text(encoding="utf-8"))
        assert "synthetic" in json.dumps(meta).lower() and "ต้นแบบ" in meta["note"]
    else:
        with pytest.raises(ModelNotAvailable):
            InkDetector()
    for m in model_manager.REGISTRY:
        if m.key == "restoration":
            assert not m.available                      # no learned restoration model exists (Phase 4)
        elif m.key == "pencil" and m.available:
            meta = json.loads(m.path.with_suffix(".json").read_text(encoding="utf-8"))
            assert "synthetic" in json.dumps(meta).lower() and "ต้นแบบ" in meta["note"]

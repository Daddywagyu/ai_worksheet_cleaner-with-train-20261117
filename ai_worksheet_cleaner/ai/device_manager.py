"""Detect OpenVINO devices and VERIFY (by really running a tiny model) where inference
executes.  Never claims NPU unless the compiled model reports it."""
from __future__ import annotations

from dataclasses import dataclass, field

PRIORITY = ("NPU", "GPU", "CPU")


@dataclass
class DeviceReport:
    openvino_available: bool = False
    version: str = ""
    devices: list = field(default_factory=list)
    npu_available: bool = False
    gpu_available: bool = False
    npu_driver: str | None = None
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        if not self.openvino_available:
            return "OpenVINO ไม่พร้อมใช้งาน"
        return f"OpenVINO {self.version} — พบอุปกรณ์: {', '.join(self.devices) or 'ไม่มี'}" + \
               ("" if self.npu_available else " (ไม่พบ NPU)")


def probe() -> DeviceReport:
    rep = DeviceReport()
    try:
        import openvino as ov
        core = ov.Core()
    except Exception as e:                                      # noqa: BLE001
        rep.notes.append(f"นำเข้า OpenVINO ไม่ได้: {e}")
        return rep
    rep.openvino_available, rep.version = True, ov.__version__
    rep.devices = list(core.available_devices)
    rep.npu_available = any(d.startswith("NPU") for d in rep.devices)
    rep.gpu_available = any(d.startswith("GPU") for d in rep.devices)
    if rep.npu_available:
        try:
            rep.npu_driver = str(core.get_property("NPU", "NPU_DRIVER_VERSION"))
        except Exception:                                       # noqa: BLE001
            rep.notes.append("อ่านเวอร์ชันไดรเวอร์ NPU ไม่ได้ — ตรวจสอบไดรเวอร์ Intel NPU")
    else:
        rep.notes.append("ไม่พบ NPU: ตรวจสอบว่าติดตั้งไดรเวอร์ Intel NPU บน Windows แล้ว")
    return rep


def choose_device(preference: str, rep: DeviceReport) -> tuple[str, str]:
    """preference: AUTO | NPU | GPU | CPU.  Returns (device, reason); falls back NPU>GPU>CPU."""
    avail = {d.split(".")[0] for d in rep.devices}
    order = PRIORITY if preference == "AUTO" else (preference,) + tuple(d for d in PRIORITY if d != preference)
    for d in order:
        if d in avail:
            why = "ตามที่เลือก" if d == preference or preference == "AUTO" else f"{preference} ไม่พร้อม จึงสำรองเป็น {d}"
            return d, why
    return "CPU", "ไม่พบอุปกรณ์ที่ต้องการ"


def self_test(device: str) -> dict:
    """Compile + run a tiny Relu model on `device`; report the device that really executed it."""
    import numpy as np
    import openvino as ov
    import openvino.opset13 as ops

    p = ops.parameter([1, 3, 8, 8], dtype=np.float32, name="x")
    model = ov.Model([ops.relu(p)], [p], "selftest")
    core = ov.Core()
    cm = core.compile_model(model, device)
    out = cm(np.random.default_rng(0).normal(size=(1, 3, 8, 8)).astype(np.float32))[0]
    try:
        actual = list(cm.get_property("EXECUTION_DEVICES"))
    except Exception:                                           # noqa: BLE001
        actual = [device]
    return {"requested": device, "actual": actual, "ok": bool((out >= 0).all())}

"""Checkpoint -> ONNX -> OpenVINO IR (FP16) with numerical verification.
   python -m training.export_onnx --ckpt runs/x/best.pt --out models/ink_segmentation"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.model import UNetMobileNetV3

TILE = 256


def export(ckpt: str | None, out_dir: str, tile: int = TILE, seed: int = 0, name: str = "ink_unet", fp16: bool = True) -> dict:
    import openvino as ov
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = UNetMobileNetV3(pretrained=False)
    meta = {"classes": [], "trained_on": "UNTRAINED (random weights)"}
    if ckpt:
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(data["model"]); meta = data["meta"]
    model.eval()
    onnx_path = out / f"{name}.onnx"
    x = torch.rand(2, 3, tile, tile)
    torch.onnx.export(model, x[:1], str(onnx_path), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(out / f"{name}.xml"), compress_to_fp16=fp16)
    with torch.no_grad():
        ref = model(x[:1]).numpy()
    got = ov.Core().compile_model(str(out / f"{name}.xml"), "CPU")(x[:1].numpy())[0]
    rep = {"max_abs_logit_diff": float(np.abs(ref - got).max()),
           "argmax_agreement": float((ref.argmax(1) == got.argmax(1)).mean()), "precision": "fp16" if fp16 else "fp32",
           "tile": tile, "meta": meta,
           "note": "ต้นแบบ — ไม่ได้ผ่านการ quantize INT8 (ต้องวัดผลกระทบต่อความแม่นยำก่อน)"}
    (out / f"{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    onnx_path.unlink(missing_ok=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="models/ink_segmentation")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    print(json.dumps(export(a.ckpt, a.out, fp16=not a.fp32), ensure_ascii=False, indent=1))

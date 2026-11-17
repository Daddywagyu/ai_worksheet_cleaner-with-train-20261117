"""Evaluate on the held-out TEST worksheets and compare with the Phase-1 colour pipeline.
   python -m training.evaluate --data DS/manifest.json --ckpt runs/x/best.pt --out runs/x/eval.json
OCR text-preservation rate is NOT measured (no OCR engine bundled)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ai.tiling import predict_probs
from config import Settings
from core import mask_generator, restoration_engine
from training.classes import CLASS_NAMES, INK_CLASSES, NUM_CLASSES, TRAIN_DPI
from training.dataset import load_manifest
from training.metrics import binary_scores, confusion, per_class, psnr, ssim
from training.model import UNetMobileNetV3


def torch_infer(model, device):
    def f(x):
        with torch.no_grad():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()
    return f


def evaluate(manifest, ckpt, split="test", max_images=0, tile=256, restore=True, device="cpu"):
    root, m = load_manifest(manifest)
    entries = [e for e in m["entries"] if e["split"] == split]
    if max_images:
        entries = entries[:max_images]
    model = UNetMobileNetV3(pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"]); model.eval().to(device)
    infer = torch_infer(model, device)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    ink_ai, ink_cls, rows = [], [], []
    s = Settings(mode="balanced", process_dpi=TRAIN_DPI)
    for e in entries:
        dirty = cv2.imread(str(root / e["dirty"])); target = cv2.imread(str(root / e["target"]))
        gt = cv2.imread(str(root / e["labels"]), cv2.IMREAD_GRAYSCALE)
        probs = predict_probs(dirty, infer, tile=tile, stride=tile * 3 // 4)
        pred = probs.argmax(0).astype(np.uint8)
        cm += confusion(pred, gt, NUM_CLASSES)
        gt_ink = np.isin(gt, INK_CLASSES); ai_ink = np.isin(pred, INK_CLASSES)
        cls_ink = mask_generator.generate_masks(dirty, s).final > 0
        ink_ai.append(binary_scores(ai_ink, gt_ink)); ink_cls.append(binary_scores(cls_ink, gt_ink))
        if restore:
            reg = cv2.dilate(gt_ink.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            row = {"dirty_vs_target": [psnr(dirty, target), ssim(dirty, target)]}
            for name, mk in (("ai", ai_ink), ("classical", cls_ink)):
                mask = cv2.dilate(mk.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
                rest = restoration_engine.restore(dirty, mask, s).image
                row[name] = [psnr(rest, target), ssim(rest, target), psnr(rest[reg], target[reg])]
            row["dirty_region_psnr"] = psnr(dirty[reg], target[reg])
            rows.append(row)
    pc = per_class(cm)
    mean = lambda L, k: float(np.mean([r[k] for r in L]))
    rep = {"split": split, "n_images": len(entries), "data_source": m.get("source"),
           "per_class": {CLASS_NAMES[c]: v for c, v in pc.items()},
           "ink_binary_ai": {k: mean(ink_ai, k) for k in ink_ai[0]},
           "ink_binary_classical_phase1": {k: mean(ink_cls, k) for k in ink_cls[0]},
           "ocr_text_preservation": "not measured (no OCR engine bundled)"}
    if rows:
        rep["restoration"] = {
            "full_page_psnr_ssim": {n: [float(np.mean([r[n][0] for r in rows])), float(np.mean([r[n][1] for r in rows]))] for n in ("ai", "classical")} |
                                   {"no_processing": [float(np.mean([r["dirty_vs_target"][0] for r in rows])), float(np.mean([r["dirty_vs_target"][1] for r in rows]))]},
            "stroke_region_psnr_db": {"ai": float(np.mean([r["ai"][2] for r in rows])), "classical": float(np.mean([r["classical"][2] for r in rows])),
                                      "no_processing": float(np.mean([r["dirty_region_psnr"] for r in rows]))}}
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--max-images", type=int, default=0); ap.add_argument("--no-restore", action="store_true")
    a = ap.parse_args()
    r = evaluate(a.data, a.ckpt, a.split, a.max_images, restore=not a.no_restore)
    Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1)); print(json.dumps(r, ensure_ascii=False, indent=1))

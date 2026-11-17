"""Pencil evaluation through the REAL runtime path (OpenVINO model + protection mask), compared with the
conservative classical fallback.  Reports: pencil precision/recall/F1/IoU, false positives on CLEAN pages
(gray printed content must survive), share of pencil pixels that the protection mask makes un-removable,
and restoration PSNR in the stroke region.     OCR preservation is NOT measured (no OCR engine bundled)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ai.pencil_detector import PencilDetector
from config import Settings
from core import mask_generator as mg, restoration_engine
from training.classes import PENCIL, TRAIN_DPI
from training.manifest import load_manifest
from training.metrics import binary_scores, psnr


def evaluate_pencil(manifest, split="test", max_images=0, mode="balanced", restore_n=6, model_path=None, device="CPU"):
    root, m = load_manifest(manifest)
    entries = [e for e in m["entries"] if e["split"] == split]
    entries = entries[:max_images] if max_images else entries
    det = PencilDetector(device, model_path)
    s = Settings(mode=mode, detect_pencil=True, detect_red=False, detect_blue=False, process_dpi=TRAIN_DPI)
    out = {"ai": [], "classical": [], "ai_override": []}
    fp = {"ai": [], "classical": [], "ai_override": []}
    damage = {"ai": [], "ai_override": []}                     # printed pixels wrongly included in the removal mask
    so = Settings(**{**s.to_dict(), "pencil_override_print": True})
    unremovable, rest = [], []
    for n, e in enumerate(entries):
        dirty = cv2.imread(str(root / e["dirty"])); clean = cv2.imread(str(root / e["target"]))
        gt = cv2.imread(str(root / e["labels"]), cv2.IMREAD_GRAYSCALE) == PENCIL
        res = {}
        lab = cv2.imread(str(root / e["labels"]), cv2.IMREAD_GRAYSCALE)
        for name, d, st in (("ai", det, s), ("classical", None, s), ("ai_override", det, so)):
            ms = mg.generate_masks(dirty, st, None, d)
            res[name] = ms
            out[name].append(binary_scores(ms.final > 0, gt))
            fp[name].append(int((mg.generate_masks(clean, st, None, d).final > 0).sum()))
            if name in damage:
                damage[name].append(int(((ms.final > 0) & (lab == 1)).sum()))
        unremovable.append(float((gt & (res["ai"].protection > 0)).sum() / max(1, gt.sum())))
        if n < restore_n and gt.any():
            reg = cv2.dilate(gt.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            row = {"none": psnr(dirty[reg], clean[reg])}
            for name in ("ai", "classical", "ai_override"):
                r = restoration_engine.restore(dirty, res[name].final, s).image
                row[name] = psnr(r[reg], clean[reg])
            rest.append(row)
    avg = lambda L: {k: float(np.mean([x[k] for x in L])) for k in L[0]}
    rep = {"split": split, "n_images": len(entries), "mode": mode, "data_source": m.get("source"),
           "pencil_ai": avg(out["ai"]), "pencil_classical_fallback": avg(out["classical"]), "pencil_ai_with_print_override": avg(out["ai_override"]),
           "printed_pixels_wrongly_in_removal_mask_per_page": {k: float(np.mean(v)) for k, v in damage.items()},
           "clean_page_false_positive_px_per_page": {k: float(np.mean(v)) for k, v in fp.items()},
           "clean_pages_with_any_false_positive": {k: float(np.mean([x > 0 for x in v])) for k, v in fp.items()},
           "gt_pencil_px_inside_protection_unremovable_share": float(np.mean(unremovable)),
           "restoration_stroke_region_psnr_db": {k: float(np.mean([r[k] for r in rest])) for k in ("none", "ai", "classical", "ai_override")} if rest else None,
           "ocr_text_preservation": "not measured (no OCR engine bundled)"}
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--mode", default="balanced")
    a = ap.parse_args()
    r = evaluate_pencil(a.data, a.split, a.max_images, a.mode)
    Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1)); print(json.dumps(r, ensure_ascii=False, indent=1))

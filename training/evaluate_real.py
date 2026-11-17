"""Evaluate the CURRENT pipelines on real paired data (weak labels from training/real_pdf_pairs.py).
Counts are accumulated per method so several runs (page ranges / sets) can be merged.
Strict = pixel-exact; tolerant = within 2 px (the registration of the two scans is not perfect)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ai.ink_detector import InkDetector
from ai.pencil_detector import PencilDetector
from config import Settings
from core import mask_generator as mg
from training.classes import BLUE, PENCIL, RED
from training.manifest import load_manifest

K = np.ones((5, 5), np.uint8)          # +-2 px tolerance


def tally(pred, gt):
    pred, gt = pred > 0, gt > 0
    gd, pd_ = cv2.dilate(gt.astype(np.uint8), K) > 0, cv2.dilate(pred.astype(np.uint8), K) > 0
    return {"tp": int((pred & gt).sum()), "fp": int((pred & ~gt).sum()), "fn": int((~pred & gt).sum()),
            "tp_pred_tol": int((pred & gd).sum()), "pred": int(pred.sum()), "gt": int(gt.sum()), "gt_tol_hit": int((gt & pd_).sum())}


def run(manifest, pages, modes=("conservative", "balanced"), dpi=200, device="CPU"):
    root, m = load_manifest(manifest)
    entries = [e for e in m["entries"]][pages[0]:pages[1]]
    ink, pen = InkDetector(device), PencilDetector(device)
    acc: dict = {}

    def add(name, t):
        a = acc.setdefault(name, {})
        for k, v in t.items():
            a[k] = a.get(k, 0) + v

    for e in entries:
        dirty = cv2.imread(str(root / e["dirty"])); lab = cv2.imread(str(root / e["labels"]), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(root / e["target"]))
        g_pen, g_pencil = (lab == RED) | (lab == BLUE), lab == PENCIL
        g_all = g_pen | g_pencil
        printed_dark = (cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) < 100) & (lab == 1)
        for mode in modes:
            s = Settings(mode=mode, process_dpi=dpi)
            sp = Settings(mode=mode, process_dpi=dpi, detect_pencil=True, detect_red=False, detect_blue=False)
            sa = Settings(mode=mode, process_dpi=dpi, ai_mode="ai_only", detect_handwriting=True)
            runs = {
                "pen_classical": mg.generate_masks(dirty, s),
                "pencil_classical": mg.generate_masks(dirty, sp),
                "pen_ai_ink_model": mg.generate_masks(dirty, sa, ink),
                "pencil_ai_model": mg.generate_masks(dirty, sp, None, pen),
            }
            for name, ms in runs.items():
                f = ms.final
                tgt = g_pencil if "pencil" in name else g_pen
                add(f"{mode}/{name}/vs_own_target", tally(f, tgt))
                add(f"{mode}/{name}/vs_all_ink", tally(f, g_all))
                add(f"{mode}/{name}/printed_dark_removed", {"hit": int(((f > 0) & printed_dark).sum()), "total": int(printed_dark.sum())})
            combo = cv2.bitwise_or(runs["pen_classical"].final, runs["pencil_ai_model"].final)
            add(f"{mode}/combo_classicalpen+aipencil/vs_all_ink", tally(combo, g_all))
            combo2 = cv2.bitwise_or(runs["pen_classical"].final, runs["pencil_classical"].final)
            add(f"{mode}/combo_classicalpen+classicalpencil/vs_all_ink", tally(combo2, g_all))
    return {"pages": [pages[0], pages[1]], "n": len(entries), "acc": acc}


def summarize(acc):
    out = {}
    for k, a in acc.items():
        if "tp" in a:
            p = a["tp"] / max(1, a["tp"] + a["fp"]); r = a["tp"] / max(1, a["tp"] + a["fn"])
            out[k] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
                      "iou": round(a["tp"] / max(1, a["tp"] + a["fp"] + a["fn"]), 3),
                      "precision_tol2px": round(a["tp_pred_tol"] / max(1, a["pred"]), 3),
                      "recall_tol2px": round(a["gt_tol_hit"] / max(1, a["gt"]), 3), "gt_px": a["gt"], "pred_px": a["pred"]}
        else:
            out[k] = {"printed_dark_px_removed": a["hit"], "of_total": a["total"], "share": round(a["hit"] / max(1, a["total"]), 5)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pages", default="0:24"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.pages.split(":"))
    r = run(a.data, (lo, hi)); Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(summarize(r["acc"]), indent=1)[:3000])

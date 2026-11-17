"""Held-out evaluation on real paired data (full 200-DPI pages) for ONE OpenVINO model plus the classical
colour/pencil pipeline.  Counts accumulate so page ranges can be run separately and merged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ai.ink_detector import InkDetector
from config import Settings
from core import color_detector, mask_generator as mg, protection_engine, stroke_analyzer
from training.classes import BLUE, PENCIL, RED
from training.evaluate_real import tally
from training.manifest import load_manifest

THR = (0.5, 0.7, 0.85)


def run(manifest, model_xml, splits=("test",), sets=None, pages=(0, 10**9), dpi=200):
    root, m = load_manifest(manifest)
    ents = [e for e in m["entries"] if e["split"] in splits and (sets is None or e["set"] in sets)][pages[0]:pages[1]]
    det = InkDetector("CPU", model_xml)
    s = Settings(mode="balanced", process_dpi=dpi)
    acc, per_set = {}, {}

    def add(name, t, sname):
        for d in (acc, per_set.setdefault(sname, {})):
            a = d.setdefault(name, {})
            for k, v in t.items():
                a[k] = a.get(k, 0) + v

    for e in ents:
        dirty = cv2.imread(str(root / e["dirty"])); lab = cv2.imread(str(root / e["labels"]), 0)
        gt = {"red": lab == RED, "blue": lab == BLUE, "pencil": lab == PENCIL}
        prot = protection_engine.build_protection(dirty, dpi, s.preservation).mask
        keep = cv2.bitwise_not(prot)
        probs = det._probs(dirty, dpi)
        h, w = dirty.shape[:2]
        L, _, _, chroma, _ = color_detector.lab_chroma(dirty)
        step = max(1, int(np.sqrt(L.size / 300_000))); paper = float(np.percentile(L[::step, ::step], 85))
        dark = (paper - L.astype(np.float32)) > 8
        pc = {c: cv2.resize(probs[i], (w, h), interpolation=cv2.INTER_LINEAR) for c, i in (("red", RED), ("blue", BLUE), ("pencil", PENCIL))}
        for thr in THR:
            for name in ("red", "blue", "pencil"):
                mk = ((pc[name] >= thr) & dark).astype(np.uint8) * 255
                if name != "pencil":
                    mk &= ((chroma > 6) | True).astype(np.uint8) * 255
                mk = cv2.bitwise_and(stroke_analyzer.expand(mk, s), keep)
                add(f"ai/{name}/thr{thr}", tally(mk, gt[name]), e["set"])
        # classical baselines
        for sens in (0.5, 0.75, 0.9):
            sc = Settings(mode="balanced", process_dpi=dpi, sens_red=sens, sens_blue=sens)
            f = mg.generate_masks(dirty, sc).final
            add(f"classical_colour/sens{sens}/red+blue", tally(f, gt["red"] | gt["blue"]), e["set"])
        fp = mg.generate_masks(dirty, Settings(mode="balanced", process_dpi=dpi, detect_pencil=True, detect_red=False, detect_blue=False)).final
        add("classical_pencil/balanced", tally(fp, gt["pencil"]), e["set"])
        # damage to printed dark pixels by the AI pencil@0.7 mask
        clean = cv2.imread(str(root / e["target"]), 0)
        pd = (clean < 100) & (lab == 1)
        mk = cv2.bitwise_and(stroke_analyzer.expand(((pc["pencil"] >= 0.7) & dark).astype(np.uint8) * 255, s), keep)
        for d in (acc, per_set.setdefault(e["set"], {})):
            a = d.setdefault("printed_dark_removed_by_ai_pencil0.7", {"hit": 0, "total": 0})
            a["hit"] += int(((mk > 0) & pd).sum()); a["total"] += int(pd.sum())
    return {"n": len(ents), "acc": acc, "per_set": per_set}


def summarize(acc):
    out = {}
    for k, a in acc.items():
        if "tp" in a:
            p = a["tp"] / max(1, a["tp"] + a["fp"]); r = a["tp"] / max(1, a["tp"] + a["fn"])
            out[k] = {"recall": round(r, 3), "precision": round(p, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
                      "recall_tol2px": round(a["gt_tol_hit"] / max(1, a["gt"]), 3), "precision_tol2px": round(a["tp_pred_tol"] / max(1, a["pred"]), 3), "gt_px": a["gt"]}
        else:
            out[k] = {"share": round(a["hit"] / max(1, a["total"]), 5), "px": a["hit"], "of": a["total"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=""); ap.add_argument("--pages", default="0:1000")
    a = ap.parse_args()
    lo, hi = map(int, a.pages.split(":"))
    r = run(a.data, a.model, sets=a.sets.split(",") if a.sets else None, pages=(lo, hi))
    Path(a.out).write_text(json.dumps(r, indent=1)); print("pages", r["n"])

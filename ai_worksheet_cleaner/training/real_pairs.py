"""Build a manifest from REAL paired scans: <dir>/dirty/NAME.* and <dir>/clean/NAME.*.
Labels are derived automatically from |dirty-clean| (weak labels; review them before trusting)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from training.classes import BACKGROUND, BLUE, OTHER, PRINTED, RED
from training.synthetic_data import split_worksheets


def labels_from_pair(dirty: np.ndarray, clean: np.ndarray, thr: int = 40) -> np.ndarray:
    if dirty.shape != clean.shape:
        clean = cv2.resize(clean, (dirty.shape[1], dirty.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(dirty.astype(np.int16) - clean.astype(np.int16)).max(axis=2) > thr
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    hsv = cv2.cvtColor(dirty, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    lab = np.where(clean.min(axis=2) < 200, PRINTED, BACKGROUND).astype(np.uint8)
    lab[diff & (s > 60) & ((h < 12) | (h > 165))] = RED
    lab[diff & (s > 60) & (h >= 90) & (h <= 135)] = BLUE
    lab[diff & ~np.isin(lab, (RED, BLUE))] = OTHER
    return lab


def build_manifest(root, out_dir, seed=0):
    root, out = Path(root), Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (root / "dirty").iterdir() if p.is_file())
    split = split_worksheets(range(len(names)), seed=seed)
    entries = []
    for i, n in enumerate(names):
        d = next((root / "dirty").glob(n + ".*")); c = next((root / "clean").glob(n + ".*"))
        di, ci = cv2.imread(str(d)), cv2.imread(str(c))
        lab = labels_from_pair(di, ci)
        for k, arr in (("dirty", di), ("target", cv2.resize(ci, (di.shape[1], di.shape[0]))), ("labels", lab)):
            cv2.imwrite(str(out / "images" / f"{n}_{k}.png"), arr)
        entries.append({"worksheet": i, "split": split[i], "dirty": f"images/{n}_dirty.png",
                        "target": f"images/{n}_target.png", "labels": f"images/{n}_labels.png"})
    (out / "manifest.json").write_text(json.dumps({"source": "real_pairs", "entries": entries}, indent=1))
    return out / "manifest.json"

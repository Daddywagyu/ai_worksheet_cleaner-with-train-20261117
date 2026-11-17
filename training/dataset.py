"""Datasets.  Splits are by ORIGINAL WORKSHEET id (manifest 'split' field)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.classes import ALL_INK_CLASSES as INK_CLASSES


from training.manifest import load_manifest, worksheets_in  # noqa: F401  (re-exported)


class WorksheetDataset(Dataset):
    def __init__(self, manifest_path, split="train", crop=192, augment=True, crops_per_image=4, seed=0):
        self.root, m = load_manifest(manifest_path)
        self.entries = [e for e in m["entries"] if e["split"] == split]
        if not self.entries:
            raise ValueError(f"ไม่มีข้อมูลในชุด '{split}'")
        self.crop, self.augment, self.cpi, self.seed = crop, augment, crops_per_image, seed

    def __len__(self):
        return len(self.entries) * self.cpi

    def _read(self, rel, flag):
        img = cv2.imread(str(self.root / rel), flag)
        if img is None:
            raise FileNotFoundError(rel)
        return img

    def __getitem__(self, i):
        e = self.entries[i // self.cpi]
        rng = np.random.default_rng([self.seed, i, int(torch.initial_seed() % 2 ** 31)]) if self.augment else np.random.default_rng([self.seed, i])
        img = self._read(e["dirty"], cv2.IMREAD_COLOR)
        lab = self._read(e["labels"], cv2.IMREAD_GRAYSCALE)
        h, w = lab.shape
        c = self.crop
        if rng.random() < 0.7 and np.isin(lab, INK_CLASSES).any():      # bias crops toward handwriting
            ys, xs = np.where(np.isin(lab, INK_CLASSES))
            j = rng.integers(0, len(ys))
            y0 = int(np.clip(ys[j] - rng.integers(8, c - 8), 0, h - c)); x0 = int(np.clip(xs[j] - rng.integers(8, c - 8), 0, w - c))
        else:
            y0, x0 = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
        img, lab = img[y0:y0 + c, x0:x0 + c], lab[y0:y0 + c, x0:x0 + c]
        if self.augment:
            if rng.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), float(rng.uniform(0.3, 1.0)))
            if rng.random() < 0.4:                                      # scan JPEG artefacts
                ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 95))])
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12), 0, 255)
        x = torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)).float() / 255.0
        return x, torch.from_numpy(np.ascontiguousarray(lab)).long()

"""Phase 2: synthetic data + worksheet-level split (no torch needed)."""
import json

import cv2
import numpy as np

from training import synthetic_data as sd
from training.classes import BLUE, OTHER, RED
from training.synthetic_data import generate_dataset, make_sample, split_worksheets


def test_sample_is_deterministic():
    a, b = make_sample(3, 0, n_strokes=6), make_sample(3, 0, n_strokes=6)
    assert np.array_equal(a.dirty, b.dirty) and np.array_equal(a.labels, b.labels)
    assert not np.array_equal(a.dirty, make_sample(3, 1, n_strokes=6).dirty)     # variants differ


def test_labels_valid_and_paired_target_is_clean_outside_strokes():
    s = make_sample(4, 0, n_strokes=8)
    assert set(np.unique(s.labels)) <= {0, 1, 2, 3, 5}
    ink = np.isin(s.labels, (RED, BLUE, OTHER))
    assert ink.sum() > 200
    far = cv2.dilate(ink.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    diff = np.abs(s.dirty.astype(int) - s.target.astype(int)).max(axis=2)
    assert diff[far].max() == 0                                   # target == dirty away from handwriting
    assert diff[ink].mean() > 20                                  # and differs where handwriting is


def test_class_colours_are_consistent():
    for seed in range(6):
        s = make_sample(seed, 0, n_strokes=10)
        for cls, test in ((RED, lambda p: p[:, 2] > p[:, 0]), (BLUE, lambda p: p[:, 0] > p[:, 2])):
            px = s.dirty[s.labels == cls].astype(int)
            if len(px) > 100:
                assert test(px).mean() > 0.7, (seed, cls)


def test_split_is_by_worksheet_and_disjoint(tmp_path):
    sp = split_worksheets(range(20), seed=3)
    assert set(sp.values()) == {"train", "val", "test"} and len(sp) == 20
    assert split_worksheets(range(20), seed=3) == sp               # stable
    man = generate_dataset(tmp_path / "ds", n_worksheets=7, versions=2, seed=5)
    entries = json.loads(man.read_text())["entries"]
    by_ws = {}
    for e in entries:
        by_ws.setdefault(e["worksheet"], set()).add(e["split"])
    assert all(len(v) == 1 for v in by_ws.values())                # all versions of a worksheet stay together
    splits = {s for v in by_ws.values() for s in v}
    assert splits == {"train", "val", "test"} and len(entries) == 14
    assert (tmp_path / "ds" / entries[0]["dirty"]).is_file()


def test_real_pairs_importer(tmp_path):
    from training.real_pairs import build_manifest, labels_from_pair
    s = make_sample(9, 0, n_strokes=6)
    for d in ("dirty", "clean"):
        (tmp_path / "raw" / d).mkdir(parents=True)
    for i in range(6):
        cv2.imwrite(str(tmp_path / "raw" / "dirty" / f"p{i}.png"), s.dirty)
        cv2.imwrite(str(tmp_path / "raw" / "clean" / f"p{i}.png"), s.target)
    lab = labels_from_pair(s.dirty, s.target)
    gt_ink = np.isin(s.labels, (RED, BLUE, OTHER))
    assert (np.isin(lab, (RED, BLUE, OTHER)) & gt_ink).sum() / gt_ink.sum() > 0.7
    m = build_manifest(tmp_path / "raw", tmp_path / "out")
    assert len(json.loads(m.read_text())["entries"]) == 6

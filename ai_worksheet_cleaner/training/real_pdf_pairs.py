"""Real paired data from two PDFs (same pages): CLEAN scan + WRITTEN scan.

Steps per page: render -> register clean onto written (ORB+RANSAC affine, ECC refine) -> detect ADDED ink
(darker than anything the clean page could show nearby) -> classify by colour: red / blue / pencil(achromatic).
Labels are WEAK (derived automatically): handwriting over black print cannot be separated by differencing.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from core import color_detector
from training.classes import BACKGROUND, BLUE, PENCIL, PRINTED, RED


def render_pages(pdf: str | Path, dpi: int = 200) -> list[np.ndarray]:
    doc = pymupdf.open(str(pdf))
    out = []
    for p in doc:
        pix = p.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False, colorspace=pymupdf.csRGB)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        out.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return out


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def register(clean: np.ndarray, dirty: np.ndarray):
    """Warp `clean` into the frame of `dirty`.  Returns (warped, info)."""
    h, w = dirty.shape[:2]
    c = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA) if clean.shape[:2] != (h, w) else clean
    g1, g2 = cv2.GaussianBlur(_gray(c), (0, 0), 1.2), cv2.GaussianBlur(_gray(dirty), (0, 0), 1.2)
    orb = cv2.ORB_create(6000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    M = None
    if d1 is not None and d2 is not None:
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
        m = sorted(m, key=lambda x: x.distance)[:1500]
        if len(m) >= 12:
            src = np.float32([k1[x.queryIdx].pt for x in m]); dst = np.float32([k2[x.trainIdx].pt for x in m])
            M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    # ECC refinement on a half-size image (robust to the added handwriting: it is sparse)
    try:
        s = 0.5
        a = cv2.resize(g2, None, fx=s, fy=s).astype(np.float32); b = cv2.resize(g1, None, fx=s, fy=s).astype(np.float32)
        W = M.astype(np.float32).copy(); W[:, 2] *= s
        # cv2.findTransformECC maps template(a) coords -> input(b) coords; invert our clean->dirty matrix
        Winv = cv2.invertAffineTransform(W)
        _, Winv = cv2.findTransformECC(a, b, Winv, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        W = cv2.invertAffineTransform(Winv); W[:, 2] /= s; M = W
    except cv2.error:
        pass
    warped = cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, {"matrix": M.tolist()}


def alignment_score(warped: np.ndarray, dirty: np.ndarray) -> float:
    """Share of dark clean-print pixels that are also dark (within 2 px) in the written scan."""
    gc, gd = _gray(warped), _gray(dirty)
    dark = gc < 90
    near = cv2.erode(gd, np.ones((5, 5), np.uint8)) < 140           # darkest value within 2 px
    return float((dark & near).sum() / max(1, dark.sum()))


def derive_labels(dirty: np.ndarray, warped_clean: np.ndarray, dpi: int = 200):
    """Returns labels (uint8) and a dict of diagnostics."""
    k = dpi / 200.0
    # match scanner gain/offset: map dirty paper level to clean paper level
    gd, gc = _gray(dirty).astype(np.float32), _gray(warped_clean).astype(np.float32)
    pd_, pc = np.percentile(gd, 80), np.percentile(gc, 80)
    gain = pc / max(pd_, 1.0)
    gd = np.clip(gd * gain, 0, 255)
    r = max(2, int(round(2 * k)))
    clean_darkest_near = cv2.erode(gc, np.ones((2 * r + 1, 2 * r + 1), np.uint8))      # darkest clean pixel nearby
    added = (clean_darkest_near - gd) > 28                                              # dirty darker than any clean pixel nearby
    added = cv2.morphologyEx(added.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(added, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= 25 * k * k                                                  # drop scanner dust
    added = keep[lab].astype(np.uint8)

    _, a, b, chroma, hue = color_detector.lab_chroma(dirty)
    labels = np.where(gc < 200, PRINTED, BACKGROUND).astype(np.uint8)
    is_red = ((hue <= 55) | (hue >= 320)) & (chroma > 9)
    is_blue = (hue >= 200) & (hue <= 315) & (chroma > 9)
    ink = added > 0
    labels[ink] = PENCIL                                                                # achromatic / gray default
    labels[ink & is_red] = RED
    labels[ink & is_blue] = BLUE
    # soften: colour pixels adjacent to a stroke of the same colour (anti-aliased edges)
    for cls, m in ((RED, is_red), (BLUE, is_blue)):
        halo = cv2.dilate((labels == cls).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        labels[halo & ink & m] = cls
    return labels, {"gain": float(gain), "added_px": int(ink.sum())}


def build_dataset(clean_pdf, dirty_pdf, out_dir, dpi=200, test_frac=0.2, val_frac=0.1, seed=0, min_align=0.80, set_name="s1"):
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True)
    C, D = render_pages(clean_pdf, dpi), render_pages(dirty_pdf, dpi)
    if len(C) != len(D):
        raise ValueError(f"จำนวนหน้าไม่เท่ากัน: clean={len(C)} dirty={len(D)}")
    ids = list(range(len(D)))
    from training.synthetic_data import split_worksheets
    split = split_worksheets(ids, val=val_frac, test=test_frac, seed=seed)
    entries, report = [], []
    for i, (c, d) in enumerate(zip(C, D)):
        warped, info = register(c, d)
        score = alignment_score(warped, d)
        labels, diag = derive_labels(d, warped, dpi)
        rec = {"page": i, "alignment_score": round(score, 3), **diag,
               "red_px": int((labels == RED).sum()), "blue_px": int((labels == BLUE).sum()), "pencil_px": int((labels == PENCIL).sum())}
        report.append(rec)
        if score < min_align:
            rec["excluded"] = "alignment"
            continue
        stem = f"{set_name}_p{i:03d}"
        for k, arr in (("dirty", d), ("target", warped), ("labels", labels)):
            cv2.imwrite(str(out / "images" / f"{stem}_{k}.png"), arr)
        entries.append({"worksheet": f"{set_name}_{i}", "split": split[i], "dirty": f"images/{stem}_dirty.png",
                        "target": f"images/{stem}_target.png", "labels": f"images/{stem}_labels.png"})
    (out / "manifest.json").write_text(json.dumps({"source": "real_pdf_pairs", "dpi": dpi, "weak_labels": True, "entries": entries}, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return out / "manifest.json", report


# ---------------------------------------------------------------- combine several sets
def _down_labels(lab: np.ndarray, f: float) -> np.ndarray:
    """Downscale labels without losing thin strokes (area vote, ink classes win ties)."""
    h, w = lab.shape
    nh, nw = int(round(h * f)), int(round(w * f))
    out = np.zeros((nh, nw), np.uint8)
    out[cv2.resize((lab == PRINTED).astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) >= 0.4] = PRINTED
    best = np.zeros((nh, nw), np.float32)
    for cls in (RED, BLUE, PENCIL):
        m = cv2.resize((lab == cls).astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
        sel = (m >= 0.2) & (m > best)
        out[sel] = cls
        best = np.maximum(best, np.where(sel, m, 0))
    return out


def merge_sets(set_dirs: dict, split_by_set: dict, out_dir, train_dpi=100, src_dpi=200):
    """set_dirs {name: dir with manifest.json}; split_by_set {name: train|val|test}.
    Writes manifest_200.json (evaluation, full resolution) and manifest_100.json (training, downscaled)."""
    out = Path(out_dir); (out / "images100").mkdir(parents=True, exist_ok=True)
    f = train_dpi / src_dpi
    e200, e100 = [], []
    for name, d in set_dirs.items():
        d = Path(d); m = json.loads((d / "manifest.json").read_text())
        for e in m["entries"]:
            stem = Path(e["dirty"]).name.replace("_dirty.png", "")
            rel = lambda p: str(Path(__import__("os").path.relpath(d / p, out)))
            ent = {"worksheet": f"{name}_{e['worksheet']}", "set": name, "split": split_by_set[name],
                   "dirty": rel(e["dirty"]), "target": rel(e["target"]), "labels": rel(e["labels"])}
            e200.append(ent)
            di, ci, li = (cv2.imread(str(d / e[k]), flag) for k, flag in (("dirty", 1), ("target", 1), ("labels", 0)))
            sm = lambda im: cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            for k, arr in (("dirty", sm(di)), ("target", sm(ci)), ("labels", _down_labels(li, f))):
                cv2.imwrite(str(out / "images100" / f"{stem}_{k}.png"), arr)
            e100.append({**ent, **{k: f"images100/{stem}_{k}.png" for k in ("dirty", "target", "labels")}})
    (out / "manifest_200.json").write_text(json.dumps({"source": "real_pdf_pairs", "dpi": src_dpi, "weak_labels": True, "entries": e200}, indent=1))
    (out / "manifest_100.json").write_text(json.dumps({"source": "real_pdf_pairs", "dpi": train_dpi, "weak_labels": True, "entries": e100}, indent=1))
    return out / "manifest_200.json", out / "manifest_100.json"

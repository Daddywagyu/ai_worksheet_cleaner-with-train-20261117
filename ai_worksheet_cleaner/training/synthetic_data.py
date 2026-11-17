"""Synthetic children's worksheets with pixel-exact labels and paired clean targets.

NOTE: everything here is SYNTHETIC (rendered fonts, procedural cartoons, procedural
"handwriting").  It exists to make the training pipeline runnable and testable; it is not
a substitute for real scanned worksheets.  Thai text is not generated (no Thai font bundled).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from training.classes import BACKGROUND, BLUE, OTHER, PENCIL, PRINTED, RED, TRAIN_DPI

WORDS = ("apple ball cat dog egg fish goat hat ink jar kite lion moon nest owl pig queen rain sun tree "
         "umbrella van water box yak zebra one two three four five six seven eight nine ten add subtract "
         "count circle match trace write read draw color big small happy sad red blue green yellow").split()
FONTS = ("helv", "tiro", "cour", "hebo", "tibo")


# ------------------------------------------------------------------ clean worksheet
def _cartoon(page, rng, x, y, s):
    pal = [(0.85, 0.1, 0.1), (0.1, 0.3, 0.85), (0.95, 0.75, 0.1), (0.2, 0.65, 0.25), (0.6, 0.3, 0.7), (0.95, 0.5, 0.15)]
    kind = rng.integers(0, 4)
    c = pal[rng.integers(0, len(pal))]
    dark = tuple(max(0.0, v - 0.25) for v in c)
    if kind == 0:                                           # round character with eyes
        page.draw_oval(pymupdf.Rect(x, y, x + s, y + s), color=dark, fill=c)
        for ex in (0.3, 0.7):
            page.draw_circle((x + ex * s, y + 0.38 * s), 0.09 * s, color=(1, 1, 1), fill=(1, 1, 1))
            page.draw_circle((x + ex * s, y + 0.38 * s), 0.035 * s, color=(0, 0, 0), fill=(0, 0, 0))
        page.draw_line((x + 0.3 * s, y + 0.7 * s), (x + 0.7 * s, y + 0.7 * s), color=(0, 0, 0), width=1.5)
    elif kind == 1:                                         # house
        page.draw_rect(pymupdf.Rect(x, y + 0.4 * s, x + s, y + s), color=dark, fill=c)
        page.draw_polyline([(x - 0.1 * s, y + 0.4 * s), (x + 0.5 * s, y), (x + 1.1 * s, y + 0.4 * s), (x - 0.1 * s, y + 0.4 * s)],
                           color=dark, fill=pal[(rng.integers(0, len(pal)))], closePath=True)
    elif kind == 2:                                         # star-ish triangle
        page.draw_polyline([(x, y + s), (x + s / 2, y), (x + s, y + s), (x, y + s)], color=dark, fill=c, closePath=True)
    else:                                                   # flower
        for a in range(6):
            cx = x + s / 2 + 0.3 * s * np.cos(a * np.pi / 3)
            cy = y + s / 2 + 0.3 * s * np.sin(a * np.pi / 3)
            page.draw_circle((cx, cy), 0.2 * s, color=dark, fill=c)
        page.draw_circle((x + s / 2, y + s / 2), 0.15 * s, color=(0.9, 0.7, 0.1), fill=(0.95, 0.8, 0.2))


def build_clean_pdf_bytes(rng: np.random.Generator) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    font = FONTS[rng.integers(0, len(FONTS))]
    page.insert_text((50, 55), " ".join(rng.choice(WORDS, 3)).title() + " Worksheet", fontsize=int(rng.integers(16, 24)), fontname="hebo")
    y = 95.0
    while y < 770:
        sec = rng.choice(["text", "math", "trace", "table", "cartoons", "choices"], p=[.25, .2, .15, .12, .18, .1])
        if sec == "text":
            for _ in range(int(rng.integers(2, 5))):
                page.insert_text((50, y), " ".join(rng.choice(WORDS, int(rng.integers(5, 10)))), fontsize=int(rng.integers(10, 14)), fontname=font)
                y += 22
        elif sec == "math":
            for i in range(int(rng.integers(2, 4))):
                a, b = rng.integers(1, 20, 2)
                page.insert_text((50, y), f"{i + 1}) {a} {rng.choice(['+', '-'])} {b} = ______", fontsize=13, fontname=font)
                y += 26
        elif sec == "trace":
            page.insert_text((50, y), "Trace:", fontsize=10, fontname=font)
            y += 14
            for _ in range(int(rng.integers(1, 3))):
                x = 50.0
                step = float(rng.integers(5, 9))
                while x < 50 + float(rng.integers(250, 480)):
                    page.draw_circle((x, y), 1.1, color=(0.4, 0.4, 0.4), fill=(0.4, 0.4, 0.4))
                    x += step
                y += 26
        elif sec == "table":
            rows, cols, cw = int(rng.integers(2, 5)), int(rng.integers(2, 5)), float(rng.integers(60, 110))
            for r in range(rows + 1):
                page.draw_line((50, y + r * 24), (50 + cols * cw, y + r * 24), color=(0, 0, 0), width=1)
            for c in range(cols + 1):
                page.draw_line((50 + c * cw, y), (50 + c * cw, y + rows * 24), color=(0, 0, 0), width=1)
            y += rows * 24 + 14
        elif sec == "cartoons":
            s = float(rng.integers(50, 90))
            for k in range(int(rng.integers(1, 4))):
                _cartoon(page, rng, 60 + k * (s + 40), y, s)
            y += s + 22
        else:
            page.insert_text((50, y), f"Circle the answer:   A. {rng.choice(WORDS)}    B. {rng.choice(WORDS)}    C. {rng.choice(WORDS)}",
                             fontsize=11, fontname=font)
            y += 28
    return doc.tobytes()


def render_pdf_bytes(data: bytes, dpi: int) -> np.ndarray:
    doc = pymupdf.open(stream=data, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False, colorspace=pymupdf.csRGB)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def add_gray_print(page_bytes: bytes, rng: np.random.Generator) -> bytes:
    """Hard negatives for pencil detection: PRINTED light/mid-gray content (gray text, gray grid,
    gray illustration fills, dashed lines).  Always labelled 'printed'."""
    doc = pymupdf.open(stream=page_bytes, filetype="pdf")
    pg = doc[0]
    for _ in range(int(rng.integers(2, 6))):
        kind = rng.choice(["text", "grid", "fill", "dash"])
        x, y = float(rng.integers(40, 400)), float(rng.integers(100, 760))
        g = float(rng.uniform(0.3, 0.7))
        if kind == "text":
            pg.insert_text((x, y), " ".join(rng.choice(WORDS, 4)), fontsize=int(rng.integers(9, 14)), color=(g, g, g), fontname="helv")
        elif kind == "grid":
            n = int(rng.integers(3, 7))
            for i in range(n):
                pg.draw_line((x, y + i * 14), (x + 150, y + i * 14), color=(g, g, g), width=0.8)
        elif kind == "fill":
            pg.draw_rect(pymupdf.Rect(x, y, x + float(rng.integers(30, 90)), y + float(rng.integers(20, 60))), color=(g, g, g),
                         fill=(min(1, g + 0.2),) * 3)
        else:
            pg.draw_line((x, y), (x + 200, y), color=(g, g, g), width=1.2, dashes="[3 3] 0")
    out = doc.tobytes()
    doc.close()
    return out


def _grain(shape, rng, scale=1.0):
    """Paper tooth: graphite deposits unevenly -> high-frequency multiplicative texture."""
    h, w = shape[:2]
    f = rng.normal(0, 1, (h, w)).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), 0.7 * scale)
    f = (f - f.mean()) / (f.std() + 1e-6)
    return np.clip(0.62 + 0.38 * f, 0.0, 1.0)


def _pencil_alpha(shape, rng, paths, thickness, pressure, grain):
    base = _stroke_alpha(shape, paths, thickness, rng)
    h, w = shape[:2]
    pr = cv2.resize(rng.uniform(0.6, 1.0, (3, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)   # pressure drift
    return np.clip(base * pressure * pr * grain, 0, 1)


def add_pencil(img: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n: int, printed: np.ndarray):
    """Graphite strokes (class PENCIL).  img is float32 BGR modified in place."""
    h, w = labels.shape
    s = w / 827.0
    ys, xs = np.where(printed > 0)
    grain = _grain(labels.shape, rng)
    for _ in range(n):
        if len(xs) and rng.random() < 0.5:
            j = rng.integers(0, len(xs)); cx, cy = float(xs[j]), float(ys[j])
        else:
            cx, cy = rng.uniform(40 * s, w - 40 * s), rng.uniform(40 * s, h - 40 * s)
        bw, bh = rng.uniform(30, 170) * s, rng.uniform(14, 60) * s
        kind = rng.choice(["circle", "check", "cross", "underline", "arrow", "scribble", "text", "hatch"])
        th = rng.uniform(1.4, 3.6) * s
        pressure = rng.uniform(0.3, 0.9)
        if kind == "text":
            a = _text_alpha(labels.shape, rng, cx, cy, rng.uniform(0.9, 2.0) * s, th * 0.7) * pressure * grain
        elif kind == "hatch":
            paths = [_densify([(cx - bw / 2 + i * bw / 9, cy - bh / 2), (cx - bw / 2 + i * bw / 9 + bh / 3, cy + bh / 2)], 4) for i in range(10)]
            a = _pencil_alpha(labels.shape, rng, paths, th, pressure, grain)
        else:
            a = _pencil_alpha(labels.shape, rng, _paths(rng, kind, cx, cy, bw, bh), th, pressure, grain)
        col = float(rng.uniform(45, 95))
        tint = np.array([col * 1.04, col, col * 0.97], np.float32)            # graphite: slightly cool, achromatic
        for c in range(3):
            img[..., c] = img[..., c] * (1 - a) + tint[c] * a
        labels[a > 0.14] = PENCIL


# ------------------------------------------------------------------ handwriting
def _smooth_noise(rng, n, amp):
    k = max(3, n // 8)
    base = rng.normal(0, 1, k)
    return np.interp(np.linspace(0, k - 1, n), np.arange(k), base) * amp


def _densify(pts, per_seg=12):
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, per_seg, endpoint=False):
            out.append((a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t))
    out.append(pts[-1])
    return np.array(out, np.float32)


def _paths(rng, kind, cx, cy, w, h):
    if kind == "circle":
        t = np.linspace(0, 2 * np.pi * rng.uniform(1.0, 1.15), 60)
        r = 1 + _smooth_noise(rng, 60, 0.06)
        return [np.stack([cx + w / 2 * r * np.cos(t), cy + h / 2 * r * np.sin(t)], 1)]
    if kind == "check":
        return [_densify([(cx - w / 3, cy), (cx - w / 8, cy + h / 2), (cx + w / 2, cy - h / 2)])]
    if kind == "cross":
        return [_densify([(cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2)]), _densify([(cx + w / 2, cy - h / 2), (cx - w / 2, cy + h / 2)])]
    if kind == "underline":
        return [_densify([(cx - w / 2, cy), (cx + w / 2, cy + rng.uniform(-6, 6))])]
    if kind == "arrow":
        tip = (cx + w / 2, cy)
        return [_densify([(cx - w / 2, cy), tip]), _densify([(tip[0] - h / 3, cy - h / 3), tip, (tip[0] - h / 3, cy + h / 3)])]
    if kind == "scribble":
        pts = [(cx - w / 2 + i * w / 8, cy + (h / 2 if i % 2 else -h / 2) * rng.uniform(0.6, 1)) for i in range(9)]
        return [_densify(pts, 6)]
    return []                                                # text handled separately


def _stroke_alpha(shape, paths, thickness, rng):
    m = np.zeros(shape[:2], np.uint8)
    for p in paths:
        p = p + np.stack([_smooth_noise(rng, len(p), 1.5), _smooth_noise(rng, len(p), 1.5)], 1)
        cv2.polylines(m, [p.astype(np.int32)], False, 255, max(1, int(thickness)), cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 0.6).astype(np.float32) / 255.0


def _text_alpha(shape, rng, x, y, size, thickness):
    m = np.zeros(shape[:2], np.uint8)
    txt = "".join(rng.choice(list("0123456789abcdexyz"), int(rng.integers(1, 4))))
    font = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX][rng.integers(0, 3)]
    cv2.putText(m, txt, (int(x), int(y)), font, size, 255, max(1, int(thickness)), cv2.LINE_AA)
    ang = rng.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((float(x), float(y)), float(ang), 1.0)
    return cv2.GaussianBlur(cv2.warpAffine(m, M, (shape[1], shape[0])), (0, 0), 0.6).astype(np.float32) / 255.0


def _ink(rng, cls):
    if cls == RED:
        h, s, v = rng.choice([rng.integers(0, 8), rng.integers(172, 180)]), rng.integers(150, 256), rng.integers(120, 256)
    elif cls == BLUE:
        h, s, v = rng.integers(100, 126), rng.integers(120, 256), rng.integers(90, 231)
    else:                                                   # black / gray / green / purple pens
        h, s, v = int(rng.choice([0, 60, 70, 140])), int(rng.integers(0, 120)), int(rng.integers(20, 110))
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def add_handwriting(clean: np.ndarray, printed: np.ndarray, rng: np.random.Generator, n_strokes=None, pencil_strokes: int = 0):
    """Returns (dirty, labels) where labels in {0:bg,1:printed,2:red,3:blue,4:pencil,5:other}."""
    h, w = clean.shape[:2]
    s = w / 827.0                                           # geometry authored for 100-DPI A4
    img = clean.astype(np.float32).copy()
    labels = np.where(printed > 0, PRINTED, BACKGROUND).astype(np.uint8)
    ys, xs = np.where(printed > 0)
    n = int(n_strokes if n_strokes is not None else rng.integers(3, 13))
    for _ in range(n):
        cls = int(rng.choice([RED, BLUE, OTHER], p=[0.4, 0.4, 0.2]))
        if len(xs) and rng.random() < 0.55:                  # often on top of printed content
            j = rng.integers(0, len(xs)); cx, cy = float(xs[j]), float(ys[j])
        else:
            cx, cy = rng.uniform(40 * s, w - 40 * s), rng.uniform(40 * s, h - 40 * s)
        bw, bh = rng.uniform(30, 160) * s, rng.uniform(14, 60) * s
        kind = rng.choice(["circle", "check", "cross", "underline", "arrow", "scribble", "text"])
        th = rng.uniform(1.6, 5.0) * s if rng.random() < 0.8 else rng.uniform(6, 11) * s   # pen vs marker
        if kind == "text":
            a = _text_alpha(clean.shape, rng, cx, cy, rng.uniform(0.9, 2.0) * s, th * 0.8)
        else:
            a = _stroke_alpha(clean.shape, _paths(rng, kind, cx, cy, bw, bh), th, rng)
        strength = rng.uniform(0.8, 1.0)
        col = np.array(_ink(rng, cls), np.float32)
        if th > 6 * s and rng.random() < 0.7:                # translucent marker
            for c in range(3):
                img[..., c] *= (1 - a * 0.9) + (col[c] / 255.0) * a * 0.9
        else:
            for c in range(3):
                img[..., c] = img[..., c] * (1 - a * strength) + col[c] * a * strength
        labels[a > 0.35] = cls
    if pencil_strokes:
        add_pencil(img, labels, np.random.default_rng([int(rng.integers(0, 2 ** 31)), 4]), pencil_strokes, printed)
    return np.clip(img, 0, 255).astype(np.uint8), labels


# ------------------------------------------------------------------ scan simulation
def photometric_params(rng):
    return {"tint": rng.uniform(0.9, 1.0, 3).astype(np.float32), "shade_amp": float(rng.uniform(0, 0.12)),
            "shade_seed": int(rng.integers(0, 2 ** 31)), "noise": float(rng.uniform(0.3, 2.5)),
            "noise_seed": int(rng.integers(0, 2 ** 31)), "gamma": float(rng.uniform(0.9, 1.15)),
            "blur": float(rng.choice([0, 0, 0.5, 0.8]))}


def apply_photometric(img: np.ndarray, p: dict) -> np.ndarray:
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255.0
    x = np.power(x, p["gamma"]) * p["tint"]
    small = np.random.default_rng(p["shade_seed"]).normal(0, 1, (4, 4)).astype(np.float32)
    shade = 1 + p["shade_amp"] * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    x = x * shade[..., None]
    if p["blur"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), p["blur"])
    x = x * 255.0 + np.random.default_rng(p["noise_seed"]).normal(0, p["noise"], x.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


@dataclass
class Sample:
    dirty: np.ndarray
    target: np.ndarray
    labels: np.ndarray


def make_clean(ws_seed: int, dpi: int = TRAIN_DPI, hard_gray: bool = False):
    rng = np.random.default_rng(ws_seed)
    pdf = build_clean_pdf_bytes(rng)
    if hard_gray:
        pdf = add_gray_print(pdf, np.random.default_rng([ws_seed, 99]))
    clean = render_pdf_bytes(pdf, dpi)
    printed = (clean.min(axis=2) < 235).astype(np.uint8) * 255
    return clean, printed


def make_sample(ws_seed: int, variant: int, dpi: int = TRAIN_DPI, n_strokes=None, pencil: bool = False) -> Sample:
    """pencil=True: adds graphite strokes and gray printed hard-negatives (Phase 3 data)."""
    clean, printed = make_clean(ws_seed, dpi, hard_gray=pencil)
    rng = np.random.default_rng([ws_seed, variant, 7])
    n_pen = n_strokes if n_strokes is not None else (int(rng.integers(0, 6)) if pencil else None)
    n_pencil = int(np.random.default_rng([ws_seed, variant, 13]).integers(2, 9)) if pencil else 0
    dirty, labels = add_handwriting(clean, printed, rng, n_pen, n_pencil)
    p = photometric_params(np.random.default_rng([ws_seed, variant, 11]))
    return Sample(apply_photometric(dirty, p), apply_photometric(clean, p), labels)


# ------------------------------------------------------------------ dataset on disk + split
def split_worksheets(ids, val=0.15, test=0.15, seed=0):
    """Split by ORIGINAL WORKSHEET id (never by patch/variant) -> no leakage."""
    ids = sorted(set(ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_val, n_test = max(1, round(len(ids) * val)), max(1, round(len(ids) * test))
    sp = {}
    for rank, i in enumerate(order):
        sp[ids[i]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return sp


def generate_dataset(out_dir, n_worksheets=60, versions=2, seed=0, dpi=TRAIN_DPI, pencil: bool = False) -> Path:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ids = list(range(n_worksheets))
    split = split_worksheets(ids, seed=seed)
    entries = []
    for ws in ids:
        base = seed * 100_003 + ws
        for v in range(versions):
            smp = make_sample(base, v, dpi, pencil=pencil)
            stem = f"ws{ws:04d}_v{v}"
            cv2.imwrite(str(out / "images" / f"{stem}_dirty.png"), smp.dirty)
            cv2.imwrite(str(out / "images" / f"{stem}_target.png"), smp.target)
            cv2.imwrite(str(out / "images" / f"{stem}_labels.png"), smp.labels)
            entries.append({"worksheet": ws, "split": split[ws], "dirty": f"images/{stem}_dirty.png",
                            "target": f"images/{stem}_target.png", "labels": f"images/{stem}_labels.png"})
    (out / "manifest.json").write_text(json.dumps({"source": "synthetic", "dpi": dpi, "seed": seed, "pencil": pencil, "entries": entries}, indent=1))
    return out / "manifest.json"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--versions", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pencil", action="store_true", help="Phase 3: add graphite strokes + gray printed hard negatives")
    a = ap.parse_args()
    print(generate_dataset(a.out, a.n, a.versions, a.seed, pencil=a.pencil))

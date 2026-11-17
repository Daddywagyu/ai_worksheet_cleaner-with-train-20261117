"""Synthetic children's worksheet: text, math, tracing dots, table, cartoon art + hand marks."""
from __future__ import annotations

import cv2
import numpy as np
import pymupdf

from tests import synthetic as syn


def build_worksheet_pdf(path, pages: int = 1):
    doc = pymupdf.open()
    for p in range(pages):
        pg = doc.new_page(width=595, height=842)
        pg.insert_text((50, 60), f"Worksheet {p + 1}: Count and Trace", fontsize=20, fontname="hebo")
        for i, y in enumerate((110, 140, 170)):
            pg.insert_text((50, y), f"{i + 1}) {3 + i} + {4 + i} = ____      Circle the right answer:  A. {7 + 2 * i}  B. {8 + 2 * i}",
                           fontsize=11)
        # dotted tracing guides
        for y in (250, 290):
            x = 50.0
            while x < 400:
                pg.draw_circle((x, y), 1.1, color=(0.45, 0.45, 0.45), fill=(0.45, 0.45, 0.45))
                x += 6
        pg.insert_text((50, 238), "Trace the line:", fontsize=10)
        # table
        for r in range(4):
            pg.draw_line((50, 330 + r * 24), (350, 330 + r * 24), color=(0, 0, 0), width=1)
        for c in range(4):
            pg.draw_line((50 + c * 100, 330), (50 + c * 100, 402), color=(0, 0, 0), width=1)
        # printed cartoon art (blue character, red apple) -- must never be removed
        pg.draw_oval(pymupdf.Rect(400, 120, 520, 240), color=(0, 0, 0.7), fill=(0.1, 0.3, 0.85))
        pg.draw_circle((435, 165), 9, color=(1, 1, 1), fill=(1, 1, 1))
        pg.draw_circle((485, 165), 9, color=(1, 1, 1), fill=(1, 1, 1))
        pg.draw_circle((435, 165), 3, color=(0, 0, 0), fill=(0, 0, 0))
        pg.draw_circle((485, 165), 3, color=(0, 0, 0), fill=(0, 0, 0))
        pg.draw_oval(pymupdf.Rect(410, 440, 500, 530), color=(0.6, 0, 0), fill=(0.85, 0.1, 0.1))
        pg.draw_line((455, 440), (462, 420), color=(0.1, 0.5, 0.1), width=3)
    doc.save(str(path))
    doc.close()
    return path


def add_marks(clean: np.ndarray):
    """Red circle + check, blue strike-through & scribble.  Returns (dirty, pen_gt)."""
    h, w = clean.shape[:2]
    s = w / 1654.0
    dirty = clean.astype(np.float32).copy()
    gt = np.zeros((h, w), np.uint8)

    def draw(col, pts, th, closed=False):
        m = np.zeros((h, w), np.uint8)
        cv2.polylines(m, [np.array([(int(x * s), int(y * s)) for x, y in pts], np.int32)], closed,
                      255, max(1, int(th * s)), cv2.LINE_AA)
        a = m.astype(np.float32) / 255.0
        for c in range(3):
            dirty[..., c] = dirty[..., c] * (1 - a) + col[c] * a
        return (a > 0.5).astype(np.uint8) * 255

    ell = [(int(700 + 90 * np.cos(t)), int(318 + 28 * np.sin(t))) for t in np.linspace(0, 6.4, 40)]
    gt |= draw((35, 35, 205), ell, 8)                                          # red circle (answer)
    gt |= draw((40, 40, 210), [(880, 720), (920, 770), (1010, 650)], 9)      # red check mark, blank area
    gt |= draw((195, 70, 28), [(140, 380), (900, 395)], 9)                     # blue strike-through
    gt |= draw((200, 75, 30), [(200, 1250), (500, 1290), (260, 1330), (560, 1370)], 9)  # blue scribble
    dirty += np.random.default_rng(5).normal(0, 1.0, dirty.shape).astype(np.float32)
    return np.clip(dirty, 0, 255).astype(np.uint8), gt


def make_pair(tmp_dir):
    pdf = build_worksheet_pdf(tmp_dir / "ws.pdf")
    clean = syn.render(pdf, 0, 200)
    dirty, gt = add_marks(clean)
    return pdf, clean, dirty, gt

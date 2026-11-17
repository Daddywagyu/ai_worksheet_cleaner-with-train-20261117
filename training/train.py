"""Train the ink segmentation U-Net.   python -m training.train --data DS/manifest.json --out runs/x"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.classes import ALL_INK_CLASSES, CLASS_NAMES, IGNORE_INDEX, INK_CLASSES, NUM_CLASSES, TRAIN_DPI
from training.dataset import WorksheetDataset, worksheets_in
from training.metrics import confusion, per_class
from training.model import UNetMobileNetV3, count_params

CLASS_WEIGHTS = [1.0, 1.5, 4.0, 4.0, 4.0, 4.0, 0.0]     # background, printed, red, blue, pencil, other, (uncertain: runtime only)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def dice_loss(logits, target, classes=ALL_INK_CLASSES, eps=1.0):
    p = F.softmax(logits, 1)
    loss, n = 0.0, 0
    for c in classes:
        t = (target == c).float(); pc = p[:, c]
        if t.sum() == 0:
            continue
        loss = loss + 1 - (2 * (pc * t).sum() + eps) / (pc.sum() + t.sum() + eps); n += 1
    return loss / max(n, 1)


def loss_fn(logits, target, weight):
    return F.cross_entropy(logits, target, weight=weight, ignore_index=IGNORE_INDEX) + 0.5 * dice_loss(logits, target)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        cm += confusion(pred, y.numpy(), NUM_CLASSES)
    pc = per_class(cm)
    ink_iou = [pc[c]["iou"] for c in ALL_INK_CLASSES if c in pc]
    return float(np.mean(ink_iou)) if ink_iou else 0.0, pc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=192); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="auto"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop early after N minutes (0 = no limit)")
    ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet init (needs Internet at training time)")
    ap.add_argument("--crops-per-image", type=int, default=6)
    ap.add_argument("--epochs-per-run", type=int, default=0, help="stop after N epochs in this invocation (use with --resume)")
    ap.add_argument("--init", default="", help="initialise weights from a checkpoint (fine-tuning)")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/last.pt (model, optimizer, schedule, log)")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    tr = WorksheetDataset(a.data, "train", a.crop, True, a.crops_per_image, a.seed)
    va = WorksheetDataset(a.data, "val", a.crop, False, 4, a.seed)
    assert not (worksheets_in(a.data, "train") & worksheets_in(a.data, "val")), "data leakage between train/val"
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=0)
    model = UNetMobileNetV3(pretrained=a.pretrained).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"]); print("init from", a.init, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.15)
    w = torch.tensor(CLASS_WEIGHTS, device=dev)
    t0, step, best, log, start_ep = time.time(), 0, -1.0, [], 0
    if a.resume and (out / "last.pt").is_file():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best, log, start_ep = ck["step"], ck["best"], ck["log"], ck["epoch"]
        print(f"resumed at epoch {start_ep}, step {step}", flush=True)
    print(f"device={dev} params={count_params(model):,} train_samples={len(tr)} val_samples={len(va)}", flush=True)
    stop = False
    for ep in range(start_ep, a.epochs):
        model.train(); run = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y, w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += float(loss.detach()); step += 1
            if (a.max_steps and step >= a.max_steps) or (a.max_minutes and (time.time() - t0) / 60 > a.max_minutes):
                stop = True; break
        miou, pc = validate(model, vl, dev)
        rec = {"epoch": ep + 1, "step": step, "train_loss": run / max(1, len(tl)), "val_ink_miou": miou,
               "minutes": round((time.time() - t0) / 60, 2)}
        log.append(rec); print(json.dumps(rec), flush=True)
        meta = {"classes": CLASS_NAMES, "train_dpi": TRAIN_DPI, "tile": a.crop, "trained_on": "synthetic worksheets"
                if json.loads(Path(a.data).read_text()).get("source") == "synthetic" else "see manifest",
                "seed": a.seed, "steps": step, "val_ink_miou": miou}
        if miou > best:
            best = miou; torch.save({"model": model.state_dict(), "meta": meta}, out / "best.pt")
        torch.save({"model": model.state_dict(), "meta": meta, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "log": log, "epoch": ep + 1 if not stop else ep}, out / "last.pt")
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
        if stop or (a.epochs_per_run and ep + 1 - start_ep >= a.epochs_per_run):
            break
    print("best val ink mIoU:", round(best, 4), flush=True)


if __name__ == "__main__":
    main()

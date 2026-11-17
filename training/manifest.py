"""Torch-free manifest helpers (so evaluation/runtime code does not need PyTorch)."""
import json
from pathlib import Path


def load_manifest(path):
    path = Path(path)
    return path.parent, json.loads(path.read_text())


def worksheets_in(manifest_path, split):
    _, m = load_manifest(manifest_path)
    return {e["worksheet"] for e in m["entries"] if e["split"] == split}

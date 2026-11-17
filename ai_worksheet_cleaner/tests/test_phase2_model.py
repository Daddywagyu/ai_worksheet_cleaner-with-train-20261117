"""Phase 2: model, training loop, metrics (needs torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training import train as T
from training.metrics import binary_scores, confusion, per_class, psnr, ssim
from training.model import UNetMobileNetV3, count_params


def test_model_output_shape_and_size():
    m = UNetMobileNetV3().eval()
    with torch.no_grad():
        y = m(torch.rand(2, 3, 256, 256))
    assert y.shape == (2, 7, 256, 256) and torch.isfinite(y).all()
    assert count_params(m) < 3_000_000


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = UNetMobileNetV3()
    x = torch.rand(2, 3, 96, 96)
    y = torch.randint(0, 4, (2, 96, 96))
    y[:, 20:60, 20:60] = 2
    w = torch.tensor(T.CLASS_WEIGHTS)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    first = None
    for i in range(25):
        loss = T.loss_fn(m(x), y, w)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or float(loss)
    assert float(loss) < 0.7 * first


def test_training_cli_smoke_and_no_leakage(tmp_path):
    from training.dataset import worksheets_in
    from training.synthetic_data import generate_dataset
    man = generate_dataset(tmp_path / "ds", n_worksheets=7, versions=1, seed=2)
    assert not (worksheets_in(man, "train") & worksheets_in(man, "val"))
    assert not (worksheets_in(man, "train") & worksheets_in(man, "test"))
    T.main(["--data", str(man), "--out", str(tmp_path / "run"), "--epochs", "1", "--max-steps", "2",
            "--batch", "2", "--crop", "128", "--crops-per-image", "2"])
    ck = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    assert "model" in ck and ck["meta"]["trained_on"].startswith("synthetic")


def test_confusion_and_scores():
    gt = np.array([[0, 0, 2, 2], [3, 3, 5, 255]])
    pr = np.array([[0, 2, 2, 2], [3, 0, 5, 0]])
    cm = confusion(pr, gt, 7)
    assert cm.sum() == 7                                           # ignore label excluded
    pc = per_class(cm)
    assert pc[2]["precision"] == pytest.approx(2 / 3) and pc[2]["recall"] == 1.0
    assert pc[3]["recall"] == 0.5 and pc[5]["iou"] == 1.0
    b = binary_scores(np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 1], bool))
    assert b["precision"] == 0.5 and b["recall"] == 0.5 and b["iou"] == pytest.approx(1 / 3)


def test_psnr_ssim():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert psnr(a, a) == float("inf") and ssim(a, a) == pytest.approx(1.0)
    b = np.clip(a.astype(int) + 10, 0, 255).astype(np.uint8)
    assert psnr(a, b) == pytest.approx(20 * np.log10(255 / 10), abs=0.3)
    assert ssim(a, b) < 1.0

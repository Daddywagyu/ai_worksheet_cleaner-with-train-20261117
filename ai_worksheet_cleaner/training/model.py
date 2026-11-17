"""U-Net with a MobileNetV3-Small encoder (torchvision).  Only ops that OpenVINO/NPU handle
well: conv, hard-swish, bilinear resize, concat.  No pretrained weights are bundled; pass
pretrained=True on a machine with Internet access at TRAINING time to start from ImageNet."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from training.classes import NUM_CLASSES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNetMobileNetV3(nn.Module):
    # feature taps of mobilenet_v3_small: (index, channels, stride)
    TAPS = ((0, 16, 2), (1, 16, 4), (3, 24, 8), (8, 48, 16), (12, 576, 32))

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False, dec=(96, 64, 48, 32)):
        super().__init__()
        enc = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = enc.features[:13]
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        c = [t[1] for t in self.TAPS]                       # 16,16,24,48,576
        self.d4 = _block(c[4] + c[3], dec[0])               # stride 16
        self.d3 = _block(dec[0] + c[2], dec[1])             # stride 8
        self.d2 = _block(dec[1] + c[1], dec[2])             # stride 4
        self.d1 = _block(dec[2] + c[0], dec[3])             # stride 2
        self.head = nn.Conv2d(dec[3] + 3, num_classes, 1)   # full-resolution skip of the input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float RGB in [0,1], shape (B,3,H,W) with H,W multiples of 32.  Returns logits (B,C,H,W)."""
        xin = x
        x = (x - self.mean) / self.std
        taps = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (0, 1, 3, 8, 12):
                taps[i] = x
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        y = self.d4(torch.cat([up(taps[12], taps[8]), taps[8]], 1))
        y = self.d3(torch.cat([up(y, taps[3]), taps[3]], 1))
        y = self.d2(torch.cat([up(y, taps[1]), taps[1]], 1))
        y = self.d1(torch.cat([up(y, taps[0]), taps[0]], 1))
        y = torch.cat([up(y, xin), xin], 1)
        return self.head(y)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

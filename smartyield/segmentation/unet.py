"""A compact U-Net for binary vegetation / background segmentation.

U-Net (Ronneberger et al., 2015) is an encoder-decoder CNN. The encoder
repeatedly halves the resolution to learn *what* is in the image (leaf vs soil
vs water). The decoder upsamples back to full resolution. Skip connections
copy fine detail from the encoder into the decoder, so the mask knows *where*
each leaf edge is. The channel widths are kept small (~1.1 M parameters at base width 12) so
training and inference run on a CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, widths=(16, 32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.downs.append(DoubleConv(c, w))
            c = w
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.up_convs.append(DoubleConv(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, conv, s in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([s, x], 1))
        return self.head(x)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()

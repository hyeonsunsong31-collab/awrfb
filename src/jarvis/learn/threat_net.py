"""Threat predictor network (PyTorch): P(our HP or guard drops within 100/200/300/500 ms) from the
opponent's last frames.

Per frame: MobileNetV3-small (ImageNet weights when available) -> 128-d embedding. Over time: a small
temporal convolution over the window of embeddings -> one logit per horizon. For deployment the network is
exported as a *streaming* graph: each frame runs the encoder once and the head reuses the previous
embeddings, so live cost is one MobileNet pass per frame instead of eight.
"""

from __future__ import annotations

import torch
from torch import nn

HORIZONS = (100, 200, 300, 500)
# ImageNet statistics in BGR order: crops come straight from OpenCV frames.
_MEAN = (0.406 * 255, 0.456 * 255, 0.485 * 255)
_STD = (0.225 * 255, 0.224 * 255, 0.229 * 255)


class FrameEncoder(nn.Module):
    def __init__(self, dim: int = 128, backbone: str = "mobilenet_v3_small", pretrained: bool = True) -> None:
        super().__init__()
        if backbone == "tiny":  # tests and quick experiments
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            feat = 64
        else:
            import torchvision

            weights = None
            if pretrained:
                try:
                    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                except AttributeError:
                    weights = None
            try:
                m = torchvision.models.mobilenet_v3_small(weights=weights)
            except Exception:  # no internet for the weights: train from scratch
                m = torchvision.models.mobilenet_v3_small(weights=None)
            self.net = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
            feat = 576
        self.proj = nn.Linear(feat, dim)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, H, W, 3) uint8 BGR -> (N, dim)."""
        x = (x.permute(0, 3, 1, 2).float() - self.mean) / self.std
        return self.proj(torch.flatten(self.net(x), 1))


class TemporalHead(nn.Module):
    def __init__(self, dim: int = 128, horizons: int = len(HORIZONS)) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU(), nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU())
        self.out = nn.Linear(dim * 2, horizons)

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """embs: (B, T, dim) -> logits (B, H)."""
        h = self.conv(embs.transpose(1, 2))
        return self.out(torch.cat([h[:, :, -1], h.mean(-1)], 1))


class ThreatNet(nn.Module):
    def __init__(self, window: int = 8, dim: int = 128, backbone: str = "mobilenet_v3_small", pretrained: bool = True) -> None:
        super().__init__()
        self.window = window
        self.dim = dim
        self.encoder = FrameEncoder(dim, backbone, pretrained)
        self.head = TemporalHead(dim)

    def forward(self, frames: torch.Tensor, warmup: torch.Tensor | None = None) -> torch.Tensor:
        """frames: (B, T, H, W, 3) uint8. ``warmup`` (B,) zeroes that many leading embeddings, as live
        tracking does before T frames of the opponent have been seen."""
        b, t = frames.shape[:2]
        e = self.encoder(frames.reshape(b * t, *frames.shape[2:])).reshape(b, t, -1)
        if warmup is not None:
            idx = torch.arange(t, device=frames.device).view(1, t, 1)
            e = e * (idx >= warmup.view(b, 1, 1)).float()
        return self.head(e)


class StreamingThreat(nn.Module):
    """Export wrapper: (previous T-1 embeddings, newest frame) -> (newest embedding, logits)."""

    def __init__(self, net: ThreatNet) -> None:
        super().__init__()
        self.encoder = net.encoder
        self.head = net.head

    def forward(self, prev: torch.Tensor, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(frame)
        return emb, self.head(torch.cat([prev, emb.unsqueeze(1)], 1))

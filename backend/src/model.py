"""
DualStreamDetector — EfficientNet-B0 (RGB) + SRM (frequency) with gated attention fusion.
Copy of the training architecture used in exp6.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import numpy as np


# ── SRM Filter Bank ─────────────────────────────────────────────────────────

def get_srm_kernels() -> torch.Tensor:
    """Return the 3 canonical SRM high-pass filters as a (3,1,5,5) tensor."""
    f1 = np.array([
        [ 0,  0, 0,  0,  0],
        [ 0, -1, 2, -1,  0],
        [ 0,  2,-4,  2,  0],
        [ 0, -1, 2, -1,  0],
        [ 0,  0, 0,  0,  0],
    ], dtype=np.float32) / 4.0

    f2 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1],
    ], dtype=np.float32) / 12.0

    f3 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32) / 2.0

    kernels = np.stack([f1, f2, f3], axis=0)[:, np.newaxis]
    return torch.from_numpy(kernels)


class SRMConv(nn.Module):
    """Fixed SRM filter applied per RGB channel (groups=3)."""

    def __init__(self):
        super().__init__()
        base = get_srm_kernels()                      # (3,1,5,5)
        weight = base.repeat(3, 1, 1, 1)             # (9,1,5,5)  -- but we want (3,3,5,5)
        # Correct shape: out_channels=3, in_channels/group=1, groups=3 → weight (3,1,5,5)
        self.register_buffer("weight", base)          # (3,1,5,5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) → apply each filter to each channel independently
        # groups=3 means: out_channels must be divisible by groups
        # Use groups=1 and average over RGB channels instead
        # Replicate filter for each input channel
        w = self.weight.repeat(1, 3, 1, 1) / 3.0    # (3,3,5,5)
        return torch.nn.functional.conv2d(x, w, padding=2)


# ── Gated Attention Fusion ───────────────────────────────────────────────────

class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([rgb, freq], dim=1))
        return g * rgb + (1 - g) * freq


# ── SRM Stream ───────────────────────────────────────────────────────────────

class SRMStream(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.srm = SRMConv()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.srm(x)
        feat = self.net(noise).flatten(1)
        return self.proj(feat)


# ── Full Dual-Stream Detector ────────────────────────────────────────────────

class DualStreamDetector(nn.Module):
    def __init__(self, pretrained: bool = True, srm_dim: int = 128, drop: float = 0.3):
        super().__init__()

        # RGB stream — EfficientNet-B0
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eff = tvm.efficientnet_b0(weights=weights)
        rgb_dim = eff.classifier[1].in_features          # 1280
        eff.classifier = nn.Identity()
        self.rgb_stream = eff

        # Frequency stream — SRM + small CNN
        self.freq_stream = SRMStream(out_dim=srm_dim)

        # Fusion
        self.fusion = GatedFusion(dim=rgb_dim)
        # Project freq to rgb_dim so gating dimensions match
        self.freq_proj = nn.Linear(srm_dim, rgb_dim)

        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(rgb_dim, 256),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_feat  = self.rgb_stream(x)                        # (B, 1280)
        freq_feat = self.freq_proj(self.freq_stream(x))       # (B, 1280)
        fused     = self.fusion(rgb_feat, freq_feat)          # (B, 1280)
        return self.head(fused).squeeze(1)                    # (B,)

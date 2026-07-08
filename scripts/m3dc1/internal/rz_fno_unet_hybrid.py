"""Minimal FNO + U-Net hybrid for RZ field prediction (additive backend)."""
from __future__ import annotations

import torch
import torch.nn as nn

from surge.model.backends.fno2d import _FNO2dNet
from surge.model.backends.unet import _UNetNet


class FNOUNetHybrid(nn.Module):
    """Sum of FNO2d (global/spectral) and U-Net (local) branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        fno_modes: int = 64,
        fno_hidden: int = 32,
        unet_base: int = 48,
    ) -> None:
        super().__init__()
        self.fno = _FNO2dNet(in_channels, out_channels, hidden_channels=fno_hidden,
                             n_modes=fno_modes, n_layers=4)
        self.unet = _UNetNet(in_channels, out_channels, base_channels=unet_base, depth=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(x) + self.unet(x)

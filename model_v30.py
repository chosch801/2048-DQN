"""Compatibility network for evaluating the archived V3.0 checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AfterstateValueNetV30(nn.Module):
    """Recreate the scalar V3.0 afterstate CNN without training support."""

    def __init__(self, board_size: int = 4):
        super().__init__()
        if board_size != 4:
            raise ValueError("V3 currently supports only a 4x4 board")

        self.num_input_channels = 18
        self.features = nn.Sequential(
            nn.Conv2d(self.num_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def _encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim == 3:
            observation = observation.reshape(observation.shape[0], -1)
        if observation.ndim != 2 or observation.shape[1] != 16:
            raise ValueError(
                "observation must have shape (batch, 16), "
                f"got {tuple(observation.shape)}"
            )

        levels = torch.round(observation).long().clamp_(0, 16)
        one_hot = F.one_hot(levels, num_classes=17).to(dtype=torch.float32)
        one_hot = one_hot.permute(0, 2, 1).reshape(-1, 17, 4, 4)
        log_plane = (levels.to(dtype=torch.float32) / 16.0).reshape(-1, 1, 4, 4)
        return torch.cat((one_hot, log_plane), dim=1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self._encode(observation))).squeeze(-1)

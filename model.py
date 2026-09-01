"""CNN afterstate value network used by the V3 DQN agent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AfterstateValueNet(nn.Module):
    """Estimate the value of a deterministic afterstate.

    The network receives a flattened log2 board and returns one scalar per
    afterstate.  The agent evaluates the four deterministic action outcomes
    with shared weights and combines each value with that action's immediate
    merge reward.  This keeps the DQN target-network/TD machinery while
    removing the random tile spawn from the network input itself.
    """

    def __init__(self, board_size: int = 4):
        super().__init__()
        if board_size != 4:
            raise ValueError("V3 currently supports only a 4x4 board")

        # One-hot planes for exponents 0..16 plus a normalized exponent plane.
        # Exponent 0 represents an empty cell.  Exponent 16 covers 65536 so
        # that evaluation does not fail as soon as a strong policy gets there.
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
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        """Return a tensor of shape ``(batch,)`` with afterstate values."""

        encoded = self._encode(observation)
        return self.head(self.features(encoded)).squeeze(-1)


if __name__ == "__main__":
    network = AfterstateValueNet()
    dummy = torch.tensor(
        [
            [0, 0, 1, 0, 3, 2, 0, 0, 4, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    values = network(dummy)
    print(f"参数量: {sum(p.numel() for p in network.parameters()):,}")
    print(f"输入: {dummy.shape} -> 输出: {values.shape}")


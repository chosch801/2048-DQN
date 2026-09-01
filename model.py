"""Distributional CNN afterstate value network used by the V3 DQN agent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedNoisyLinear(nn.Module):
    """Factorized Gaussian noise for value-based exploration."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_mu = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
        self.reset_parameters(sigma_init)

    @staticmethod
    def _scale_noise(noise: torch.Tensor) -> torch.Tensor:
        return noise.sign() * noise.abs().sqrt()

    def reset_parameters(self, sigma_init: float) -> None:
        mu_range = 1.0 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / self.in_features**0.5)
        self.bias_sigma.data.fill_(sigma_init / self.out_features**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.linear(inputs, self.weight_mu, self.bias_mu)

        input_noise = self._scale_noise(
            torch.randn(self.in_features, device=inputs.device)
        )
        output_noise = self._scale_noise(
            torch.randn(self.out_features, device=inputs.device)
        )
        weight = self.weight_mu + self.weight_sigma * output_noise.outer(
            input_noise
        )
        bias = self.bias_mu + self.bias_sigma * output_noise
        return F.linear(inputs, weight, bias)


class ResidualBlock(nn.Module):
    """A small residual block that preserves the 4x4 board geometry."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = F.relu(self.conv1(inputs), inplace=True)
        outputs = self.conv2(outputs)
        return F.relu(outputs + residual, inplace=True)


class AfterstateValueNet(nn.Module):
    """Estimate a return distribution for a deterministic afterstate.

    The network receives a flattened log2 board and returns a distribution per
    afterstate.  The agent evaluates the four deterministic action outcomes
    with shared weights and combines each value with that action's immediate
    merge reward.  This keeps the DQN target-network/TD machinery while
    removing the random tile spawn from the network input itself.

    ``num_quantiles`` independent outputs retain the useful distributional
    DQN signal from V2 without bringing back its Transformer or action-query
    head.  The policy uses their mean for risk-neutral action selection.  A
    noisy quantile head supplies parameter-space exploration during training;
    ``eval()`` removes that noise for deterministic inference.
    """

    def __init__(self, board_size: int = 4, num_quantiles: int = 51):
        super().__init__()
        if board_size != 4:
            raise ValueError("V3 currently supports only a 4x4 board")
        if num_quantiles < 1:
            raise ValueError("num_quantiles must be at least 1")

        self.num_quantiles = int(num_quantiles)

        # One-hot planes for exponents 0..16 plus a normalized exponent plane
        # and explicit row/column coordinates.  The coordinate planes make
        # corner and edge preferences easier to learn than zero padding alone.
        # Exponent 0 represents an empty cell.  Exponent 16 covers 65536 so
        # that evaluation does not fail as soon as a strong policy gets there.
        self.num_input_channels = 20
        self.register_buffer(
            "coordinates",
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1.0, 1.0, 4),
                    torch.linspace(-1.0, 1.0, 4),
                    indexing="ij",
                )
            ).unsqueeze(0),
            persistent=False,
        )
        self.features = nn.Sequential(
            nn.Conv2d(self.num_input_channels, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(96),
            ResidualBlock(96),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.quantile_head = FactorizedNoisyLinear(256, self.num_quantiles)
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
        coordinates = self.coordinates.to(device=observation.device)
        coordinates = coordinates.expand(observation.shape[0], -1, -1, -1)
        return torch.cat((one_hot, log_plane, coordinates), dim=1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return quantile values with shape ``(batch, num_quantiles)``."""

        encoded = self._encode(observation)
        return self.quantile_head(self.head(self.features(encoded)))


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

import math

import torch
from torch import nn


class FGMagnitudeGate(nn.Module):
    """Two-layer FG gate; zero-init last layer makes ratio exp(tau * h) start at 1."""

    def __init__(self, n_fg: int, out_features: int, hidden_dim: int = 64):
        super(FGMagnitudeGate, self).__init__()
        self.layer1 = nn.Linear(n_fg, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, out_features)
        self.reset_fg_parameters()

    def reset_fg_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, fg_features: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(fg_features)))

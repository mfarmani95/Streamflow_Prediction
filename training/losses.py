"""Loss functions for streamflow regression."""

from __future__ import annotations

import torch
from torch import nn


def build_loss(name: str = "mse") -> nn.Module:
    normalized = name.lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss function: {name}")


def nse_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Nash-Sutcliffe loss, where lower is better."""
    numerator = torch.sum((target - prediction) ** 2)
    denominator = torch.sum((target - torch.mean(target)) ** 2) + eps
    return numerator / denominator

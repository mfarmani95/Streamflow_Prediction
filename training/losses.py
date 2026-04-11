"""Loss functions for streamflow regression."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """Abstract base class for loss functions in this project."""

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__()
        self.config = config

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss."""
        raise NotImplementedError


class MaskedMSELoss(BaseLoss):
    """Mean squared error loss that ignores NaN values in the target tensor."""

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        if not torch.any(mask):
            return predictions.sum() * 0.0

        squared_error = (predictions[mask] - targets[mask]) ** 2
        return torch.mean(squared_error)


class MaskedMAELoss(BaseLoss):
    """Mean absolute error loss that ignores NaN values in the target tensor."""

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        if not torch.any(mask):
            return predictions.sum() * 0.0

        absolute_error = torch.abs(predictions[mask] - targets[mask])
        return torch.mean(absolute_error)


class KGELoss(BaseLoss):
    """Kling-Gupta Efficiency loss.

    KGE is a higher-is-better metric with an ideal value of 1. This loss
    returns ``1 - KGE`` so minimizing the loss maximizes KGE.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)

    def _target_scalar(self, name: str) -> Optional[float]:
        if not isinstance(self.config, dict):
            return None
        value = self.config.get(name)
        if value is None and isinstance(self.config.get("scalers"), dict):
            value = self.config["scalers"].get(name)
        if value is None:
            return None
        return float(torch.as_tensor(value).reshape(-1)[0])

    def _maybe_original_units(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._target_scalar("target_mean")
        std = self._target_scalar("target_std")
        if mean is None or std is None:
            return predictions, targets

        mean_tensor = predictions.new_tensor(mean)
        std_tensor = predictions.new_tensor(std)
        return (
            predictions * (std_tensor + 1e-6) + mean_tensor,
            targets * (std_tensor + 1e-6) + mean_tensor,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions, targets = self._maybe_original_units(predictions, targets)
        mask = ~torch.isnan(targets)
        obs_flow = targets[mask]
        simulated_flow = predictions[mask]

        if obs_flow.numel() == 0:
            return predictions.sum() * 0.0

        eps = torch.finfo(obs_flow.dtype).eps

        mu_s = torch.mean(simulated_flow)
        mu_o = torch.mean(obs_flow)
        std_s = torch.std(simulated_flow, unbiased=False)
        std_o = torch.std(obs_flow, unbiased=False)

        r_num = torch.sum((simulated_flow - mu_s) * (obs_flow - mu_o))
        r_den = torch.sqrt(
            torch.sum((simulated_flow - mu_s) ** 2)
            * torch.sum((obs_flow - mu_o) ** 2)
        )
        r = r_num / (r_den + eps)

        alpha = (std_s + eps) / (std_o + eps)
        beta = (mu_s + eps) / (mu_o + eps)

        kge = 1.0 - torch.sqrt(
            (r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
        )
        return 1.0 - kge


def build_loss(name: str = "mse", config: Optional[Any] = None) -> nn.Module:
    normalized = name.lower()
    if normalized in {"mse", "masked_mse"}:
        return MaskedMSELoss(config)
    if normalized in {"mae", "masked_mae"}:
        return MaskedMAELoss(config)
    if normalized == "kge":
        return KGELoss(config)
    raise ValueError(f"Unsupported loss function: {name}")


def nse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable Nash-Sutcliffe loss, where lower is better."""
    mask = ~torch.isnan(target)
    prediction = prediction[mask]
    target = target[mask]
    if target.numel() == 0:
        return prediction.sum() * 0.0

    numerator = torch.sum((target - prediction) ** 2)
    denominator = torch.sum((target - torch.mean(target)) ** 2) + eps
    return numerator / denominator

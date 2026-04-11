"""Configuration defaults for the streamflow prediction project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


DEFAULT_DYNAMIC_INPUTS: Tuple[str, ...] = ("prcp", "tmax", "tmin", "srad", "vp")
DEFAULT_TARGET_VARIABLE = "qobs"


@dataclass
class DataConfig:
    seq_len: int = 30
    forecast_horizon: int = 1
    dynamic_inputs: Tuple[str, ...] = DEFAULT_DYNAMIC_INPUTS
    target_variable: str = DEFAULT_TARGET_VARIABLE
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass
class ModelConfig:
    model_name: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    checkpoint_path: str = "outputs/best_model.pt"

"""LSTM model for streamflow prediction."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LSTMStreamflowModel(nn.Module):
    """Predict streamflow from a meteorological sequence and optional static attributes."""

    def __init__(
        self,
        num_dynamic_features: int,
        num_static_features: int = 0,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        static_embedding_size: int = 16,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.num_static_features = num_static_features
        self.lstm = nn.LSTM(
            input_size=num_dynamic_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        if num_static_features > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(num_static_features, static_embedding_size),
                nn.ReLU(),
            )
            head_input_size = hidden_size + static_embedding_size
        else:
            self.static_encoder = None
            head_input_size = hidden_size

        self.head = nn.Sequential(
            nn.Linear(head_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        dynamic_sequence: torch.Tensor,
        static_attributes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, (hidden, _) = self.lstm(dynamic_sequence)
        features = hidden[-1]

        if self.static_encoder is not None:
            if static_attributes is None:
                raise ValueError("static_attributes are required for this model.")
            static_features = self.static_encoder(static_attributes)
            features = torch.cat([features, static_features], dim=-1)

        return self.head(features).squeeze(-1)

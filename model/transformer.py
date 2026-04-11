"""Transformer encoder model for streamflow prediction."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for daily sequences."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1], :]


class TransformerStreamflowModel(nn.Module):
    """Transformer encoder for streamflow prediction."""

    def __init__(
        self,
        num_dynamic_features: int,
        num_static_features: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        static_embedding_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_static_features = num_static_features
        self.input_projection = nn.Linear(num_dynamic_features, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if num_static_features > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(num_static_features, static_embedding_size),
                nn.ReLU(),
            )
            head_input_size = d_model + static_embedding_size
        else:
            self.static_encoder = None
            head_input_size = d_model

        self.head = nn.Sequential(
            nn.Linear(head_input_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        dynamic_sequence: torch.Tensor,
        static_attributes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.input_projection(dynamic_sequence)
        features = self.position(features)
        encoded = self.encoder(features)
        pooled = encoded[:, -1, :]

        if self.static_encoder is not None:
            if static_attributes is None:
                raise ValueError("static_attributes are required for this model.")
            static_features = self.static_encoder(static_attributes)
            pooled = torch.cat([pooled, static_features], dim=-1)

        return self.head(pooled).squeeze(-1)

"""PyTorch dataset for windowed streamflow sequence samples."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class StreamflowSequenceDataset(Dataset):
    """Container for prebuilt streamflow sequence samples."""

    def __init__(
        self,
        dynamic_sequences: torch.Tensor,
        targets: torch.Tensor,
        static_attributes: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if dynamic_sequences.ndim != 3:
            raise ValueError("dynamic_sequences must have shape (samples, seq_len, features).")
        if targets.ndim == 2 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        if targets.ndim != 1:
            raise ValueError("targets must have shape (samples,) or (samples, 1).")
        if dynamic_sequences.shape[0] != targets.shape[0]:
            raise ValueError("dynamic_sequences and targets must have the same sample count.")
        if static_attributes is not None and static_attributes.shape[0] != targets.shape[0]:
            raise ValueError("static_attributes must have the same sample count as targets.")

        self.dynamic_sequences = dynamic_sequences.float()
        self.static_attributes = None if static_attributes is None else static_attributes.float()
        self.targets = targets.float()
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = {
            "dynamic": self.dynamic_sequences[index],
            "target": self.targets[index],
        }
        if self.static_attributes is not None:
            sample["static"] = self.static_attributes[index]
        return sample

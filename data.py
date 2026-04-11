"""Compatibility exports for the assignment's requested data module name."""

from dataset.minicamels_dataset import summarize_dataset
from dataset.sequence_dataset import StreamflowSequenceDataset

__all__ = ["StreamflowSequenceDataset", "summarize_dataset"]

"""Training workflow for streamflow sequence models."""

from __future__ import annotations

from argparse import Namespace


def train_model(args: Namespace) -> None:
    """Train a sequence model and save the best checkpoint."""
    raise NotImplementedError(
        "Training will be implemented after the MiniCAMELS dataloaders are connected."
    )

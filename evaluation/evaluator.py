"""Evaluation workflow for trained checkpoints."""

from __future__ import annotations

from argparse import Namespace


def evaluate_checkpoint(args: Namespace) -> None:
    """Evaluate a trained checkpoint on held-out MiniCAMELS basins."""
    raise NotImplementedError(
        "Evaluation will be implemented after the MiniCAMELS dataloaders are connected."
    )

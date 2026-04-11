"""Plotting helpers for exploratory analysis, training curves, and evaluation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history: Mapping[str, Sequence[float]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_predicted_observed(dates, observed, predicted, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, observed, label="Observed", linewidth=1.5)
    ax.plot(dates, predicted, label="Predicted", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_parity(observed, predicted, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    observed_values = np.asarray(observed)
    predicted_values = np.asarray(predicted)
    lower = float(np.nanmin([observed_values.min(), predicted_values.min()]))
    upper = float(np.nanmax([observed_values.max(), predicted_values.max()]))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(observed_values, predicted_values, s=12, alpha=0.5)
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.set_xlabel("Observed streamflow")
    ax.set_ylabel("Predicted streamflow")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def create_all_plots(args: Namespace) -> None:
    raise NotImplementedError(
        "Plot orchestration will be implemented after training/evaluation outputs exist."
    )

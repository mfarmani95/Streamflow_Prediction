"""Training workflow for streamflow sequence models."""

from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from dataset.minicamels_dataset import build_dataloaders
from model import build_model
from training.early_stopper import EarlyStopper
from training.losses import build_loss
from util.metrics import nse


def _resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _model_kwargs(args: Namespace, metadata: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "num_dynamic_features": len(metadata["dynamic_inputs"]),
        "num_static_features": len(metadata["static_attributes"]),
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    if args.model == "lstm":
        base["hidden_size"] = args.hidden_size
    elif args.model == "transformer":
        base["d_model"] = args.hidden_size
        base["nhead"] = args.nhead
        base["dim_feedforward"] = args.dim_feedforward
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return base


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dynamic = batch["dynamic"].to(device)
    static = batch["static"].to(device)
    target = batch["target"].to(device)
    return dynamic, static, target


def _target_to_original(values: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    scalers = metadata["scalers"]
    mean = float(np.asarray(scalers["target_mean"]).reshape(-1)[0])
    std = float(np.asarray(scalers["target_std"]).reshape(-1)[0])
    return values * (std + 1e-6) + mean


def _run_validation(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    metadata: Dict[str, Any],
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            dynamic, static, target = _move_batch(batch, device)
            prediction = model(dynamic, static)
            loss = criterion(prediction, target)
            batch_size = target.shape[0]
            loss_sum += float(loss.item()) * batch_size
            n_samples += batch_size
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

    if n_samples == 0:
        return float("nan"), float("nan")

    pred_norm = np.concatenate(predictions)
    target_norm = np.concatenate(targets)
    pred = _target_to_original(pred_norm, metadata)
    obs = _target_to_original(target_norm, metadata)
    return loss_sum / n_samples, nse(obs, pred)


def _write_history_csv(history: Dict[str, list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", *keys])
        writer.writeheader()
        for idx in range(len(history[keys[0]])):
            row = {"epoch": idx + 1}
            row.update({key: history[key][idx] for key in keys})
            writer.writerow(row)


def train_model(args: Namespace) -> None:
    """Train a sequence model and save the best checkpoint."""
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders, metadata = build_dataloaders(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        seed=args.seed,
        limit_basins=args.limit_basins,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_kwargs = _model_kwargs(args, metadata)
    model = build_model(args.model, **model_kwargs).to(device)
    criterion = build_loss(args.loss, config={"scalers": metadata["scalers"]})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    history = {"train_loss": [], "val_loss": [], "val_nse": []}
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"Using device: {device}")
    print(f"Samples: {metadata['sample_counts']}")
    print(f"Basins: {metadata['basin_counts']}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in loaders["train"]:
            dynamic, static, target = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(dynamic, static)
            loss = criterion(prediction, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = target.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / max(1, train_samples)
        val_loss, val_nse = _run_validation(
            model,
            loaders["val"],
            criterion,
            device,
            metadata,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_nse"].append(val_nse)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_nse={val_nse:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_kwargs": model_kwargs,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "metadata": metadata,
                },
                checkpoint_path,
            )

        if stopper.step(val_loss):
            print(f"Early stopping after epoch {epoch}.")
            break

    history_path = output_dir / "training_history.csv"
    _write_history_csv(history, history_path)

    from visualization import plot_training_history

    plot_training_history(history, str(output_dir / "training_history.png"))
    print(f"Best checkpoint saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")

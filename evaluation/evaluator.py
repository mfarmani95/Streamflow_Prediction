"""Evaluation workflow for trained checkpoints."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from dataset.minicamels_dataset import build_dataloaders
from model import build_model
from util.metrics import regression_metrics


def _resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _target_to_original(values: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    scalers = metadata["scalers"]
    mean = float(np.asarray(scalers["target_mean"]).reshape(-1)[0])
    std = float(np.asarray(scalers["target_std"]).reshape(-1)[0])
    return values * (std + 1e-6) + mean


def _predict_test_set(model, loader, device, metadata: Dict[str, Any]) -> pd.DataFrame:
    model.eval()
    records = []

    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            target = batch["target"].to(device)
            prediction = model(dynamic, static)

            pred = _target_to_original(prediction.detach().cpu().numpy(), metadata)
            obs = _target_to_original(target.detach().cpu().numpy(), metadata)

            for basin_id, date, observed, predicted in zip(
                batch["basin_id"],
                batch["date"],
                obs,
                pred,
            ):
                records.append(
                    {
                        "basin_id": basin_id,
                        "date": date,
                        "observed": float(observed),
                        "predicted": float(predicted),
                    }
                )

    return pd.DataFrame.from_records(records)


def _metrics_by_basin(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basin_id, group in predictions.groupby("basin_id"):
        metrics = regression_metrics(group["observed"], group["predicted"])
        rows.append({"basin_id": basin_id, **metrics, "n_samples": int(len(group))})
    return pd.DataFrame(rows).sort_values("nse", ascending=False)


def evaluate_checkpoint(args: Namespace) -> None:
    """Evaluate a trained checkpoint on held-out MiniCAMELS basins."""
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _torch_load(checkpoint_path, device)
    metadata = checkpoint["metadata"]

    loaders, _ = build_dataloaders(
        data_dir=args.data_dir,
        seq_len=metadata["seq_len"],
        forecast_horizon=metadata["forecast_horizon"],
        batch_size=args.batch_size,
        dynamic_inputs=metadata["dynamic_inputs"],
        target_variable=metadata["target_variable"],
        static_attributes=metadata["static_attributes"],
        split_ids=metadata["splits"],
        scalers=metadata["scalers"],
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    predictions = _predict_test_set(model, loaders["test"], device, metadata)
    predictions_path = output_dir / "test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    overall = regression_metrics(predictions["observed"], predictions["predicted"])
    basin_metrics = _metrics_by_basin(predictions)
    basin_metrics_path = output_dir / "test_metrics_by_basin.csv"
    basin_metrics.to_csv(basin_metrics_path, index=False)

    best_basin = basin_metrics.iloc[0]["basin_id"]
    worst_basin = basin_metrics.iloc[-1]["basin_id"]
    report = {
        "overall": overall,
        "best_basin_by_nse": best_basin,
        "worst_basin_by_nse": worst_basin,
        "prediction_file": str(predictions_path),
        "basin_metrics_file": str(basin_metrics_path),
    }
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2))

    from visualization import plot_parity, plot_predicted_observed

    plot_parity(
        predictions["observed"],
        predictions["predicted"],
        str(output_dir / "test_parity.png"),
    )
    for label, basin_id in [("best", best_basin), ("worst", worst_basin)]:
        subset = predictions[predictions["basin_id"] == basin_id]
        plot_predicted_observed(
            pd.to_datetime(subset["date"]),
            subset["observed"],
            subset["predicted"],
            str(output_dir / f"test_{label}_basin_timeseries.png"),
        )

    print(json.dumps(report, indent=2))

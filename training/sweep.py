"""Hyperparameter sweep runner."""

from __future__ import annotations

import csv
import json
import re
from argparse import Namespace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from util.config import load_yaml_config, train_defaults_from_config


_DEFAULT_SWEEP_GRID = {
    "seq_len": [30, 60, 90, 120, 360],
    "hidden_size": [32, 64, 128, 256],
    "batch_size": [26, 32, 64, 128],
}
_DEFAULT_SWEEP_CONSTANTS = {"loss": "kge", "lr": 1e-3}
_SWEEP_CONTROL_KEYS = {
    "grid",
    "output_root",
    "eval_batch_size",
    "evaluate",
    "skip_existing",
    "dry_run",
}
_PARAM_ALIASES = {
    "batch": "batch_size",
    "batches": "batch_size",
    "batch_sizes": "batch_size",
    "hidden": "hidden_size",
    "hidden_sizes": "hidden_size",
    "learning_rate": "lr",
    "learning_rates": "lr",
    "lrs": "lr",
    "sequence_length": "seq_len",
    "sequence_lengths": "seq_len",
    "seq": "seq_len",
    "seq_lens": "seq_len",
}
_RUN_NAME_TOKENS = {
    "seq_len": "seq",
    "hidden_size": "hidden",
    "batch_size": "batch",
    "window_stride": "stride",
    "forecast_horizon": "horizon",
    "lr": "lr",
    "loss": "loss",
    "dropout": "drop",
    "num_layers": "layers",
    "weight_decay": "wd",
    "model": "model",
}
_ZERO_PADDED_NAME_PARAMS = {"seq_len", "hidden_size", "batch_size", "window_stride"}
_SUMMARY_PREFERRED_FIELDS = [
    "run_name",
    "model",
    "seq_len",
    "forecast_horizon",
    "window_stride",
    "hidden_size",
    "num_layers",
    "nhead",
    "dim_feedforward",
    "dropout",
    "batch_size",
    "lr",
    "loss",
    "weight_decay",
    "grad_clip",
    "patience",
    "min_delta",
    "seed",
    "split_strategy",
    "split_stratify_attribute",
    "train_basin_count",
    "val_basin_count",
    "test_basin_count",
    "output_dir",
    "checkpoint",
    "status",
    "mse",
    "mae",
    "rmse",
    "nse",
    "kge",
    "best_basin_by_nse",
    "worst_basin_by_nse",
]


def _base_train_args(config_path: str) -> Dict[str, Any]:
    defaults = train_defaults_from_config(load_yaml_config(config_path))
    return {
        "config": config_path,
        "model": defaults.get("model", "lstm"),
        "seq_len": defaults.get("seq_len", 30),
        "forecast_horizon": defaults.get("forecast_horizon", 1),
        "window_stride": defaults.get("window_stride"),
        "dynamic_inputs": defaults.get(
            "dynamic_inputs",
            ["prcp", "tmax", "tmin", "srad", "vp"],
        ),
        "target_variable": defaults.get("target_variable", "qobs"),
        "static_attributes": defaults.get("static_attributes"),
        "epochs": defaults.get("epochs", 20),
        "batch_size": defaults.get("batch_size", 64),
        "lr": defaults.get("lr", 1e-3),
        "hidden_size": defaults.get("hidden_size", 64),
        "num_layers": defaults.get("num_layers", 1),
        "nhead": defaults.get("nhead", 4),
        "dim_feedforward": defaults.get("dim_feedforward", 128),
        "dropout": defaults.get("dropout", 0.0),
        "loss": defaults.get("loss", "mse"),
        "weight_decay": defaults.get("weight_decay", 0.0),
        "grad_clip": defaults.get("grad_clip", 1.0),
        "patience": defaults.get("patience", 10),
        "min_delta": defaults.get("min_delta", 0.0),
        "seed": defaults.get("seed", 42),
        "split_strategy": defaults.get("split_strategy", "random"),
        "split_stratify_attribute": defaults.get("split_stratify_attribute", "aridity"),
        "device": defaults.get("device", "auto"),
        "num_workers": defaults.get("num_workers", 0),
        "limit_basins": defaults.get("limit_basins"),
        "train_basin_count": defaults.get("train_basin_count"),
        "val_basin_count": defaults.get("val_basin_count"),
        "test_basin_count": defaults.get("test_basin_count"),
        "data_dir": defaults.get("data_dir"),
        "output_dir": defaults.get("output_dir", "outputs"),
        "checkpoint": defaults.get("checkpoint", "outputs/best_model.pt"),
    }


def _canonical_param_name(name: str) -> str:
    normalized = name.strip().replace("-", "_").lower()
    return _PARAM_ALIASES.get(normalized, normalized)


def _as_grid_values(value: Any) -> List[Any]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("Sweep grid values cannot be empty.")
    return list(values)


def _grid_from_sweep_config(sweep_config: Mapping[str, Any]) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    nested_grid = sweep_config.get("grid", {}) or {}
    if nested_grid and not isinstance(nested_grid, Mapping):
        raise ValueError("sweep.grid must be a mapping of parameter names to lists.")

    for raw_name, values in nested_grid.items():
        grid[_canonical_param_name(raw_name)] = _as_grid_values(values)

    for raw_name, values in sweep_config.items():
        if raw_name in _SWEEP_CONTROL_KEYS:
            continue
        if isinstance(values, list):
            grid[_canonical_param_name(raw_name)] = _as_grid_values(values)

    return grid


def _constants_from_sweep_config(sweep_config: Mapping[str, Any]) -> Dict[str, Any]:
    constants: Dict[str, Any] = {}
    for raw_name, value in sweep_config.items():
        if raw_name in _SWEEP_CONTROL_KEYS or isinstance(value, list):
            continue
        constants[_canonical_param_name(raw_name)] = value
    return constants


def _override_grid(grid: Dict[str, List[Any]], param_name: str, values: Sequence[Any] | None) -> None:
    if values is not None:
        grid[param_name] = list(values)


def _override_constant(
    grid: Dict[str, List[Any]],
    constants: Dict[str, Any],
    param_name: str,
    value: Any,
) -> None:
    if value is not None:
        grid.pop(param_name, None)
        constants[param_name] = value


def _resolve_bool(cli_value: bool | None, config_value: Any, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if config_value is not None:
        return bool(config_value)
    return default


def _resolve_sweep_setup(args: Namespace) -> tuple[Dict[str, List[Any]], Dict[str, Any], Dict[str, Any]]:
    config = load_yaml_config(args.config)
    sweep_config = config.get("sweep", {}) or {}
    if not isinstance(sweep_config, Mapping):
        raise ValueError("sweep must be a mapping in the YAML config.")

    grid = _grid_from_sweep_config(sweep_config)
    if not grid:
        grid = {key: list(values) for key, values in _DEFAULT_SWEEP_GRID.items()}

    constants = {**_DEFAULT_SWEEP_CONSTANTS, **_constants_from_sweep_config(sweep_config)}
    _override_grid(grid, "seq_len", args.seq_lens)
    _override_grid(grid, "hidden_size", args.hidden_sizes)
    _override_grid(grid, "batch_size", args.batch_sizes)
    _override_constant(grid, constants, "loss", args.loss)
    _override_constant(grid, constants, "lr", args.lr)

    options = {
        "output_root": args.output_root or sweep_config.get("output_root", "outputs/sweeps"),
        "eval_batch_size": (
            args.eval_batch_size
            if args.eval_batch_size is not None
            else sweep_config.get("eval_batch_size", 256)
        ),
        "evaluate": _resolve_bool(args.evaluate, sweep_config.get("evaluate"), True),
        "skip_existing": _resolve_bool(args.skip_existing, sweep_config.get("skip_existing"), False),
        "dry_run": _resolve_bool(args.dry_run, sweep_config.get("dry_run"), False),
    }
    return grid, constants, options


def _grid_combinations(grid: Mapping[str, Sequence[Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
    param_names = list(grid.keys())
    value_lists = [_as_grid_values(grid[param_name]) for param_name in param_names]
    combinations = [
        dict(zip(param_names, values))
        for values in product(*value_lists)
    ]
    return param_names, combinations


def _format_value_for_name(param_name: str, value: Any) -> str:
    if param_name in _ZERO_PADDED_NAME_PARAMS and isinstance(value, int):
        return f"{value:03d}"
    if isinstance(value, float):
        text = f"{value:g}"
    elif isinstance(value, (list, tuple)):
        text = "-".join(str(item) for item in value)
    else:
        text = str(value)
    text = text.replace("-", "m").replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text or "value"


def _run_name(train_args: Mapping[str, Any], grid_params: Sequence[str]) -> str:
    pieces = []
    for param_name in grid_params:
        token = _RUN_NAME_TOKENS.get(param_name, param_name)
        pieces.append(f"{token}{_format_value_for_name(param_name, train_args[param_name])}")
    return "_".join(pieces) if pieces else "single_run"


def _read_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    report = json.loads(path.read_text())
    overall = report.get("overall", {})
    return {
        "mse": overall.get("mse"),
        "mae": overall.get("mae"),
        "rmse": overall.get("rmse"),
        "nse": overall.get("nse"),
        "kge": overall.get("kge"),
        "best_basin_by_nse": report.get("best_basin_by_nse"),
        "worst_basin_by_nse": report.get("worst_basin_by_nse"),
    }


def _run_config_matches(config_path: Path, train_args: Mapping[str, Any]) -> bool:
    if not config_path.exists():
        return False
    try:
        saved_config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return False

    ignored_keys = {"checkpoint", "output_dir"}
    for key, value in train_args.items():
        if key in ignored_keys:
            continue
        if saved_config.get(key) != value:
            return False
    return True


def _write_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_fields = {field for row in rows for field in row}
    fields = [field for field in _SUMMARY_PREFERRED_FIELDS if field in row_fields]
    fields.extend(sorted(row_fields - set(fields)))
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_sweep(args: Namespace) -> None:
    """Run the requested hyperparameter grid."""
    base = _base_train_args(args.config)
    grid, constants, options = _resolve_sweep_setup(args)
    grid_params, combinations = _grid_combinations(grid)
    output_root = Path(options["output_root"])
    summary_path = output_root / "sweep_results.csv"
    rows: List[Dict[str, Any]] = []

    print(f"Prepared {len(combinations)} runs.")
    print(f"Sweep grid: {grid}")

    for combination in combinations:
        train_args_dict = {**base, **constants, **combination}
        if train_args_dict.get("window_stride") is None:
            train_args_dict["window_stride"] = train_args_dict["seq_len"]

        for name in (
            "device",
            "num_workers",
            "limit_basins",
            "train_basin_count",
            "val_basin_count",
            "test_basin_count",
        ):
            value = getattr(args, name, None)
            if value is not None:
                train_args_dict[name] = value

        run_name = _run_name(train_args_dict, grid_params)
        run_dir = output_root / run_name
        checkpoint = run_dir / "best_model.pt"
        metrics_path = run_dir / "test_metrics.json"
        history_path = run_dir / "training_history.csv"
        train_args_dict["output_dir"] = str(run_dir)
        train_args_dict["checkpoint"] = str(checkpoint)

        row = {
            **{
                key: train_args_dict.get(key)
                for key in _SUMMARY_PREFERRED_FIELDS
                if key not in {"run_name", "status", "mse", "mae", "rmse", "nse", "kge", "best_basin_by_nse", "worst_basin_by_nse"}
            },
            "run_name": run_name,
            "status": "pending",
        }

        if options["dry_run"]:
            details = ", ".join(f"{key}={train_args_dict[key]}" for key in grid_params)
            extra_details = []
            if "loss" not in grid_params:
                extra_details.append(f"loss={train_args_dict['loss']}")
            if "lr" not in grid_params:
                extra_details.append(f"lr={train_args_dict['lr']}")
            extra_details.extend(
                [
                    f"window_stride={train_args_dict['window_stride']}",
                    f"output_dir={run_dir}",
                ]
            )
            if extra_details:
                details = f"{details}, {', '.join(extra_details)}" if details else ", ".join(extra_details)
            print(
                "DRY RUN | "
                f"{run_name}: {details}"
            )
            row["status"] = "dry_run"
            rows.append(row)
            continue

        completion_path = metrics_path if options["evaluate"] else history_path
        if options["skip_existing"] and completion_path.exists():
            if _run_config_matches(run_dir / "run_config.json", train_args_dict):
                print(f"Skipping existing run: {run_name}")
                row.update(_read_metrics(metrics_path))
                row["status"] = "skipped_existing"
                rows.append(row)
                _write_summary(rows, summary_path)
                continue
            print(f"Existing run differs from current config; rerunning: {run_name}")

        run_dir.mkdir(parents=True, exist_ok=True)
        train_args = Namespace(**train_args_dict)
        (run_dir / "run_config.json").write_text(
            json.dumps(vars(train_args), indent=2, default=str)
        )

        print(f"Starting run: {run_name}")
        from training.trainer import train_model

        train_model(train_args)

        if options["evaluate"]:
            from evaluation.evaluator import evaluate_checkpoint

            eval_args = Namespace(
                checkpoint=str(checkpoint),
                data_dir=train_args.data_dir,
                output_dir=str(run_dir),
                batch_size=options["eval_batch_size"],
                device=train_args.device,
                num_workers=train_args.num_workers,
            )
            evaluate_checkpoint(eval_args)
            row.update(_read_metrics(metrics_path))

        row["status"] = "completed"
        rows.append(row)
        _write_summary(rows, summary_path)

    if options["dry_run"]:
        print("Dry run complete; no files were written.")
        return

    _write_summary(rows, summary_path)
    print(f"Sweep summary saved to {summary_path}")

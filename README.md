# Streamflow Prediction with Sequence Models

This project implements HWRS640 Assignment 4.

## Structure

- `main.py` - root CLI entry point
- `cli/` - command-line parsing and orchestration
- `dataset/` - MiniCAMELS access, preprocessing, sequence dataset, dataloaders
- `model/` - sequence models such as LSTM and Transformer
- `training/` - trainer, losses, checkpointing, early stopping
- `evaluation/` - test-time evaluation workflows
- `util/` - config, metrics, data utilities, logging utilities
- `visualization.py` - plotting functions
- `configs/` - reusable experiment configuration files
- `outputs/` - checkpoints, metrics, and figures

The assignment lists `data.py`, `train.py`, `utils.py`, and `model.py` as simple
module names. This repository keeps the folder structure you requested. Thin
compatibility modules are included for `data.py`, `train.py`, and `utils.py`.
The model code lives in the `model/` package, because a repository cannot have
both a root `model.py` file and a `model/` directory with the same name.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Google Colab

```python
!git clone https://github.com/mfarmani95/Streamflow_Prediction.git
%cd Streamflow_Prediction
!pip install -r requirements.txt

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

## CLI

```bash
python3 main.py summarize-data
python3 main.py summarize-data --make-plots
python3 main.py analyze-data --config configs/default.yaml
python3 main.py train --config configs/default.yaml
python3 main.py train --model lstm --seq-len 30 --epochs 20 --loss mse --device auto
python3 main.py sweep --config configs/default.yaml --dry-run
python3 main.py sweep --config configs/default.yaml --skip-existing
python3 main.py sweep-plots --sweep-root outputs/sweeps
python3 main.py evaluate --checkpoint outputs/best_model.pt
python3 main.py plot --checkpoint outputs/best_model.pt
```

When using `--config`, command-line flags override YAML values. For example:

```bash
python3 main.py train --config configs/default.yaml --epochs 2 --limit-basins 10
```

The default loss is masked MSE, available as `--loss mse` or
`--loss masked_mse`. You can also use `--loss mae`, `--loss masked_mae`, or
`--loss kge`.

The sweep grid is controlled from the `sweep` section of the YAML config. The
default assignment grid is:

```yaml
sweep:
  output_root: outputs/sweeps
  loss: kge
  learning_rate: 0.001
  grid:
    seq_len: [30, 60, 90, 120, 360]
    hidden_size: [32, 64, 128, 256]
    batch_size: [26, 32, 64, 128]
```

Any training argument can be swept by adding it to `sweep.grid`, for example
`learning_rate: [0.0001, 0.0005, 0.001]` or `dropout: [0.0, 0.1, 0.2]`.
Each run is saved under `outputs/sweeps/` with its own checkpoint, history,
metrics, and plots. Sweep runs use non-overlapping windows by setting the
stride equal to each sequence length when `window_stride` is null.

After the sweep finishes, use `python3 main.py sweep-plots --sweep-root
outputs/sweeps` to create combined training-history figures under
`outputs/sweeps/comparison_plots/`. These compare one hyperparameter at a time
for the variables that changed across the sweep runs.

The default YAML uses a fixed basin split of 30 training basins, 10 validation
basins, and 10 test basins. The split is shuffled deterministically with
`seed: 42`, then reused across all sweep runs.

The split-aware exploratory plots are written to `outputs/data_analysis/`.
They include streamflow and forcing distributions, static attribute plots such
as aridity and q_mean, basin locations by split, missing-value diagnostics,
correlation heatmaps, non-overlapping sequence target distributions, and
example hydrographs.

## Current Implementation Notes

The default data split is by basin: 70% train, 15% validation, and 15% test.
Normalization statistics are fitted only from the training basins, then reused
for validation and test. The default static attributes are physical/climate
catchment descriptors and exclude discharge-derived signatures such as
`q_mean`, `runoff_ratio`, `hfd_mean`, and `baseflow_index` to avoid leakage.

MiniCAMELS is installed directly from GitHub because it is not published on
PyPI:

```bash
python3 -m pip install git+https://github.com/BennettHydroLab/minicamels.git
```

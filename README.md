# Streamflow Prediction with Sequence Models

This project implements HWRS640 Assignment 4 using a package layout similar to
your Noah/soil-solver style projects.

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
python3 main.py train --model lstm --seq-len 30 --epochs 20 --loss mse --device auto
python3 main.py evaluate --checkpoint outputs/best_model.pt
python3 main.py plot --checkpoint outputs/best_model.pt
```

The default loss is masked MSE, available as `--loss mse` or
`--loss masked_mse`. You can also use `--loss mae`, `--loss masked_mae`, or
`--loss kge`.

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

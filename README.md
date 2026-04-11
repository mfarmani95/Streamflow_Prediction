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

## CLI

```bash
python3 main.py summarize-data
python3 main.py train --model lstm --seq-len 30 --epochs 20
python3 main.py evaluate --checkpoint outputs/best_model.pt
python3 main.py plot --checkpoint outputs/best_model.pt
```

## Current Implementation Notes

The repository structure, model classes, metrics, losses, early stopper, and
plotting helpers are scaffolded. The next step is to connect `dataset/` to the
actual `minicamels.MiniCamels` API, then implement the full trainer and
evaluation workflow on top of those dataloaders.

"""Evaluation metrics for streamflow prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def mse(observed, predicted) -> float:
    obs = _as_array(observed)
    pred = _as_array(predicted)
    return float(np.mean((obs - pred) ** 2))


def mae(observed, predicted) -> float:
    obs = _as_array(observed)
    pred = _as_array(predicted)
    return float(np.mean(np.abs(obs - pred)))


def rmse(observed, predicted) -> float:
    return float(np.sqrt(mse(observed, predicted)))


def nse(observed, predicted, eps: float = 1e-12) -> float:
    obs = _as_array(observed)
    pred = _as_array(predicted)
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2) + eps
    return float(1.0 - numerator / denominator)


def kge(observed, predicted, eps: float = 1e-12) -> float:
    obs = _as_array(observed)
    pred = _as_array(predicted)
    r = np.corrcoef(obs, pred)[0, 1] if obs.size > 1 else np.nan
    alpha = (np.std(pred) + eps) / (np.std(obs) + eps)
    beta = (np.mean(pred) + eps) / (np.mean(obs) + eps)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def regression_metrics(observed, predicted) -> Dict[str, float]:
    return {
        "mse": mse(observed, predicted),
        "mae": mae(observed, predicted),
        "rmse": rmse(observed, predicted),
        "nse": nse(observed, predicted),
        "kge": kge(observed, predicted),
    }

"""Sequence model package."""

from model.lstm import LSTMStreamflowModel
from model.transformer import TransformerStreamflowModel


def build_model(model_name: str, **kwargs):
    normalized = model_name.lower()
    if normalized == "lstm":
        return LSTMStreamflowModel(**kwargs)
    if normalized == "transformer":
        return TransformerStreamflowModel(**kwargs)
    raise ValueError(f"Unsupported model: {model_name}")


__all__ = ["LSTMStreamflowModel", "TransformerStreamflowModel", "build_model"]

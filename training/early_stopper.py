"""Early stopping utility."""

from __future__ import annotations


class EarlyStopper:
    """Stop training when a validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

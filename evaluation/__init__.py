"""Evaluation package."""

__all__ = ["evaluate_checkpoint"]


def __getattr__(name):
    if name == "evaluate_checkpoint":
        from evaluation.evaluator import evaluate_checkpoint

        return evaluate_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

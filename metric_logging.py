"""Dependency-light helpers for persisting scalar training metrics.

Training metrics can arrive as Python numbers, NumPy scalars, or detached
PyTorch tensors.  W&B accepts most of those objects directly, while the
standard JSON encoder does not.  Keep the conversion in a small module that
can be tested without importing the full training stack.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def jsonable_metric(value: Any) -> Any:
    """Recursively convert tensor/array-like metric values to JSON types."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable_metric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_metric(item) for item in value]

    # PyTorch tensors: detach before moving to CPU so persisting metrics never
    # retains an autograd graph or attempts to serialize a CUDA object.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()

    numel = getattr(value, "numel", None)
    if callable(numel):
        if int(numel()) == 1:
            return jsonable_metric(value.item())
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return jsonable_metric(tolist())

    # NumPy scalars expose item(); NumPy arrays expose tolist().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable_metric(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return jsonable_metric(tolist())

    raise TypeError(
        f"Metric value of type {type(value).__name__} is not JSON serializable"
    )


def metric_scalar(value: Any) -> float:
    """Return one metric value as a Python float, rejecting vectors."""

    converted = jsonable_metric(value)
    if isinstance(converted, bool) or not isinstance(converted, (int, float)):
        raise TypeError(f"Expected a scalar metric, got {type(converted).__name__}")
    return float(converted)


def append_metrics_jsonl(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Append one fully converted metric record and return that record."""

    converted = jsonable_metric(payload)
    if not isinstance(converted, dict):  # Defensive: payload is typed as a mapping.
        raise TypeError("Metric payload must serialize to a JSON object")
    with Path(path).open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
        metrics_file.flush()
    return converted

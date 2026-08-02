from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def create_model(name: str, **kwargs: Any) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        known = ", ".join(sorted(_MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}. Known models: {known}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)

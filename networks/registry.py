from __future__ import annotations

from typing import Callable, Dict, Any
import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    def deco(fn: Callable[..., nn.Module]):
        _MODEL_REGISTRY[name] = fn
        return fn
    return deco

def create_model(name: str, **kwargs: Any) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        known = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Known models: {known}")
    return _MODEL_REGISTRY[name](**kwargs)

def list_models():
    return sorted(_MODEL_REGISTRY.keys())

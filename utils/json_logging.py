import json
from dataclasses import asdict, is_dataclass
from typing import Any

def save_json(path: str, obj: Any, indent: int = 2) -> None:
    if is_dataclass(obj):
        payload = asdict(obj)
    else:
        payload = obj
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)

from __future__ import annotations

from typing import Dict, List

from games.base import GameAdapter
from games.pong_adapter import pong_adapter


_REGISTRY: Dict[str, GameAdapter] = {
    "pong": pong_adapter(),
}


def list_games() -> List[GameAdapter]:
    return list(_REGISTRY.values())


def get_game(name: str) -> GameAdapter:
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown game '{name}'. Available: {available}")
    return _REGISTRY[key]

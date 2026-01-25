from __future__ import annotations

from typing import Dict, List

from games.base import GameAdapter
from games.maze_adapter import maze_adapter
from games.template_adapter import template_adapter


_REGISTRY: Dict[str, GameAdapter] = {
    "maze": maze_adapter(),
    "template": template_adapter(),
}


def list_games() -> List[GameAdapter]:
    return list(_REGISTRY.values())


def get_game(name: str) -> GameAdapter:
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown game '{name}'. Available: {available}")
    return _REGISTRY[key]

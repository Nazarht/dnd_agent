"""Завантаження світів і персонажів з JSON-файлів у game_data/."""

from pathlib import Path

from dnd_agent.config import GAME_DATA_DIR
from dnd_agent.game.character import Character
from dnd_agent.game.world import GameMap
from dnd_agent.utils.json_io import load_json


def _find_file(kind: str, name: str) -> Path:
    """Шукає game_data/{kind}/{name}.json; якщо немає, показує доступні назви."""
    folder = GAME_DATA_DIR / kind
    path = folder / f"{name}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in folder.glob("*.json")))
        raise FileNotFoundError(f"{kind}/{name}.json не знайдено. Доступні: {available}")
    return path


def load_world(name: str) -> GameMap:
    """Завантажує світ за назвою файлу, наприклад 'forest'."""
    return GameMap(**load_json(_find_file("worlds", name)))


def load_character(name: str, position: tuple[int, int]) -> Character:
    """Завантажує персонажа за назвою файлу, наприклад 'priest'.

    Позицію задає світ (world.start), тому в файлі персонажа її немає.
    """
    return Character(**load_json(_find_file("characters", name)), position=position)

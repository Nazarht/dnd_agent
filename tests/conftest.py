"""Спільні фікстури: свіжі світ і персонаж у кожному тесті, фіксоване насіння кубиків."""

import pytest

from dnd_agent.game.loader import load_character, load_world
from dnd_agent.tools.dice import seed_dice


@pytest.fixture(autouse=True)
def fixed_dice():
    """Кубики відтворювані: однакове насіння в кожному тесті."""
    seed_dice(0)


@pytest.fixture
def world():
    """Свіжа карта forest: карта змінюється на місці, тож кожен тест бере власну копію."""
    return load_world("forest")


@pytest.fixture
def priest(world):
    """Персонаж priest на стартовій клітинці світу (0, 3)."""
    return load_character("priest", world.start)

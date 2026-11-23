"""Тести завантаження світів і персонажів з game_data/."""

import pytest

from dnd_agent.game.loader import load_character, load_world


def test_load_world_reads_json_file():
    world = load_world("forest")
    assert world.name == "Темний ліс"
    assert (world.width, world.height) == (8, 8)
    assert world.start == (0, 3)


def test_load_character_takes_position_from_world():
    world = load_world("forest")
    character = load_character("priest", world.start)
    assert character.name == "Отець Маркус"
    assert character.position == (0, 3)
    assert character.hp == character.max_hp == 16


def test_missing_world_lists_available_names():
    with pytest.raises(FileNotFoundError, match="Доступні: forest"):
        load_world("nowhere")


def test_missing_character_lists_available_names():
    with pytest.raises(FileNotFoundError, match="Доступні: priest"):
        load_character("nobody", (0, 0))

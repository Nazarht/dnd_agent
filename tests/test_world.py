"""Тести карти: валідація, клітинки, редагування за правилами і опис околиць."""

import pytest
from pydantic import ValidationError

from dnd_agent.game.world import Cell, GameMap


def make_map(rows, start=(0, 0)):
    return GameMap(name="test", start=start, rows=rows)


def test_cell_returns_type_by_coordinates(world):
    assert world.cell(3, 2) == Cell.MONSTER
    assert world.cell(4, 0) == Cell.TREE


def test_cell_outside_map_is_none(world):
    assert world.cell(-1, 0) is None
    assert world.cell(8, 0) is None


@pytest.mark.parametrize(
    ("x", "y", "blocked"),
    [
        (4, 0, True),  # дерево
        (1, 1, True),  # стіна
        (6, 1, True),  # вода
        (3, 2, True),  # монстр
        (6, 3, False),  # скарб не блокує
        (0, 3, False),  # підлога
        (-1, 3, True),  # поза картою
    ],
)
def test_is_blocked(world, x, y, blocked):
    assert world.is_blocked(x, y) is blocked


def test_rejects_uneven_rows():
    with pytest.raises(ValidationError, match="однакової довжини"):
        make_map(["...", ".."])


def test_rejects_unknown_symbol():
    with pytest.raises(ValidationError, match="Невідомі символи"):
        make_map([".X"])


def test_rejects_empty_map():
    with pytest.raises(ValidationError, match="порожньою"):
        make_map([])


def test_rejects_blocked_start():
    with pytest.raises(ValidationError, match="заблокована"):
        make_map(["T.."], start=(0, 0))


def test_edit_cell_changes_map_in_place(world):
    old = world.edit_cell(3, 2, Cell.FLOOR)
    assert old == Cell.MONSTER
    assert world.cell(3, 2) == Cell.FLOOR


def test_edit_cell_rejects_outside_map(world):
    with pytest.raises(ValueError, match="поза картою"):
        world.edit_cell(9, 9, Cell.FLOOR)


def test_edit_cell_rejects_occupied_by_player(world):
    with pytest.raises(ValueError, match="персонаж стоїть"):
        world.edit_cell(0, 3, Cell.TREE, occupied=(0, 3))


@pytest.mark.parametrize(("x", "y"), [(1, 1), (6, 1)])  # стіна і вода
def test_edit_cell_rejects_protected_terrain(world, x, y):
    with pytest.raises(ValueError, match="постійний рельєф"):
        world.edit_cell(x, y, Cell.FLOOR)


def test_edit_cell_rejects_same_type(world):
    with pytest.raises(ValueError, match="вже підлога"):
        world.edit_cell(0, 3, Cell.FLOOR)


def test_to_text_marks_player_position(world):
    text = world.to_text((0, 3))
    assert text.count("@") == 1
    assert "T" in text  # інші клітинки видно


def test_surroundings_are_clipped_at_map_corner(world):
    text = world.surroundings((0, 0), 2)
    assert "-1" not in text  # координати поза картою не показуємо
    assert text.splitlines()[0].endswith(" 0 1 2")


def test_surroundings_list_nearby_monster(world):
    text = world.surroundings((2, 3), 2)
    assert "- монстр на (3, 2): dx=+1, dy=-1" in text


def test_surroundings_without_objects(world):
    text = world.surroundings((0, 0), 1)
    assert text.endswith("Об'єкти поруч:\n- немає")


@pytest.mark.xfail(
    strict=True,
    reason="Відома помилка: скарб під персонажем прихований символом @ і не перелічується",
)
def test_surroundings_show_treasure_under_player(world):
    text = world.surroundings((6, 3), 2)
    assert "скарб" in text

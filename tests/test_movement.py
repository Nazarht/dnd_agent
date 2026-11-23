"""Тести руху: зупинка на перешкодах, край карти і інструмент move_character."""

import pytest
from pydantic import ValidationError

from dnd_agent.game.world import Direction
from dnd_agent.tools.movement import MoveInput, describe_move, move_character, walk


def test_walk_moves_full_distance_on_free_floor(world):
    result = walk(world, (0, 3), Direction.EAST, 5)
    assert (result.taken, result.end, result.blocker) == (5, (5, 3), None)


def test_walk_stops_at_map_edge(world):
    result = walk(world, (0, 3), Direction.NORTH, 5)
    assert (result.taken, result.end) == (3, (0, 0))
    assert result.blocker == "край карти"


def test_walk_stops_before_monster(world):
    result = walk(world, (3, 3), Direction.NORTH, 2)
    assert (result.taken, result.end) == (0, (3, 3))
    assert result.blocker == "монстр на (3, 2)"


def test_walk_stops_before_tree(world):
    result = walk(world, (2, 4), Direction.WEST, 1)
    assert result.blocker == "дерево на (1, 4)"


def test_walk_stops_before_water(world):
    result = walk(world, (1, 6), Direction.EAST, 3)
    assert result.blocker == "вода на (2, 6)"
    assert result.taken == 0


def test_describe_blocked_move_text(world):
    result = walk(world, (2, 4), Direction.WEST, 1)
    text = describe_move(result, world)
    assert text.startswith("Не можна піти на захід: перешкода - дерево на (1, 4).")


def test_describe_partial_move_text(world):
    result = walk(world, (0, 3), Direction.NORTH, 5)
    text = describe_move(result, world)
    assert "пройдено 3 з 5" in text
    assert "Зупинено: край карти." in text


def test_move_input_rejects_zero_steps():
    with pytest.raises(ValidationError):
        MoveInput(direction="east", steps=0)


def test_move_input_rejects_more_than_five_steps():
    with pytest.raises(ValidationError):
        MoveInput(direction="east", steps=6)


def test_move_input_rejects_unknown_direction():
    with pytest.raises(ValidationError):
        MoveInput(direction="up", steps=1)


def test_move_tool_updates_only_position(world, priest):
    command = move_character.invoke(
        {"direction": "east", "steps": 2, "character": priest, "world": world, "tool_call_id": "t1"}
    )
    assert command.update["character"].position == (2, 3)
    assert "world" not in command.update  # карта рухом не змінюється


def test_move_tool_does_not_mutate_original(world, priest):
    move_character.invoke(
        {"direction": "east", "steps": 2, "character": priest, "world": world, "tool_call_id": "t1"}
    )
    assert priest.position == (0, 3)

"""Тести modify_map: успішна зміна, помилки правил і валідація вхідних даних."""

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from dnd_agent.game.world import Cell
from dnd_agent.tools.map_edit import modify_map


def edit(world, priest, **overrides):
    args = {"x": 3, "y": 2, "cell": "floor", "reason": "гоблін переможений",
            "character": priest, "world": world, "tool_call_id": "t1"}
    args.update(overrides)
    return modify_map.invoke(args)


def test_success_changes_map_and_returns_world(world, priest):
    command = edit(world, priest)
    assert command.update["world"].cell(3, 2) == Cell.FLOOR
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.content.startswith("Карту оновлено на (3, 2): монстр -> підлога.")


def test_player_cell_is_protected(world, priest):
    command = edit(world, priest, x=0, y=3, cell="tree")
    message = command.update["messages"][0]
    assert message.status == "error"
    assert "Карту не змінено" in message.content
    assert "world" not in command.update


def test_permanent_terrain_is_protected(world, priest):
    command = edit(world, priest, x=1, y=1, cell="floor")
    assert command.update["messages"][0].status == "error"


def test_unknown_cell_name_is_rejected_by_schema(world, priest):
    with pytest.raises(ValidationError):
        edit(world, priest, cell="dragon")


def test_negative_coordinate_is_rejected_by_schema(world, priest):
    with pytest.raises(ValidationError):
        edit(world, priest, x=-1)


@pytest.mark.parametrize("field", ["x", "y"])
def test_coordinate_outside_8x8_map_is_rejected_by_validator(world, priest, field):
    with pytest.raises(ValidationError, match=f"{field}=8 поза картою"):
        edit(world, priest, **{field: 8})


def test_last_cell_of_the_map_is_accepted(world, priest):
    command = edit(world, priest, x=7, y=7, cell="tree", reason="дерево виросло")
    assert command.update["world"].cell(7, 7) == Cell.TREE

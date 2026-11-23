"""Тести кубиків: розбір нотації, кидки і інструмент roll_dice."""

import random
import re

import pytest
from pydantic import ValidationError

from dnd_agent.game.character import Attribute
from dnd_agent.tools.dice import RollDiceInput, parse_dice, roll_check, roll_dice


def test_parse_dice_reads_count_and_sides():
    assert parse_dice("2d6") == (2, 6)
    assert parse_dice(" 1D20 ") == (1, 20)


@pytest.mark.parametrize("notation", ["d20", "1d7", "0d6", "21d6", "abc"])
def test_parse_dice_rejects_invalid_notation(notation):
    with pytest.raises(ValueError):
        parse_dice(notation)


def test_roll_check_total_is_sum_plus_bonuses():
    result = roll_check("3d6", "тест", attribute=Attribute.STRENGTH, attribute_bonus=2, modifier=-1,
                        rng=random.Random(7))
    assert len(result.rolls) == 3
    assert all(1 <= value <= 6 for value in result.rolls)
    assert result.total == sum(result.rolls) + 2 - 1


def test_roll_dice_input_rejects_unknown_sides():
    with pytest.raises(ValidationError):
        RollDiceInput(dice="1d7", purpose="атака")


def test_roll_dice_input_rejects_modifier_out_of_range():
    with pytest.raises(ValidationError):
        RollDiceInput(dice="1d20", purpose="атака", modifier=11)


def test_roll_dice_tool_adds_character_attribute(priest):
    text = roll_dice.invoke(
        {"dice": "1d20", "attribute": "strength", "modifier": 0, "purpose": "атака", "character": priest}
    )
    # У персонажа сила -1: має бути в тексті, і сума має це враховувати
    match = re.search(r"кубики 1d20: \[(\d+)\], сила -1 = разом (\d+)", text)
    assert match, text
    assert int(match.group(2)) == int(match.group(1)) - 1


def test_roll_dice_tool_reports_modifier(priest):
    text = roll_dice.invoke(
        {"dice": "1d4", "attribute": None, "modifier": 2, "purpose": "урон", "character": priest}
    )
    assert "модифікатор +2" in text
    assert "сила" not in text

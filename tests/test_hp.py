"""Тести HP: межі 0..max_hp, текст для LLM і інструмент modify_hp."""

import pytest
from pydantic import ValidationError

from dnd_agent.tools.hp import ModifyHpInput, apply_hp_change, describe_hp_change, modify_hp


def test_damage_lowers_hp(priest):
    change = apply_hp_change(priest, -5)
    assert (change.old_hp, change.new_hp, change.applied) == (16, 11, -5)
    assert not change.unconscious


def test_damage_cannot_go_below_zero(priest):
    change = apply_hp_change(priest, -50)
    assert change.new_hp == 0
    assert change.applied == -16
    assert change.unconscious


def test_healing_cannot_exceed_max(priest):
    change = apply_hp_change(priest, 10)
    assert change.new_hp == 16
    assert change.applied == 0


def test_describe_damage_text(priest):
    text = describe_hp_change(apply_hp_change(priest, -5), "удар пасткою")
    assert text == "Урон 5 (удар пасткою): HP 16 -> 11/16"


def test_describe_clamped_and_unconscious_text(priest):
    text = describe_hp_change(apply_hp_change(priest, -50), "вибух")
    assert "Обмежено: застосовано лише 16" in text
    assert "непритомний" in text


@pytest.mark.parametrize("delta", [0, 51, -51])
def test_modify_hp_input_rejects_bad_delta(delta):
    with pytest.raises(ValidationError):
        ModifyHpInput(delta=delta, reason="тест")


def test_modify_hp_input_rejects_empty_reason():
    with pytest.raises(ValidationError):
        ModifyHpInput(delta=-1, reason="")


def test_modify_hp_tool_updates_character_in_command(priest):
    command = modify_hp.invoke({"delta": -4, "reason": "удар", "character": priest, "tool_call_id": "t1"})
    assert command.update["character"].hp == 12
    assert command.update["messages"][0].content.startswith("Урон 4 (удар)")


def test_modify_hp_tool_does_not_mutate_original(priest):
    modify_hp.invoke({"delta": -4, "reason": "удар", "character": priest, "tool_call_id": "t1"})
    assert priest.hp == 16

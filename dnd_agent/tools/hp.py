"""Інструмент modify_hp: зміна здоров'я персонажа (урон або лікування).

HP зберігається в стані графа. Інструмент повертає Command, який оновлює
character у стані, тож єдине джерело правди - стан, а не текст відповіді LLM.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dnd_agent.game.character import Character

# Більший одноразовий зсув HP вважаємо помилкою LLM
DELTA_LIMIT = 50


class HpChange(BaseModel):
    """Результат зміни HP: скільки було, стало і що сталося."""

    old_hp: int
    new_hp: int
    max_hp: int
    requested: int
    applied: int
    unconscious: bool


def apply_hp_change(character: Character, delta: int) -> HpChange:
    """Рахує нове HP з обмеженням 0..max_hp. Не змінює character."""
    new_hp = max(0, min(character.max_hp, character.hp + delta))
    return HpChange(
        old_hp=character.hp,
        new_hp=new_hp,
        max_hp=character.max_hp,
        requested=delta,
        applied=new_hp - character.hp,
        unconscious=new_hp == 0,
    )


def describe_hp_change(change: HpChange, reason: str) -> str:
    """Текст для LLM: що сталося з HP і чи персонаж непритомний."""
    if change.requested < 0:
        head = f"Урон {-change.requested} ({reason})"
    else:
        head = f"Лікування {change.requested} ({reason})"
    text = f"{head}: HP {change.old_hp} -> {change.new_hp}/{change.max_hp}"
    if change.applied != change.requested:
        text += f". Обмежено: застосовано лише {abs(change.applied)}"
    if change.unconscious:
        text += ". Персонаж непритомний (0 HP)"
    return text


class ModifyHpInput(BaseModel):
    """Параметри modify_hp, які бачить LLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    delta: int = Field(
        ge=-DELTA_LIMIT,
        le=DELTA_LIMIT,
        description="Зміна HP: від'ємне для урону (наприклад -4), додатне для лікування (наприклад +10). "
        "Не може бути 0. Діапазон -50..50.",
    )
    reason: str = Field(
        min_length=1,
        max_length=100,
        description="Коротка причина зміни, наприклад 'удар кинджалом гобліна' або 'короткий відпочинок'.",
    )

    @field_validator("delta")
    @classmethod
    def check_not_zero(cls, value: int) -> int:
        """Нульова зміна HP нічого не робить, тож це помилка виклику."""
        if value == 0:
            raise ValueError("delta не може бути 0: вкажіть урон (від'ємне) або лікування (додатне)")
        return value


@tool(args_schema=ModifyHpInput)
def modify_hp(
    delta: int,
    reason: str,
    character: Annotated[Character, InjectedState("character")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Змінює очки здоров'я персонажа: завдає урону або лікує.

    Використовуй щоразу, коли персонаж отримує урон, лікується або втрачає HP з будь-якої причини.
    HP обмежується діапазоном 0..максимум. При 0 HP персонаж непритомний.

    Приклади:
    - Удар на 4 очки: delta=-4, reason='удар кинджалом гобліна'
    - Відпочинок під дубом: delta=+10, reason='короткий відпочинок'
    """
    change = apply_hp_change(character, delta)
    updated = character.model_copy(update={"hp": change.new_hp})
    return Command(
        update={
            "character": updated,
            "messages": [ToolMessage(content=describe_hp_change(change, reason), tool_call_id=tool_call_id)],
        }
    )

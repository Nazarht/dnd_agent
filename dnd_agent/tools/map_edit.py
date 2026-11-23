"""Інструмент modify_map: зміна клітинки на карті (наприклад, монстр переможений).

Правила (що можна змінювати) живуть у GameMap.edit_cell, інструмент лише викликає їх.
Карта змінюється на місці, але world все одно повертається в оновленні стану:
без цього checkpointer не побачить зміни і наступний хід почнеться зі старою картою.
"""

from enum import StrEnum
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from dnd_agent.game.character import Character
from dnd_agent.game.world import Cell, GameMap

# Значення для LLM - англійські імена клітинок (floor, tree, ...), як і решта ідентифікаторів
CellName = StrEnum("CellName", {cell.name: cell.name.lower() for cell in Cell})


class MapEditInput(BaseModel):
    """Параметри modify_map, які бачить LLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    x: int = Field(ge=0, description="Стовпець клітинки, відлік від 0 ліворуч.")
    y: int = Field(ge=0, description="Рядок клітинки, відлік від 0 згори.")
    cell: CellName = Field(
        description="Новий тип клітинки. Значення англійською: floor (підлога, прибирає монстра чи дерево), "
        "tree (дерево), monster (монстр), treasure (скарб), wall (стіна), water (вода)."
    )
    reason: str = Field(
        min_length=1,
        max_length=100,
        description="Коротка причина зміни, наприклад 'гоблін переможений' або 'скарб підібрано'.",
    )


@tool(args_schema=MapEditInput)
def modify_map(
    x: int,
    y: int,
    cell: CellName,
    reason: str,
    character: Annotated[Character, InjectedState("character")],
    world: Annotated[GameMap, InjectedState("world")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Змінює одну клітинку карти, наприклад прибирає переможеного монстра або відкриває шлях.

    Використовуй, коли світ змінюється: монстра переможено, скарб забрано, дерево зрубано.
    Постійний рельєф (стіни, вода) і клітинку, де стоїть персонаж, змінити не можна.
    Координати починаються з 0: x - стовпець, y - рядок.

    Приклади:
    - Гоблін на (3, 2) переможений: x=3, y=2, cell='floor', reason='гоблін переможений'
    - Скарб на (6, 3) підібрано: x=6, y=3, cell='floor', reason='скарб підібрано'
    """
    try:
        old = world.edit_cell(x, y, Cell[cell.name], occupied=character.position)
    except ValueError as error:
        return Command(
            update={
                "messages": [
                    ToolMessage(content=f"Карту не змінено: {error}.", tool_call_id=tool_call_id, status="error")
                ]
            }
        )
    return Command(
        update={
            "world": world,  # потрібно, щоб checkpointer зберіг зміну
            "messages": [
                ToolMessage(
                    content=f"Карту оновлено на ({x}, {y}): {old.label} -> {Cell[cell.name].label}. "
                    f"Причина: {reason}.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )

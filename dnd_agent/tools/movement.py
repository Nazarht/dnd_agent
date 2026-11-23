"""Інструмент move_character: рух персонажа на 1-5 клітинок у одному напрямку.

Рух іде клітинка за клітинкою і зупиняється на першій перешкоді (стіна, вода,
дерево, монстр або край карти). Персонаж стоїть на останній вільній клітинці.
Карта не змінюється, оновлюється лише позиція персонажа в стані.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from dnd_agent.game.character import Character
from dnd_agent.game.world import Direction, GameMap

MAX_STEPS = 5
# Скільки клітинок навколо нової позиції показувати в відповіді
VIEW_RADIUS = 2


class MoveResult(BaseModel):
    """Результат руху: скільки кроків вдалося зробити і що зупинило персонажа."""

    direction: Direction
    requested: int
    taken: int
    start: tuple[int, int]
    end: tuple[int, int]
    blocker: str | None


def describe_blocker(world: GameMap, x: int, y: int) -> str:
    """Опис перешкоди на клітинці (x, y): назва клітинки або край карти."""
    cell = world.cell(x, y)
    if cell is None:
        return "край карти"
    return f"{cell.label} на ({x}, {y})"


def walk(world: GameMap, start: tuple[int, int], direction: Direction, steps: int) -> MoveResult:
    """Рахує рух крок за кроком. Не змінює ні карту, ні персонажа."""
    dx, dy = direction.delta
    x, y = start
    taken = 0
    blocker = None
    for _ in range(steps):
        nx, ny = x + dx, y + dy
        if world.is_blocked(nx, ny):
            blocker = describe_blocker(world, nx, ny)
            break
        x, y = nx, ny
        taken += 1
    return MoveResult(
        direction=direction,
        requested=steps,
        taken=taken,
        start=start,
        end=(x, y),
        blocker=blocker,
    )


def describe_move(result: MoveResult, world: GameMap) -> str:
    """Текст для LLM: що сталося з рухом і що видно навколо нової позиції."""
    d = result.direction.label
    x, y = result.end
    sx, sy = result.start
    if result.taken == 0:
        head = f"Не можна піти на {d}: перешкода - {result.blocker}. Персонаж лишається на ({x}, {y})."
    elif result.blocker:
        head = (
            f"Рух на {d} з ({sx}, {sy}): пройдено {result.taken} з {result.requested}, тепер на ({x}, {y}). "
            f"Зупинено: {result.blocker}."
        )
    else:
        head = f"Рух на {d} з ({sx}, {sy}): пройдено {result.taken}, тепер на ({x}, {y})."
    view = world.surroundings((x, y), VIEW_RADIUS)
    return f"{head}\nОколиця (радіус {VIEW_RADIUS}):\n{view}"


class MoveInput(BaseModel):
    """Параметри move_character, які бачить LLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    direction: Direction = Field(
        description="Напрямок руху. Значення англійською: north (північ), south (південь), "
        "east (схід), west (захід)."
    )
    steps: int = Field(
        ge=1,
        le=MAX_STEPS,
        description="Кількість клітинок від 1 до 5. Персонаж зупиняється раніше, якщо натрапить на "
        "перешкоду: стіна, вода, дерево, монстр або край карти.",
    )


@tool(args_schema=MoveInput)
def move_character(
    direction: Direction,
    steps: int,
    character: Annotated[Character, InjectedState("character")],
    world: Annotated[GameMap, InjectedState("world")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Переміщує персонажа в одному напрямку на до 5 клітинок.

    Рух зупиняється на першій перешкоді: стіна, вода, дерево, монстр або край карти.
    Результат повідомляє, скільки пройдено, і показує околицю нової позиції.
    Монстри блокують рух, тож спершу з ними треба розібратися.

    Приклади:
    - Три клітинки на схід: direction='east', steps=3
    - На південь до річки: direction='south', steps=5
    """
    result = walk(world, character.position, direction, steps)
    moved = character.model_copy(update={"position": result.end})
    return Command(
        update={
            "character": moved,
            "messages": [ToolMessage(content=describe_move(result, world), tool_call_id=tool_call_id)],
        }
    )

"""Інструмент roll_dice: кидок кубиків з модифікатором атрибуту персонажа.

Кубики кидає RNG модуля. seed_dice(...) задає насіння, щоб тести були відтворювані.
Функція roll_check приймає власний RNG, тож тести можуть не чіпати глобальний стан.
"""

import random
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dnd_agent.game.character import Attribute, Character

ALLOWED_SIDES = (4, 6, 8, 10, 12, 20, 100)
MAX_DICE = 20
MODIFIER_LIMIT = 10
_DICE_PATTERN = re.compile(r"^(\d+)d(\d+)$")

_rng = random.Random()


def seed_dice(seed: int | None) -> None:
    """Задає насіння глобального RNG: однакове насіння дає однакові кидки."""
    _rng.seed(seed)


def parse_dice(notation: str) -> tuple[int, int]:
    """Розбирає 'NdM' на (кількість, грані). Кидає ValueError, якщо формат або значення неправильні."""
    match = _DICE_PATTERN.match(notation.strip().lower())
    if not match:
        raise ValueError(f"Некоректна нотація кубиків {notation!r}: очікується формат NdM, наприклад 1d20")
    count, sides = int(match.group(1)), int(match.group(2))
    if not 1 <= count <= MAX_DICE:
        raise ValueError(f"Кількість кубиків має бути від 1 до {MAX_DICE}, отримано {count}")
    if sides not in ALLOWED_SIDES:
        raise ValueError(f"Кубик з {sides} гранями не існує. Доступні: {', '.join(map(str, ALLOWED_SIDES))}")
    return count, sides


class DiceRoll(BaseModel):
    """Результат кидка: окремі кубики, бонуси і сума."""

    dice: str
    purpose: str
    rolls: list[int]
    attribute: Attribute | None
    attribute_bonus: int
    modifier: int
    total: int


def roll_check(
    dice: str,
    purpose: str,
    attribute: Attribute | None = None,
    attribute_bonus: int = 0,
    modifier: int = 0,
    rng: random.Random = _rng,
) -> DiceRoll:
    """Кидає кубики і додає бонус атрибуту та модифікатор до суми."""
    count, sides = parse_dice(dice)
    rolls = [rng.randint(1, sides) for _ in range(count)]
    total = sum(rolls) + attribute_bonus + modifier
    return DiceRoll(
        dice=dice.strip().lower(),
        purpose=purpose,
        rolls=rolls,
        attribute=attribute,
        attribute_bonus=attribute_bonus,
        modifier=modifier,
        total=total,
    )


class RollDiceInput(BaseModel):
    """Параметри roll_dice, які бачить LLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    dice: str = Field(
        description="Кубики у форматі NdM: '1d20' для атаки чи перевірки, '2d6' для урону. "
        "N від 1 до 20; M - одне з 4, 6, 8, 10, 12, 20, 100."
    )
    attribute: Attribute | None = Field(
        default=None,
        description="Атрибут, чий модифікатор додається до кидка. Значення англійською: "
        "strength (сила), dexterity (спритність), constitution (статура), intelligence (інтелект), "
        "wisdom (мудрість), charisma (харизма). Порожньо, якщо атрибут не потрібен.",
    )
    modifier: int = Field(
        default=0,
        ge=-MODIFIER_LIMIT,
        le=MODIFIER_LIMIT,
        description="Ситуативний бонус або штраф, наприклад +2 за фланг, -1 за темряву. Діапазон -10..10.",
    )
    purpose: str = Field(
        min_length=1,
        max_length=100,
        description="Коротка причина кидка, наприклад 'атака мечем' або 'урон кинджалом гобліна'.",
    )

    @field_validator("dice")
    @classmethod
    def check_dice(cls, value: str) -> str:
        """Перевіряє нотацію NdM: формат, кількість кубиків і грані."""
        parse_dice(value)
        return value.lower()


@tool(args_schema=RollDiceInput)
def roll_dice(
    dice: str,
    attribute: Attribute | None,
    modifier: int,
    purpose: str,
    character: Annotated[Character, InjectedState("character")],
) -> str:
    """Кидає кубики для персонажа і повертає результат.

    Використовуй, коли результат невідомий: атаки, рятівні кидки, перевірки навичок,
    урон і будь-які випадкові події. Ніколи не вигадуй кидок сам.

    Приклади:
    - Атака мечем: dice='1d20', attribute='strength', purpose='атака мечем'
    - Урон кинджалом: dice='1d4', attribute=None, purpose='урон кинджалом гобліна'
    - Атака з флангу: dice='1d20', attribute='strength', modifier=2, purpose='атака з флангу'

    Модифікатор атрибуту персонаж додає автоматично.
    """
    result = roll_check(
        dice=dice,
        purpose=purpose,
        attribute=attribute,
        attribute_bonus=character.modifier(attribute) if attribute else 0,
        modifier=modifier,
        rng=_rng,
    )
    parts = [f"кубики {result.dice}: {result.rolls}"]
    if attribute:
        parts.append(f"{attribute.label} {result.attribute_bonus:+d}")
    if modifier:
        parts.append(f"модифікатор {modifier:+d}")
    return f"{purpose} -> {', '.join(parts)} = разом {result.total}"

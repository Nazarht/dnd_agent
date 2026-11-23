"""Персонаж гравця: здоров'я, позиція та модифікатори атрибутів.

Character живе у стані графа, тому це Pydantic-модель: її легко серіалізувати
і перевіряти. Атрибути зберігаються як модифікатори (наприклад strength: +3).
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Attribute(StrEnum):
    """Шість атрибутів персонажа."""

    STRENGTH = "strength"
    DEXTERITY = "dexterity"
    CONSTITUTION = "constitution"
    INTELLIGENCE = "intelligence"
    WISDOM = "wisdom"
    CHARISMA = "charisma"

    @property
    def label(self) -> str:
        """Назва атрибуту для тексту гри."""
        return {
            "strength": "сила",
            "dexterity": "спритність",
            "constitution": "статура",
            "intelligence": "інтелект",
            "wisdom": "мудрість",
            "charisma": "харизма",
        }[self.value]


# Модифікатор атрибуту: від -5 до +5
Modifier = Annotated[int, Field(ge=-5, le=5)]


class Character(BaseModel):
    """Стан персонажа: ім'я, HP, позиція на карті та модифікатори атрибутів."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=2, description="Ім'я персонажа")
    description: str = Field(default="", description="Короткий опис персонажа для GM")
    hp: int = Field(ge=0, description="Поточні очки здоров'я")
    max_hp: int = Field(ge=1, description="Максимальні очки здоров'я")
    position: tuple[int, int] = Field(description="Позиція (x, y) на карті, x: стовпець, y: рядок")
    attributes: dict[Attribute, Modifier] = Field(
        description="Модифікатори всіх шести атрибутів",
    )

    @model_validator(mode="after")
    def check_character(self) -> "Character":
        """Перевіряє, що всі атрибути задані і HP не більший за максимум."""
        missing = set(Attribute) - set(self.attributes)
        if missing:
            names = ", ".join(sorted(a.value for a in missing))
            raise ValueError(f"Не задано атрибути: {names}")
        if self.hp > self.max_hp:
            raise ValueError(f"HP ({self.hp}) не може бути більшим за max_hp ({self.max_hp})")
        return self

    @property
    def is_alive(self) -> bool:
        """Персонаж живий, якщо HP більше нуля."""
        return self.hp > 0

    def modifier(self, attribute: Attribute) -> int:
        """Модифікатор атрибуту, наприклад modifier(Attribute.STRENGTH) -> 3."""
        return self.attributes[attribute]

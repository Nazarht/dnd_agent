"""Ігровий світ: сітка 8x8, типи клітинок, перевірка руху та опис околиць.

Карта зберігається як список рядків. Символи:
    . підлога   T дерево   # стіна   W вода   M монстр   $ скарб
Координати (x, y): x - стовпець зліва направо, y - рядок зверху вниз.
"""

from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator


class Cell(StrEnum):
    """Тип клітинки. Значення - символ на карті, label - назва для тексту LLM."""

    FLOOR = "."
    TREE = "T"
    WALL = "#"
    WATER = "W"
    MONSTER = "M"
    TREASURE = "$"

    @property
    def label(self) -> str:
        """Назва клітинки для тексту гри"""
        return {
            "FLOOR": "підлога",
            "TREE": "дерево",
            "WALL": "стіна",
            "WATER": "вода",
            "MONSTER": "монстр",
            "TREASURE": "скарб",
        }[self.name]


# Розмір карти гри: 8x8 клітинок. Використовується для перевірки координат у вхідних даних інструментів
MAP_SIZE = 8

# Клітинки, через які персонаж не може пройти
BLOCKING = frozenset({Cell.TREE, Cell.WALL, Cell.WATER, Cell.MONSTER})

# Постійний рельєф, який modify_map не може змінити
PROTECTED = frozenset({Cell.WALL, Cell.WATER})


class Direction(StrEnum):
    """Напрямок руху. delta - зсув (dx, dy); y зростає вниз, тож north це -1 по y."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @property
    def delta(self) -> tuple[int, int]:
        return {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}[self.value]

    @property
    def label(self) -> str:
        """Назва напрямку для тексту гри."""
        return {"north": "північ", "south": "південь", "east": "схід", "west": "захід"}[self.value]


class GameMap(BaseModel):
    """Світ: назва, опис, стартова позиція та карта. rows[y][x] - символ клітинки."""

    name: str
    description: str = ""
    start: tuple[int, int]
    rows: list[str]

    @model_validator(mode="after")
    def check_start(self) -> "GameMap":
        """Стартова клітинка має бути на карті і вільна від перешкод."""
        if self.is_blocked(*self.start):
            raise ValueError(f"Стартова клітинка {self.start} поза картою або заблокована")
        return self

    @field_validator("rows")
    @classmethod
    def check_rows(cls, rows: list[str]) -> list[str]:
        """Перевіряє, що карта не порожня, прямокутна і складається з відомих символів."""
        if not rows or not rows[0]:
            raise ValueError("Карта не може бути порожньою")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("Усі рядки карти мають бути однакової довжини")
        unknown = set("".join(rows)) - {cell.value for cell in Cell}
        if unknown:
            raise ValueError(f"Невідомі символи на карті: {sorted(unknown)}")
        return rows

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    def in_bounds(self, x: int, y: int) -> bool:
        """Чи лежить клітинка всередині карти."""
        return 0 <= x < self.width and 0 <= y < self.height

    def cell(self, x: int, y: int) -> Cell | None:
        """Тип клітинки або None, якщо координати поза картою."""
        if not self.in_bounds(x, y):
            return None
        return Cell(self.rows[y][x])

    def is_blocked(self, x: int, y: int) -> bool:
        """Чи не можна стати на клітинку: поза картою або блокуюча клітинка."""
        symbol = self.cell(x, y)
        return symbol is None or symbol in BLOCKING

    def edit_cell(self, x: int, y: int, cell: Cell, occupied: tuple[int, int] | None = None) -> Cell:
        """Змінює клітинку за правилами гри (змінює карту на місці) і повертає її попередній тип.

        occupied - клітинка, яку не можна змінювати (де стоїть персонаж).
        ValueError, якщо зміна заборонена; текст помилки бачить LLM, тому він українською.
        """
        if not self.in_bounds(x, y):
            raise ValueError(f"({x}, {y}) поза картою (x 0..{self.width - 1}, y 0..{self.height - 1})")
        if (x, y) == occupied:
            raise ValueError("персонаж стоїть на цій клітинці, її не можна змінити")
        old = self.cell(x, y)
        if old in PROTECTED:
            raise ValueError(f"{old.label} на ({x}, {y}) - постійний рельєф, його не можна змінити")
        if old is cell:
            raise ValueError(f"клітинка ({x}, {y}) вже {cell.label}, змін не потрібно")
        row = self.rows[y]
        self.rows[y] = row[:x] + cell.value + row[x + 1 :]
        return old

    def to_text(self, position: tuple[int, int] | None = None) -> str:
        """Вся карта з підписами координат. Клітинка персонажа позначається @."""
        return self._render(range(self.width), range(self.height), position)

    def surroundings(self, position: tuple[int, int], radius: int) -> str:
        """Околиця персонажа радіусом radius: сітка плюс список монстрів і скарбів поруч."""
        if radius < 1:
            raise ValueError("radius має бути не меншим за 1")
        x0, y0 = position
        # Сітку обрізаємо краями карти, щоб не показувати координати поза картою
        xs = range(max(0, x0 - radius), min(self.width, x0 + radius + 1))
        ys = range(max(0, y0 - radius), min(self.height, y0 + radius + 1))
        grid = self._render(xs, ys, position)

        # Монстри і скарби перелічуються окремо, бо це те, на що гравець може реагувати
        objects = []
        for y in ys:
            for x in xs:
                cell = self.cell(x, y)
                if cell in (Cell.MONSTER, Cell.TREASURE) and (x, y) != position:
                    dx, dy = x - x0, y - y0
                    objects.append(f"- {cell.label} на ({x}, {y}): dx={dx:+d}, dy={dy:+d}")
        if not objects:
            objects = ["- немає"]
        return "\n".join([grid, "Об'єкти поруч:", *objects])

    def _render(self, xs: range, ys: range, position: tuple[int, int] | None) -> str:
        """Малює сітку з координатами. Поза картою - порожньо, позиція персонажа - @."""
        header = "   " + "".join(f"{x:>2}" for x in xs)
        lines = [header]
        for y in ys:
            cells = []
            for x in xs:
                if position == (x, y):
                    cells.append("@")
                else:
                    cell = self.cell(x, y)
                    cells.append(" " if cell is None else cell.value)
            lines.append(f"{y:>2} " + "".join(f"{c:>2}" for c in cells))
        return "\n".join(lines)

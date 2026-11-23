"""Екран гри в терміналі: карта, стан персонажа і чат.

Лише відображення: функції приймають стан і повертають rich-об'єкти, логіки гри тут немає.
Клітинки малюються кольоровими блоками, без символів: колір означає тип клітинки.
"""

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dnd_agent.game.character import Attribute, Character
from dnd_agent.game.world import Cell, GameMap

# Колір фону клітинки за її типом
CELL_COLORS = {
    Cell.FLOOR: "grey23",
    Cell.TREE: "dark_green",
    Cell.WALL: "grey62",
    Cell.WATER: "blue",
    Cell.MONSTER: "red",
    Cell.TREASURE: "gold1",
}
PLAYER_STYLE = "bold white on magenta"
SPEAKER_STYLES = {
    "Гравець": "cyan",
    "GM": "bold yellow",
    "Система": "dim",
    "Інструмент": "magenta",
    "Результат": "dim",
    "Світ": "magenta",
    "Помилка": "bold red",
}

HP_BAR_WIDTH = 20
CHAT_LIMIT = 16


def map_panel(world: GameMap, position: tuple[int, int]) -> Panel:
    """Карта: кожна клітинка - два кольорові символи, персонаж - магентовий блок з @@."""
    lines = [Text("   " + "".join(f"{x:>2}" for x in range(world.width)), style="dim")]
    for y in range(world.height):
        line = Text(f"{y:>2} ")
        for x in range(world.width):
            if (x, y) == position:
                line.append("@@", style=PLAYER_STYLE)
            else:
                line.append("  ", style=f"on {CELL_COLORS[world.cell(x, y)]}")
        lines.append(line)
    return Panel(Group(*lines, _legend()), title=world.name, title_align="left")


def _swatch(style: str, label: str) -> Text:
    """Колірний блок і назва: пояснення одного кольору."""
    return Text.assemble(("  ", style), f" {label}")


def _legend() -> Table:
    """Пояснення кольорів у дві колонки: колір блоку -> назва клітинки."""
    items = [_swatch(f"on {CELL_COLORS[cell]}", cell.label) for cell in Cell]
    items.append(_swatch(PLAYER_STYLE, "персонаж"))
    grid = Table.grid(padding=(0, 3))
    grid.add_column()
    grid.add_column()
    for i in range(0, len(items), 2):
        grid.add_row(items[i], items[i + 1] if i + 1 < len(items) else "")
    return grid


def hp_bar(hp: int, max_hp: int) -> str:
    """Смужка HP: заповнена частина відповідає поточному HP."""
    filled = round(HP_BAR_WIDTH * hp / max_hp)
    return "█" * filled + "░" * (HP_BAR_WIDTH - filled)


def status_panel(character: Character) -> Panel:
    """Стан персонажа: ім'я, HP зі смужкою, позиція і модифікатори атрибутів."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    x, y = character.position
    table.add_row("Ім'я", character.name)
    table.add_row("HP", f"{character.hp}/{character.max_hp}  {hp_bar(character.hp, character.max_hp)}")
    table.add_row("Позиція", f"({x}, {y})")
    for attribute in Attribute:
        table.add_row(attribute.label.capitalize(), f"{character.modifier(attribute):+d}")
    return Panel(table, title="Персонаж", title_align="left")


def chat_panel(messages: list[tuple[str, str]]) -> Panel:
    """Останні репліки: пари (мовець, текст)."""
    text = Text()
    for speaker, message in messages[-CHAT_LIMIT:]:
        text.append(f"{speaker}: ", style=SPEAKER_STYLES.get(speaker, ""))
        text.append(f"{message}\n")
    return Panel(text, title="Чат", title_align="left")


def screen(world: GameMap, character: Character, messages: list[tuple[str, str]]) -> Group:
    """Весь екран: карта і стан поруч, під ними чат."""
    top = Table.grid(expand=True)
    top.add_column()
    top.add_column()
    top.add_row(map_panel(world, character.position), status_panel(character))
    return Group(top, chat_panel(messages))

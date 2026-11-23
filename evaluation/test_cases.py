"""Тест-кейси оцінювання: один хід гравця з очікуваними інструментами і результатом.

Кожен кейс запускається в новій грі з власним станом. Перевіряється: які інструменти
викликала модель, чи спрацювала подія світу, яка позиція і HP після ходу, чи не було зупинки.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """Один кейс: запит гравця і очікуваний результат. None у полі означає «не перевіряється»."""

    id: str
    query: str
    description: str
    expect_tools: frozenset[str] = frozenset()
    expect_position: tuple[int, int] | None = None
    expect_hp: int | None = None
    expect_event: str | None = None
    # Стартова позиція; None - стартова клітинка світу (0, 3)
    start_position: tuple[int, int] | None = None


CASES = [
    EvalCase(
        id="move_two_steps",
        query="Іду на схід два кроки.",
        description="Простий рух без перешкод: має бути інструмент руху і позиція (2, 3).",
        expect_tools=frozenset({"move_character"}),
        expect_position=(2, 3),
    ),
    EvalCase(
        id="attack_roll",
        query="Атакую гобліна мечем!",
        description="Гоблін поруч (3, 2): атака має бути кидком кубиків, а не вигаданим результатом.",
        expect_tools=frozenset({"roll_dice"}),
        start_position=(2, 2),
    ),
    EvalCase(
        id="trap_damage",
        query="Мене вдарила пастка, це 5 шкоди.",
        description="Урон від відомої кількості має змінити HP через modify_hp: 16 -> 11.",
        expect_tools=frozenset({"modify_hp"}),
        expect_hp=11,
    ),
    EvalCase(
        id="blocked_by_tree",
        query="Іду на захід на одну клітинку.",
        description="Дерево блокує рух: персонаж має лишитися на місці (2, 4), хід не ламається.",
        expect_tools=frozenset({"move_character"}),
        expect_position=(2, 4),
        start_position=(2, 4),
    ),
    EvalCase(
        id="monster_attacks",
        query="Огляньмося навколо.",
        description="Монстр поруч на (3, 2) атакує сам: подія світу і кидок, без запиту гравця.",
        expect_tools=frozenset({"roll_dice"}),
        expect_event="world_event",
        start_position=(2, 2),
    ),
]

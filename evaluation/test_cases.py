"""Тест-кейси оцінювання: один хід гравця з очікуваними інструментами і результатом.

Кожен кейс запускається в новій грі з власним станом. Перевіряється: які інструменти
викликала модель (і в якому порядку, якщо порядок важливий), чи спрацювала подія світу,
яка позиція і HP після ходу, чи змінилась карта, чи не було зупинки.

Складність кейсу:
    simple  - одна дія, один інструмент
    medium  - одна мета, але потрібне знання стану або кілька викликів
    complex - ланцюжок різних інструментів, де кожна наступна дія залежить від попередньої
"""

from dataclasses import dataclass
from typing import Literal

Difficulty = Literal["simple", "medium", "complex"]


@dataclass(frozen=True)
class EvalCase:
    """Один кейс: запит гравця і очікуваний результат. None у полі означає «не перевіряється»."""

    id: str
    query: str
    description: str
    difficulty: Difficulty
    expect_tools: frozenset[str] = frozenset()
    # Інструменти, які мають зустрітися в цьому порядку (між ними можуть бути інші виклики)
    expect_sequence: tuple[str, ...] = ()
    expect_position: tuple[int, int] | None = None
    expect_hp: int | None = None
    expect_event: str | None = None
    # Клітинка, яка після ходу має бути підлогою (монстра прибрано)
    expect_floor: tuple[int, int] | None = None
    # Стартова позиція; None - стартова клітинка світу (0, 3)
    start_position: tuple[int, int] | None = None


CASES = [
    EvalCase(
        id="move_two_steps",
        query="Іду на схід два кроки.",
        description="Простий рух без перешкод: має бути інструмент руху і позиція (2, 3).",
        difficulty="simple",
        expect_tools=frozenset({"move_character"}),
        expect_position=(2, 3),
    ),
    EvalCase(
        id="trap_damage",
        query="Мене вдарила пастка, це 5 шкоди.",
        description="Урон від відомої кількості має змінити HP через modify_hp: 16 -> 11.",
        difficulty="simple",
        expect_tools=frozenset({"modify_hp"}),
        expect_hp=11,
    ),
    EvalCase(
        id="attack_roll",
        query="Атакую гобліна мечем!",
        description="Гоблін поруч (3, 2): атака має бути кидком кубиків, а не вигаданим результатом.",
        difficulty="medium",
        expect_tools=frozenset({"roll_dice"}),
        start_position=(2, 2),
    ),
    EvalCase(
        id="blocked_by_tree",
        query="Іду на захід на одну клітинку.",
        description="Дерево блокує рух: персонаж має лишитися на місці (2, 4), хід не ламається.",
        difficulty="medium",
        expect_tools=frozenset({"move_character"}),
        expect_position=(2, 4),
        start_position=(2, 4),
    ),
    EvalCase(
        id="monster_attacks",
        query="Огляньмося навколо.",
        description="Монстр поруч на (3, 2) атакує сам: подія світу і кидок, без запиту гравця.",
        difficulty="medium",
        expect_tools=frozenset({"roll_dice"}),
        expect_event="world_event",
        start_position=(2, 2),
    ),
    EvalCase(
        id="walk_to_treasure",
        query="Іди на схід до скарбу.",
        description="До скарбу на (6, 3) шість клітинок, а за один виклик можна максимум п'ять: "
        "потрібні два виклики руху поспіль.",
        difficulty="medium",
        expect_tools=frozenset({"move_character"}),
        expect_sequence=("move_character", "move_character"),
        expect_position=(6, 3),
    ),
    EvalCase(
        id="approach_attack_remove",
        query="Підійди до гобліна, атакуй його мечем і, якщо переміг, прибери його з карти. "
        "Гоблін слабкий: будь-який кидок атаки його вбиває.",
        description="Ланцюжок із трьох залежних дій: рух до гобліна на (3, 2), кидок атаки, "
        "потім modify_map прибирає гобліна. Гоблін блокує рух, тож персонаж зупиняється на (4, 2).",
        difficulty="complex",
        expect_tools=frozenset({"move_character", "roll_dice", "modify_map"}),
        expect_sequence=("move_character", "roll_dice", "modify_map"),
        expect_position=(4, 2),
        expect_floor=(3, 2),
        start_position=(5, 2),
    ),
]

"""Події, які починає сам світ, а не гравець: монстр поруч атакує персонажа.

Тут лише правило, хто поруч. Числа атаки і текст для LLM живуть окремо.
"""

from dnd_agent.game.character import Character
from dnd_agent.game.world import Cell, Direction, GameMap


def monsters_next_to(character: Character, world: GameMap) -> list[tuple[int, int]]:
    """Координати монстрів на сусідніх клітинках (по горизонталі і вертикалі).

    Мертвий персонаж уже не в бою, тож для нього список порожній.
    """
    if not character.is_alive:
        return []
    x, y = character.position
    found = []
    for direction in Direction:
        dx, dy = direction.delta
        if world.cell(x + dx, y + dy) == Cell.MONSTER:
            found.append((x + dx, y + dy))
    return found

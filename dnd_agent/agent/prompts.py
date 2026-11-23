"""Промпт гри: системна інструкція і актуальний стан, який додається до кожного виклику LLM.

Стан (HP, позиція, карта) не зберігається в історії повідомлень: він будується заново
перед кожним викликом моделі, тож модель завжди бачить поточну картину світу.
"""

from langchain_core.messages import BaseMessage, SystemMessage

from dnd_agent.agent.state import AgentState
from dnd_agent.game.character import Attribute

SYSTEM_PROMPT = """Ти - майстер гри (GM) у настільній грі D&D. Веди гру українською.

Правила:
- Усі факти про світ (кубики, HP, позиція, карта) змінюються тільки через інструменти.
  Ніколи не вигадуй результат кидка чи зміну стану сам.
- Перед дією подивися на стан нижче: позицію, HP і карту.
- Рухай персонажа інструментом move_character. Викликай по одному інструменту за раз.
- Після результату інструмента коротко опиши сцену гравцеві.
- Якщо інструмент повернув помилку, виправ виклик або поясни гравцю, чому дія неможлива.
- Монстри блокують рух: спершу переможи або обійди їх (кидок, зміна карти).
"""


def render_state(state: AgentState) -> str:
    """Блок стану гри для промпту: персонаж, атрибути і карта з позицією."""
    character = state["character"]
    world = state["world"]
    x, y = character.position
    attributes = ", ".join(f"{attr.label} {character.modifier(attr):+d}" for attr in Attribute)
    return "\n".join(
        [
            "## Стан гри",
            f"Персонаж: {character.name}. HP {character.hp}/{character.max_hp}. Позиція ({x}, {y}).",
            f"Атрибути: {attributes}.",
            f"Світ: {world.name}. {world.description}",
            "",
            "Легенда карти: . підлога, T дерево, # стіна, W вода, M монстр, $ скарб, @ персонаж.",
            "Координати: x - стовпець, y - рядок; відлік від 0.",
            world.to_text(character.position),
        ]
    )


def build_messages(state: AgentState) -> list[BaseMessage]:
    """Повідомлення для LLM: системна інструкція зі станом плюс історія розмови."""
    system = SystemMessage(content=f"{SYSTEM_PROMPT}\n{render_state(state)}")
    return [system, *state["messages"]]

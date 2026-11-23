"""Стан графа: історія повідомлень, персонаж, карта, лічильник кроків і відповідь за хід.

Персонаж і карта живуть тут, а не в тексті розмови. Інструменти оновлюють їх
через Command, checkpointer зберігає їх між ходами гравця.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from dnd_agent.agent.schemas import GMResponse
from dnd_agent.game.character import Character
from dnd_agent.game.world import GameMap


class AgentState(TypedDict):
    """Стан гри: повідомлення чату, персонаж, карта, кількість викликів моделі і відповідь за хід."""

    messages: Annotated[list[AnyMessage], add_messages]
    character: Character
    world: GameMap
    step_count: int
    # Момент (time.monotonic), після якого хід зупиняється; ставиться на першому виклику моделі в ході
    turn_deadline: float | None
    response: GMResponse | None

"""Вузли графа: agent (LLM вирішує, що робити), world_events (світ сам діє), formatter (структурована відповідь)."""

import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from dnd_agent.agent.prompts import build_messages
from dnd_agent.agent.schemas import GMAnswer, GMResponse
from dnd_agent.agent.state import AgentState
from dnd_agent.constants import MONSTER_ATTACK_DC, MONSTER_DAMAGE_DICE, WORLD_EVENT
from dnd_agent.game.events import monsters_next_to
from dnd_agent.safety.limits import LIMIT_REACHED, STOP_MESSAGE, limit_reached, model_calls_this_turn
from dnd_agent.safety.loop_detector import LOOP, LoopDetector, loop_message
from dnd_agent.safety.timeout import TIMEOUT, TIMEOUT_MESSAGE, TURN_TIMEOUT_SECONDS, deadline_passed

FORMAT_REQUEST = "Оформи останню відповідь GM у структуру: answer - розповідь гравцю, confidence - впевненість."
LOOP_DETECTOR = LoopDetector()


def make_agent_node(model: BaseChatModel, turn_timeout: float = TURN_TIMEOUT_SECONDS):
    """Створює вузол agent: модель отримує актуальний стан і відповідає або викликом інструмента, або текстом.

    Перед кожним викликом перевіряє ліміт кроків і дедлайн ходу. Дедлайн ставиться на першому виклику ходу.
    """

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if limit_reached(messages):
            # Ліміт вичерпано: модель не викликаємо, лише завершуємо хід
            return {"messages": [SystemMessage(content=STOP_MESSAGE, name=LIMIT_REACHED)]}
        repeated = LOOP_DETECTOR.find_loop(messages)
        if repeated:
            # Та сама дія вже повторювалась: модель не просимо знову, хід зупиняємо
            text = loop_message(repeated, LOOP_DETECTOR.max_repeats)
            return {"messages": [SystemMessage(content=text, name=LOOP)]}
        update = {}
        if model_calls_this_turn(messages) == 0:
            # Перший виклик ходу: від цього моменту рахується час ходу
            update["turn_deadline"] = time.monotonic() + turn_timeout
        elif deadline_passed(state.get("turn_deadline")):
            return {"messages": [SystemMessage(content=TIMEOUT_MESSAGE, name=TIMEOUT)]}
        response = model.invoke(build_messages(state))
        return {**update, "messages": [response], "step_count": state.get("step_count", 0) + 1}

    return agent_node


def _this_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Повідомлення після останнього повідомлення гравця: поточний хід."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    return messages[last_human + 1 :]


def should_continue(state: AgentState) -> Literal["tools", "world_events"]:
    """Якщо модель попросила інструмент - йдемо до tools, інакше даємо світу шанс діяти."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "world_events"


def world_events_node(state: AgentState) -> dict:
    """Світ діє сам: монстр поруч атакує персонажа. Не більше однієї події за хід гравця.

    Подію визначає код, а не LLM. LLM отримує її як повідомлення і розв'язує інструментами.
    """
    messages = state["messages"]
    if limit_reached(messages) or deadline_passed(state.get("turn_deadline")) or LOOP_DETECTOR.find_loop(messages):
        return {}
    if any(getattr(m, "name", None) == WORLD_EVENT for m in _this_turn(state["messages"])):
        return {}
    monsters = monsters_next_to(state["character"], state["world"])
    if not monsters:
        return {}
    where = ", ".join(f"({x}, {y})" for x, y in monsters)
    text = (
        f"[Подія світу] Монстри поруч атакують персонажа: {where}. Кожен монстр атакує окремо. "
        f"Кинь roll_dice з dice='1d20' і purpose='атака монстра'. "
        f"Якщо сума менша за {MONSTER_ATTACK_DC} - промах, опиши це. "
        f"Якщо не менша - влучання: кинь roll_dice з dice='{MONSTER_DAMAGE_DICE}' "
        f"і застосуй урон через modify_hp з від'ємним delta."
    )
    return {"messages": [SystemMessage(content=text, name=WORLD_EVENT)]}


def after_events(state: AgentState) -> Literal["agent", "formatter"]:
    """Якщо світ щойно додав подію - GM має її розв'язати. Інакше формуємо відповідь."""
    last = state["messages"][-1]
    return "agent" if getattr(last, "name", None) == WORLD_EVENT else "formatter"


def tools_used_this_turn(messages: list[AnyMessage]) -> list[str]:
    """Назви інструментів, викликаних після останнього повідомлення гравця."""
    return [
        call["name"]
        for message in _this_turn(messages)
        for call in getattr(message, "tool_calls", None) or []
    ]


def make_formatter_node(model: BaseChatModel):
    """Створює вузол formatter: перетворює останню відповідь на GMResponse.

    Запит до моделі не додається в історію чату, тож розмова не засмічується.
    Інструменти тут не прив'язуються: форматер лише оформлює текст.
    """
    structured = model.with_structured_output(GMAnswer)

    def formatter_node(state: AgentState) -> dict:
        messages = [*build_messages(state), HumanMessage(content=FORMAT_REQUEST)]
        answer = structured.invoke(messages)
        return {
            "response": GMResponse(
                **answer.model_dump(),
                sources=tools_used_this_turn(state["messages"]),
            )
        }

    return formatter_node

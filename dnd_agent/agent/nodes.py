"""Вузли графа: agent (LLM вирішує, що робити), маршрутизатор і formatter (структурована відповідь)."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage

from dnd_agent.agent.prompts import build_messages
from dnd_agent.agent.schemas import GMAnswer, GMResponse
from dnd_agent.agent.state import AgentState

FORMAT_REQUEST = "Оформи останню відповідь GM у структуру: answer - розповідь гравцю, confidence - впевненість."


def make_agent_node(model: BaseChatModel):
    """Створює вузол agent: модель отримує актуальний стан і відповідає або викликом інструмента, або текстом."""

    def agent_node(state: AgentState) -> dict:
        response = model.invoke(build_messages(state))
        return {"messages": [response], "step_count": state.get("step_count", 0) + 1}

    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "formatter"]:
    """Якщо модель попросила інструмент - йдемо до tools, інакше до formatter."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "formatter"


def tools_used_this_turn(messages: list[AnyMessage]) -> list[str]:
    """Назви інструментів, викликаних після останнього повідомлення гравця."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    return [
        call["name"]
        for message in messages[last_human + 1 :]
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

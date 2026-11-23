"""Збирає граф ReAct: START -> agent -> (tools -> agent)* -> formatter -> END.

build_graph приймає модель і checkpointer параметрами: у грі це ChatOpenAI і MemorySaver,
а в тестах можна підставити скриптовану модель без мережі.
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from dnd_agent.agent.nodes import make_agent_node, make_formatter_node, should_continue
from dnd_agent.agent.state import AgentState
from dnd_agent.config import MODEL_NAME
from dnd_agent.tools import ALL_TOOLS

# Власні типи стану дозволяємо серіалізувати checkpointer'у явно
SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("dnd_agent.game.character", "Character"),
        ("dnd_agent.game.character", "Attribute"),
        ("dnd_agent.game.world", "GameMap"),
        ("dnd_agent.agent.schemas", "GMResponse"),
    ]
)


def build_graph(model: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None):
    """Компілює граф. Без параметрів використовує ChatOpenAI і збереження в пам'яті."""
    base = model or ChatOpenAI(model=MODEL_NAME)
    # Один інструмент за раз: кожен виклик має бачити оновлений стан попереднього
    bound = base.bind_tools(ALL_TOOLS, parallel_tool_calls=False)

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(bound))
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("formatter", make_formatter_node(base))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver(serde=SERDE))

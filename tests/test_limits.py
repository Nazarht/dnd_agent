"""Тести ліміту кроків: підрахунок ходу, межа і зупинка зациклення."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from dnd_agent.safety.demo import run_looping_turn
from dnd_agent.safety.limits import (
    LIMIT_REACHED,
    MAX_MODEL_CALLS_PER_TURN,
    STOP_MESSAGE,
    limit_reached,
    model_calls_this_turn,
)


def turn_with_model_calls(count: int) -> list:
    """Історія: повідомлення гравця, потім count відповідей моделі з інструментами між ними."""
    messages = [HumanMessage(content="дія")]
    for i in range(count):
        messages.append(AIMessage(content="", tool_calls=[{"name": "roll_dice", "args": {}, "id": str(i)}]))
        messages.append(ToolMessage(content="ok", tool_call_id=str(i)))
    return messages


def test_counts_only_current_turn():
    messages = turn_with_model_calls(2) + [HumanMessage(content="нова дія"), AIMessage(content="текст")]
    assert model_calls_this_turn(messages) == 1


def test_tool_messages_are_not_model_calls():
    assert model_calls_this_turn(turn_with_model_calls(3)) == 3


def test_limit_is_reached_exactly_at_max():
    assert not limit_reached(turn_with_model_calls(MAX_MODEL_CALLS_PER_TURN - 1))
    assert limit_reached(turn_with_model_calls(MAX_MODEL_CALLS_PER_TURN))


def test_looping_turn_stops_at_limit():
    out = run_looping_turn()
    assert model_calls_this_turn(out["messages"]) == MAX_MODEL_CALLS_PER_TURN
    stop = out["messages"][-1]
    assert isinstance(stop, SystemMessage)
    assert stop.name == LIMIT_REACHED
    assert stop.content == STOP_MESSAGE
    assert out["response"].answer  # формувач відповіді все одно відпрацював

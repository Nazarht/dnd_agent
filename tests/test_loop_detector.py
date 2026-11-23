"""Тести детектора зациклення: повтори в ході, виключення roll_dice і зупинка в графі."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from dnd_agent.safety.demo import LOOP_ARGS, run_repeated_turn
from dnd_agent.safety.loop_detector import LOOP, LoopDetector


def call(name: str, args: dict, index: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{index}", "type": "tool_call"}])


def turn_of(*calls: AIMessage) -> list:
    """Історія: повідомлення гравця, далі виклики моделі з відповідями інструментів."""
    messages = [HumanMessage(content="дія")]
    for i, message in enumerate(calls):
        messages.append(message)
        messages.append(ToolMessage(content="ok", tool_call_id=f"c{i}"))
    return messages


def test_distinct_calls_are_not_a_loop():
    detector = LoopDetector()
    messages = turn_of(call("move_character", {"direction": "east", "steps": 1}, 0),
                       call("move_character", {"direction": "east", "steps": 2}, 1),
                       call("modify_hp", {"delta": -1, "reason": "удар"}, 2))
    assert detector.find_loop(messages) is None


def test_repeated_call_is_a_loop_at_threshold():
    detector = LoopDetector(max_repeats=3)
    messages = turn_of(*[call("move_character", LOOP_ARGS, i) for i in range(3)])
    assert detector.find_loop(messages) == 'move_character({"direction": "west", "steps": 1})'


def test_two_repeats_are_below_threshold():
    messages = turn_of(*[call("move_character", LOOP_ARGS, i) for i in range(2)])
    assert LoopDetector(max_repeats=3).find_loop(messages) is None


def test_roll_dice_is_exempt():
    roll = {"dice": "1d20", "attribute": "strength", "modifier": 0, "purpose": "атака"}
    messages = turn_of(*[call("roll_dice", roll, i) for i in range(10)])
    assert LoopDetector().find_loop(messages) is None


def test_repeats_from_previous_turn_do_not_count():
    old_turn = turn_of(*[call("move_character", LOOP_ARGS, i) for i in range(2)])
    messages = old_turn + [HumanMessage(content="нова дія"), call("move_character", LOOP_ARGS, 9)]
    assert LoopDetector().find_loop(messages) is None


def test_looping_turn_stops_the_graph():
    out = run_repeated_turn("move_character", LOOP_ARGS, times=10)
    stop = out["messages"][-1]
    assert isinstance(stop, SystemMessage)
    assert stop.name == LOOP
    # Три виклики виконано, четвертий до моделі вже не доходить
    assert sum(1 for m in out["messages"] if isinstance(m, ToolMessage)) == 3

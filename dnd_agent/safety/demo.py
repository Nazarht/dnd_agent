"""Демо захисників: ліміт кроків, таймаут ходу і детектор зациклення.

Моделі фейкові (без API), тож демо детерміновані. recursion_limit у графі - запасний
захист від нескінченного циклу, якщо сам захисник не спрацює.
"""

import time

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from dnd_agent.agent.graph import SERDE, build_graph
from dnd_agent.game.loader import load_character, load_world
from dnd_agent.safety.limits import MAX_MODEL_CALLS_PER_TURN, model_calls_this_turn
from dnd_agent.safety.loop_detector import LOOP
from dnd_agent.safety.timeout import TIMEOUT

console = Console()
# Запасний ліміт графу: значно більший за ліміт ходу, але скінченний
GRAPH_RECURSION_LIMIT = 100
# Для демо таймауту: короткий хід і повільна модель
DEMO_TURN_TIMEOUT = 0.5
DEMO_MODEL_DELAY = 0.2
# Для демо зациклення: одна і та сама дія без кінця
LOOP_ARGS = {"direction": "west", "steps": 1}


class ScriptedModel(FakeMessagesListChatModel):
    """Фейкова модель: відповіді заздалегідь задані, structured output теж фейковий."""

    delay: float = 0.0

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: schema(answer="GM зупинив хід", confidence=0.5))

    def invoke(self, input, config=None, **kwargs):
        # Затримка імітує повільний запит до моделі
        time.sleep(self.delay)
        return super().invoke(input, config, **kwargs)


def _call(name: str, args: dict, index: int) -> AIMessage:
    """Виклик інструмента з власним id, щоб повідомлення не злипались в історії."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{index}", "type": "tool_call"}])


def _roll_request(index: int) -> AIMessage:
    return _call("roll_dice", {"dice": "1d20", "attribute": "strength", "modifier": 0, "purpose": "ще один кидок"}, index)


def _run(model: ScriptedModel, turn_timeout: float, text: str = "Кидай кубики, поки не скажу зупинитися.") -> dict:
    """Один хід із заданою моделлю і таймаутом. Повертає фінальний стан графа."""
    world = load_world("forest")
    character = load_character("priest", world.start)
    app = build_graph(model=model, checkpointer=MemorySaver(serde=SERDE), turn_timeout=turn_timeout)
    config = {"configurable": {"thread_id": "safety-demo"}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    payload = {
        "messages": [HumanMessage(content=text)],
        "character": character,
        "world": world,
        "step_count": 0,
    }
    return app.invoke(payload, config)


def run_looping_turn() -> dict:
    """Хід, у якому модель хоче кидати кубики без кінця: зупиняє ліміт кроків."""
    # Відповідей більше, ніж дозволяє ліміт: якщо ліміт не спрацює, модель піде далі
    model = ScriptedModel(responses=[_roll_request(i) for i in range(MAX_MODEL_CALLS_PER_TURN * 3)])
    return _run(model, turn_timeout=60.0)


def run_timeout_turn() -> tuple[dict, float]:
    """Хід із повільною моделлю і коротким таймаутом: зупиняє дедлайн ходу. Повертає стан і час."""
    model = ScriptedModel(
        responses=[_roll_request(i) for i in range(MAX_MODEL_CALLS_PER_TURN * 3)],
        delay=DEMO_MODEL_DELAY,
    )
    started = time.monotonic()
    out = _run(model, turn_timeout=DEMO_TURN_TIMEOUT)
    return out, time.monotonic() - started


def run_repeated_turn(name: str, args: dict, times: int) -> dict:
    """Хід, у якому модель повторює один і той самий виклик times разів, потім відповідає текстом."""
    responses = [_call(name, args, i) for i in range(times)]
    responses.append(AIMessage(content="Досить."))
    return _run(ScriptedModel(responses=responses), turn_timeout=60.0, text="Повтори дію кілька разів.")


def _stop_name(messages) -> str | None:
    """Ім'я системного повідомлення-зупинки, якщо воно є в історії."""
    for message in messages:
        if getattr(message, "name", None) in {"limit", TIMEOUT, LOOP}:
            return message.name
    return None


def run_safety_demo() -> None:
    """Друкує результат трьох демо: ліміт кроків, таймаут ходу і детектор зациклення."""
    out = run_looping_turn()
    messages = out["messages"]
    tool_results = sum(1 for m in messages if isinstance(m, ToolMessage))
    console.rule("Демо 1: ліміт кроків")
    console.print(f"Ліміт викликів моделі за хід: {MAX_MODEL_CALLS_PER_TURN}")
    console.print(f"Модель викликана: {model_calls_this_turn(messages)} разів")
    console.print(f"Інструментів виконано: {tool_results}")
    console.print(f"Зупинка: {_stop_name(messages)}")
    console.print(f"Підсумок GM: {out['response'].answer}")

    out, elapsed = run_timeout_turn()
    messages = out["messages"]
    console.rule("Демо 2: таймаут ходу")
    console.print(f"Таймаут ходу: {DEMO_TURN_TIMEOUT} с, затримка моделі: {DEMO_MODEL_DELAY} с на виклик")
    console.print(f"Модель викликана: {model_calls_this_turn(messages)} разів (ліміт кроків не досягнуто)")
    console.print(f"Зупинка: {_stop_name(messages)}, час ходу: {elapsed:.2f} с")
    console.print(f"Підсумок GM: {out['response'].answer}")

    out = run_repeated_turn("move_character", LOOP_ARGS, times=10)
    messages = out["messages"]
    console.rule("Демо 3: детектор зациклення")
    console.print("Модель без кінця просить move_character з тими самими аргументами")
    console.print(f"Модель викликана: {model_calls_this_turn(messages)} разів")
    console.print(f"Зупинка: {_stop_name(messages)}")

    out = run_repeated_turn("roll_dice", {"dice": "1d20", "attribute": "strength", "modifier": 0, "purpose": "атака"}, times=4)
    messages = out["messages"]
    console.print("Для порівняння: 4 однакові кидки кубиків (roll_dice виключено з перевірки)")
    console.print(f"Модель викликана: {model_calls_this_turn(messages)} разів")
    console.print(f"Зупинка: {_stop_name(messages) or 'немає, хід завершено нормально'}")

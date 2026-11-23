"""Headless перевірка агента: реальна модель, без UI.

Надсилає сценарій ходів гравця і друкує: виклики інструментів, їх результати,
зміни стану (HP, позиція, карта) і відповідь GM. Так видно, чи модель обирає
правильні інструменти, ще до побудови UI і тестів.
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from dnd_agent.agent.graph import SERDE, build_graph
from dnd_agent.constants import WORLD_EVENT
from dnd_agent.game.loader import load_character, load_world

console = Console()


@dataclass(frozen=True)
class Turn:
    """Один хід гравця і інструменти, які модель має викликати (порожньо - без перевірки)."""

    text: str
    expect: frozenset[str] = frozenset()


SCENARIO = [
    Turn("Огляньмося навколо."),
    Turn("Іду на схід два кроки.", frozenset({"move_character"})),
    Turn("Атакую гобліна мечем!", frozenset({"roll_dice"})),
    Turn("Гоблін переможений. Прибери його з карти.", frozenset({"modify_map"})),
    Turn("Іду на схід до скарбу.", frozenset({"move_character"})),
    Turn("Мене вдарила пастка, це 5 шкоди.", frozenset({"modify_hp"})),
    Turn("Стрибаю на 10 клітинок на захід.", frozenset({"move_character"})),
]


def _turn_messages(messages):
    """Повідомлення поточного ходу: від останнього повідомлення гравця до кінця."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    return messages[last_human:]


def _print_state_changes(before: dict, after: dict) -> None:
    """Друкує лише те, що змінилось: HP, позиція і клітинки карти."""
    old_char, new_char = before["character"], after["character"]
    if old_char.hp != new_char.hp:
        console.print(f"  [cyan]HP {old_char.hp} -> {new_char.hp}[/cyan]")
    if old_char.position != new_char.position:
        console.print(f"  [cyan]позиція {old_char.position} -> {new_char.position}[/cyan]")
    for y, (old_row, new_row) in enumerate(zip(before["world"].rows, after["world"].rows)):
        for x, (old, new) in enumerate(zip(old_row, new_row)):
            if old != new:
                console.print(f"  [cyan]карта ({x}, {y}): {old} -> {new}[/cyan]")


def _print_turn(number: int, turn: Turn, out: dict, before: dict) -> list[str]:
    """Друкує один хід. Повертає назви всіх викликів інструментів у ході."""
    console.rule(f"Хід {number}")
    console.print(f"[bold]Гравець:[/bold] {turn.text}")

    calls: list[str] = []
    for message in _turn_messages(out["messages"]):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append(call["name"])
                console.print(f"  [green]→ {call['name']}[/green]({call['args']})")
        elif isinstance(message, ToolMessage):
            style = "red" if message.status == "error" else "dim"
            first_line = message.content.splitlines()[0]
            console.print(f"  [{style}]← {first_line}[/{style}]")
        elif isinstance(message, SystemMessage) and message.name == WORLD_EVENT:
            # Подію починає світ, а не гравець: позначаємо, щоб було видно в звіті
            console.print(f"  [magenta]⚡ {message.content.splitlines()[0]}[/magenta]")

    _print_state_changes(before, out)

    response = out["response"]
    console.print(f"[bold]GM:[/bold] {response.answer}")
    console.print(f"  [dim]впевненість {response.confidence:.2f}, інструменти {response.sources or 'немає'}[/dim]")

    if turn.expect:
        called = set(calls)
        if turn.expect <= called:
            console.print(f"  [green]✓ очікувані інструменти викликано: {', '.join(sorted(turn.expect))}[/green]")
        else:
            console.print(
                f"  [yellow]✗ очікувалось {', '.join(sorted(turn.expect))}, "
                f"викликано {', '.join(sorted(called)) or 'нічого'}[/yellow]"
            )
    return calls


def run_tryout(scenario: list[Turn] = SCENARIO) -> None:
    """Прогоняє сценарій через граф з реальною моделлю і друкує звіт."""
    app = build_graph(checkpointer=MemorySaver(serde=SERDE))
    config = {"configurable": {"thread_id": "tryout"}}
    world = load_world("forest")
    character = load_character("priest", world.start)

    before = {"character": character, "world": world}
    total_calls = 0
    for number, turn in enumerate(scenario, start=1):
        payload = {"messages": [HumanMessage(turn.text)]}
        if number == 1:
            payload.update(character=character, world=world, step_count=0)
        out = app.invoke(payload, config)
        total_calls += len(_print_turn(number, turn, out, before))  # кожен виклик окремо
        before = out
    console.rule("Підсумок")
    console.print(f"Ходів: {len(scenario)}, викликів інструментів: {total_calls}")

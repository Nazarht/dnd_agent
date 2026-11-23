"""Цикл гри в терміналі: ввід гравця, слеш-команди і live-оновлення ходу.

Хід іде через граф у режимі stream: інструменти, їх результати, події світу і
відповідь GM з'являються в чаті одразу, а карта і стан перемальовуються.
Граф передається ззовні, тож сесію можна перевірити без мережі.
"""

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from dnd_agent.agent.graph import build_graph
from dnd_agent.agent.prompts import render_state
from dnd_agent.audit.trajectory import TrajectoryLogger
from dnd_agent.constants import WORLD_EVENT
from dnd_agent.game.loader import load_character, load_world
from dnd_agent.ui.render import screen

console = Console()
COMMANDS_HELP = "Команди: /map - карта для GM, /state - стан для GM, /new - нова гра, /quit - вихід."


class Session:
    """Одна гра: поточний персонаж, карта, чат і потік розмови в checkpointer."""

    def __init__(self, world_name: str, character_name: str, app, logger: TrajectoryLogger):
        self.world_name = world_name
        self.character_name = character_name
        self.app = app
        self.logger = logger
        self.new_game()

    def new_game(self) -> None:
        """Нова гра: файли перечитуються, розмова починається з нового потоку, новий запуск для траєкторії."""
        self.logger.start_run()
        self.world = load_world(self.world_name)
        self.character = load_character(self.character_name, self.world.start)
        self.thread_id = str(uuid4())
        self.started = False
        self.chat = [("Система", "Нова гра. " + COMMANDS_HELP)]

    def view(self, busy: str | None = None) -> Group:
        """Екран: карта, стан і чат. busy - рядок 'думає...' під час ходу."""
        parts = [screen(self.world, self.character, self.chat)]
        if busy:
            parts.append(Text(busy, style="dim italic"))
        return Group(*parts)

    def show(self) -> None:
        console.clear()
        console.print(self.view())

    def play_turn(self, text: str) -> None:
        """Ведe один хід: кожна подія графа одразу оновлює чат, екран і траєкторію."""
        self.chat.append(("Гравець", text))
        payload = {"messages": [HumanMessage(content=text)]}
        if not self.started:
            # Перший хід передає початковий стан; далі граф бере його з checkpointer
            payload.update(character=self.character, world=self.world, step_count=0)
            self.started = True
        config = {"configurable": {"thread_id": self.thread_id}}

        self.logger.start_turn(text)
        console.clear()
        try:
            with Live(self.view(busy="GM думає..."), console=console, auto_refresh=False) as live:
                for update in self.app.stream(payload, config, stream_mode="updates"):
                    for node, values in update.items():
                        self.logger.record(node, values)
                        self._apply(values)
                    live.update(self.view(busy="GM думає..."), refresh=True)
                live.update(self.view(), refresh=True)
        except Exception as error:
            # Частковий хід теж записуємо: видно, на чому він обірвався
            self.logger.end_turn(self.app.get_state(config).values, error=repr(error))
            raise
        self.logger.end_turn(self.app.get_state(config).values)

    def _apply(self, values: dict) -> None:
        """Переносить оновлення вузла в стан сесії і в чат. Порожнє оновлення (None або {}) пропускаємо."""
        if not values:
            return
        if "character" in values:
            self.character = values["character"]
        if "world" in values:
            self.world = values["world"]
        for message in values.get("messages", []):
            self._log_message(message)
        if "response" in values:
            self.chat.append(("GM", values["response"].answer))

    def _log_message(self, message) -> None:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                self.chat.append(("Інструмент", f"{call['name']}({call['args']})"))
        elif isinstance(message, ToolMessage):
            speaker = "Помилка" if message.status == "error" else "Результат"
            self.chat.append((speaker, message.content.splitlines()[0]))
        elif isinstance(message, SystemMessage) and message.name == WORLD_EVENT:
            self.chat.append(("Світ", message.content.splitlines()[0]))

    def command(self, text: str) -> bool:
        """Виконує слеш-команду. Повертає False, якщо гра має завершитися."""
        if text == "/quit":
            return False
        if text == "/new":
            self.new_game()
            self.show()
        elif text == "/map":
            self.show()
            console.print(Panel(self.world.to_text(self.character.position), title="Карта для GM"))
        elif text == "/state":
            self.show()
            state = {"character": self.character, "world": self.world}
            console.print(Panel(render_state(state), title="Стан для GM"))
        else:
            self.show()
            console.print(f"Невідома команда {text}. {COMMANDS_HELP}")
        return True


def run_play(world_name: str, character_name: str) -> None:
    """Запускає гру в терміналі. Вихід - /quit, Ctrl+C або Ctrl+D."""
    logger = TrajectoryLogger(command="play")
    session = Session(world_name, character_name, build_graph(), logger)
    session.show()
    try:
        while True:
            try:
                text = console.input("[bold cyan]> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not text:
                continue
            if text.startswith("/"):
                if not session.command(text):
                    break
            else:
                session.play_turn(text)
    finally:
        logger.finish_run()

"""Траєкторія запуску: що робив агент у кожному ході. Файл outputs/{run_id}/trajectory.json.

Логер отримує оновлення вузлів графу (stream) і зберігає виклики моделі та інструментів,
їх результати, події світу, зміни HP і позиції, відповідь GM і причину зупинки.
Файл переписується після кожного ходу, тож видно все, що встигло відбутися.
"""

from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from dnd_agent.config import OUTPUTS_DIR
from dnd_agent.safety.limits import LIMIT_REACHED, model_calls_this_turn
from dnd_agent.safety.loop_detector import LOOP
from dnd_agent.safety.timeout import TIMEOUT
from dnd_agent.utils.json_io import save_json
from dnd_agent.utils.run_context import RunContext
from dnd_agent.utils.timing import Stopwatch, utc_now_iso

TRAJECTORY_FILE = "trajectory.json"
# Імена системних повідомлень, якими позначається зупинка ходу
STOP_NAMES = {LIMIT_REACHED, TIMEOUT, LOOP}


def plain_text(content) -> str:
    """Текст повідомлення: рядок як є, або склеєні текстові блоки (формат нових відповідей моделі)."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


class TrajectoryLogger:
    """Збирає траєкторію по ходах і зберігає її в папку поточного запуску."""

    def __init__(self, command: str = "play", outputs_dir: Path = OUTPUTS_DIR) -> None:
        self.command = command
        self.outputs_dir = outputs_dir
        self.run: RunContext | None = None
        self.turns: list[dict] = []
        self._turn: dict | None = None
        self._watch: Stopwatch | None = None

    def start_run(self) -> None:
        """Новий запуск: власна папка outputs/{run_id}, порожня траєкторія. Попередній запуск закривається."""
        self.finish_run()
        self.run = RunContext(self.command, self.outputs_dir)
        self.turns = []

    def finish_run(self, **fields) -> None:
        """Закриває поточний запуск у run_meta.json; fields - підсумки (наприклад, результати тестів)."""
        if self.run is not None:
            self.run.finish(turns=len(self.turns), **fields)
            self.run = None

    def start_turn(self, text: str, label: str | None = None) -> None:
        """Початок ходу: фіксуємо репліку гравця і запускаємо секундомір ходу.

        label - назва сценарію (наприклад, id тест-кейсу), щоб ходи можна було знайти в файлі.
        """
        self._turn = {
            "turn": len(self.turns) + 1,
            "label": label,
            "player": text,
            "started_at": utc_now_iso(),
            "steps": [],
        }
        self._watch = Stopwatch()

    def record(self, node: str, values: dict | None) -> None:
        """Записує крок вузла: виклики, результати, подію світу, стан і відповідь. Порожні оновлення пропускаємо."""
        if self._turn is None or not values:
            return
        step: dict = {"node": node, "at_ms": self._watch.elapsed_ms()}
        for message in values.get("messages", []):
            if isinstance(message, AIMessage) and message.tool_calls:
                step.setdefault("tool_calls", []).extend(
                    {"name": call["name"], "args": call["args"]} for call in message.tool_calls
                )
            elif isinstance(message, AIMessage) and plain_text(message.content):
                step["text"] = plain_text(message.content)
            elif isinstance(message, ToolMessage):
                step.setdefault("tool_results", []).append(
                    {"tool": message.name, "status": message.status, "content": message.content}
                )
            elif isinstance(message, SystemMessage) and message.name:
                step["event"] = {"kind": message.name, "text": message.content}
        if "character" in values:
            step["hp"] = values["character"].hp
            step["position"] = list(values["character"].position)
        if "response" in values:
            response = values["response"]
            step["answer"] = response.answer
            step["confidence"] = response.confidence
            step["sources"] = response.sources
        self._turn["steps"].append(step)

    def end_turn(self, state: dict, error: str | None = None) -> None:
        """Кінець ходу: підсумки (виклики, зупинка, час, фінальний стан) і збереження файлу."""
        if self._turn is None:
            return
        messages = state.get("messages", [])
        last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
        turn_messages = messages[last_human + 1 :]
        stop = next(
            (m.name for m in turn_messages if isinstance(m, SystemMessage) and m.name in STOP_NAMES),
            None,
        )
        character = state.get("character")
        self._turn.update(
            model_calls=model_calls_this_turn(messages) if messages else 0,
            tool_calls=sum(isinstance(m, ToolMessage) for m in turn_messages),
            stop=stop,
            seconds=round(self._watch.elapsed_seconds(), 3),
            final={"hp": character.hp, "position": list(character.position)} if character else None,
        )
        if error:
            self._turn["error"] = error
        self.turns.append(self._turn)
        self._turn = None
        self._save()

    def _save(self) -> None:
        if self.run is None:
            return
        save_json(
            {"run_id": self.run.run_id, "command": self.command, "turns": self.turns},
            self.run.artifact(TRAJECTORY_FILE),
        )
        self.run.update_meta(turns=len(self.turns))

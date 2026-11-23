"""Детектор зациклення: модель у ході раз за разом викликає той самий інструмент з тими самими аргументами.

Інструменти з exempt не перевіряються: повторний кидок кубиків - звичайна гра, а не зациклення.
Стану детектор не має: рішення рахується з історії повідомлень поточного ходу.
"""

import json
from collections import Counter
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage

# Ім'я системного повідомлення, яким позначається зупинка через зациклення
LOOP = "loop"


@dataclass(frozen=True)
class LoopDetector:
    """Знаходить виклик, який повторився max_repeats разів у поточному ході."""

    max_repeats: int = 3
    exempt: frozenset[str] = frozenset({"roll_dice"})

    def find_loop(self, messages: list[AnyMessage]) -> str | None:
        """Опис повторюваного виклику, наприклад "move_character({...})", або None."""
        last_human = max(i for i, m in enumerate(messages) if m.type == "human")
        counts: Counter[tuple[str, str]] = Counter()
        for message in messages[last_human + 1 :]:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls:
                if call["name"] in self.exempt:
                    continue
                key = (call["name"], json.dumps(call["args"], sort_keys=True, ensure_ascii=False))
                counts[key] += 1
                if counts[key] >= self.max_repeats:
                    return f"{key[0]}({key[1]})"
        return None


def loop_message(call: str, repeats: int) -> str:
    """Текст для LLM: що повторювалося і чому хід зупинено."""
    return (
        f"Зациклення: {call} викликано {repeats} разів у цьому ході з тими самими аргументами. "
        "Хід зупинено. Коротко опиши, що сталося, і чекай наступної дії гравця."
    )

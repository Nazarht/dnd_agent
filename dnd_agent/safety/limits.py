"""Ліміт кроків: скільки викликів моделі дозволено за один хід гравця.

Лічильник не зберігається в стані: він рахується з історії повідомлень поточного ходу,
тож скидається сам, коли гравець пише наступну дію.
"""

from langchain_core.messages import AnyMessage

# Максимум викликів моделі за один хід гравця (інструменти між ними не рахуються)
MAX_MODEL_CALLS_PER_TURN = 8

# Ім'я системного повідомлення, яким позначається зупинка через ліміт
LIMIT_REACHED = "limit"
STOP_MESSAGE = (
    "Ліміт кроків вичерпано: хід зупинено, щоб уникнути зациклення. "
    "Коротко опиши, що встиг зробити персонаж, і чекай наступної дії гравця."
)


def model_calls_this_turn(messages: list[AnyMessage]) -> int:
    """Скільки разів модель відповідала після останнього повідомлення гравця."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    return sum(1 for m in messages[last_human + 1 :] if m.type == "ai")


def limit_reached(messages: list[AnyMessage]) -> bool:
    """Чи вичерпано ліміт викликів моделі в цьому ході."""
    return model_calls_this_turn(messages) >= MAX_MODEL_CALLS_PER_TURN

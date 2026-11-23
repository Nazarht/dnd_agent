"""Вимірювання часу: монотонний секундомір і мітки часу для логів."""

import time
from datetime import datetime, timezone


class Stopwatch:
    """Секундомір на основі time.monotonic(), не залежить від зміни системного часу."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_seconds(self) -> float:
        """Скільки секунд минуло від створення секундоміра."""
        return time.monotonic() - self._start

    def elapsed_ms(self) -> int:
        """Скільки мілісекунд минуло від створення секундоміра (ціле число)."""
        return int(self.elapsed_seconds() * 1000)


def utc_now_iso() -> str:
    """Поточний час у UTC у форматі ISO 8601, наприклад для поля timestamp у логах."""
    return datetime.now(timezone.utc).isoformat()

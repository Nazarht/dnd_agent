"""Читання та запис JSON з коректним відображенням кирилиці."""

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path) -> Path:
    """Записує data у JSON-файл (UTF-8, без \\u-екранування кирилиці, відступ 2 пробіли).

    Створює батьківські папки, якщо їх немає. Об'єкти, яких JSON не знає
    (наприклад Path або datetime), записуються як рядок.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(path: Path) -> Any:
    """Читає JSON-файл у Python-об'єкт."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate(text: str, limit: int = 500) -> str:
    """Обрізає довгий текст для логів; позначає обрізання трьома крапками."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

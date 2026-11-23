"""Глобальні налаштування проєкту: модель, шляхи, завантаження .env."""

from pathlib import Path

from dotenv import load_dotenv

# dnd_agent/dnd_agent/config.py -> dnd_agent/ (корінь репозиторію)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
# Визначення світів і персонажів: game_data/worlds/*.json, game_data/characters/*.json
GAME_DATA_DIR = PROJECT_ROOT / "game_data"

# .env лежить у корені проєкту і не потрапляє до репозиторію (див. .gitignore)
load_dotenv(PROJECT_ROOT / ".env")

# Модель OpenAI з підтримкою tool calling
MODEL_NAME = "gpt-6-luna"

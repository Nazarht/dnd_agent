"""Контекст запуску: унікальний execution id та папка outputs/{execution_id}/.

Кожен запуск (play, eval, demo-safety) отримує власну папку з артефактами:
trajectory.json, test_results.json та run_meta.json.
"""

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dnd_agent.config import OUTPUTS_DIR
from dnd_agent.utils.json_io import load_json, save_json
from dnd_agent.utils.timing import utc_now_iso


class RunContext:
    """
    Один запуск агента: id, власна папка артефактів і run_meta.json.
    """

    def __init__(self, command: str, outputs_dir: Path = OUTPUTS_DIR) -> None:
        # id виду 20261121-185600-a1b2: час UTC + випадковий суфікс,
        # тож папки сортуються за часом і не конфліктують
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{timestamp}-{secrets.token_hex(2)}"
        self.command = command

        # outputs/{run_id}/ (exist_ok=False: наявна папка означала б колізію id)
        self.path = outputs_dir / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)

        # початковий run_meta.json
        self.meta_path = self.artifact("run_meta.json")
        save_json(
            {
                "run_id": self.run_id,
                "command": self.command,
                "status": "running",
                "started_at": utc_now_iso(),
            },
            self.meta_path,
        )

    def artifact(self, filename: str) -> Path:
        """Шлях до файлу всередині папки запуску, наприклад trajectory.json."""
        return self.path / filename

    def update_meta(self, **fields: Any) -> None:
        """Оновлює run_meta.json: нові поля додаються, наявні перезаписуються."""
        meta = load_json(self.meta_path)
        meta.update(fields)
        save_json(meta, self.meta_path)

    def finish(self, status: str = "finished", **fields: Any) -> None:
        """Позначає запуск завершеним і додає час завершення та підсумкові поля."""
        self.update_meta(status=status, finished_at=utc_now_iso(), **fields)

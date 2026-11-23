"""Точка входу CLI: python main.py <команда>."""

import typer
from langchain_openai import ChatOpenAI
from rich.console import Console

from dnd_agent.config import MODEL_NAME

app = typer.Typer(help="D&D ReAct-агент: гра та перевірки.")
console = Console()


@app.callback()
def cli() -> None:
    """Команди проєкту: check, play, demo-safety."""


@app.command()
def check() -> None:
    """Перевірити, що LLM доступний (smoke test)."""
    llm = ChatOpenAI(model=MODEL_NAME)
    response = llm.invoke("Say hello in 3 words")
    console.print(f"[green]OK[/green] {MODEL_NAME}: {response.content}")


@app.command()
def demo_safety() -> None:
    """Показати три захисники: ліміт кроків, таймаут ходу, детектор зациклення (без API)."""
    from dnd_agent.safety.demo import run_safety_demo

    run_safety_demo()


@app.command()
def play(world: str = "forest", character: str = "priest") -> None:
    """Грати в терміналі: карта, стан персонажа і чат."""
    from dnd_agent.ui.play import run_play

    run_play(world, character)


if __name__ == "__main__":
    app()

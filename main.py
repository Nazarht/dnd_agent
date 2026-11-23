"""Точка входу CLI: python main.py <команда>."""

import typer
from langchain_openai import ChatOpenAI
from rich.console import Console

from dnd_agent.config import MODEL_NAME

app = typer.Typer(help="D&D ReAct-агент: гра, демо та оцінювання.")
console = Console()


@app.callback()
def cli() -> None:
    """Команди проєкту: check, tryout, play, demo-safety, eval."""


@app.command()
def check() -> None:
    """Перевірити, що LLM доступний (smoke test)."""
    llm = ChatOpenAI(model=MODEL_NAME)
    response = llm.invoke("Say hello in 3 words")
    console.print(f"[green]OK[/green] {MODEL_NAME}: {response.content}")


@app.command()
def tryout() -> None:
    """Прогнати сценарій ходів через реальну модель без UI."""
    from evaluation.tryout import run_tryout

    run_tryout()


if __name__ == "__main__":
    app()

"""Запуск тест-кейсів: кожен кейс - один хід у новій грі.

Вердикт рахується з траєкторії ходу (той самий запис, що йде в trajectory.json), тож звіт
і файли не розходяться. Результат: outputs/{run_id}/test_results.json і таблиця в консолі.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.table import Table

from dnd_agent.agent.graph import SERDE, build_graph
from dnd_agent.audit.trajectory import TrajectoryLogger
from dnd_agent.config import MODEL_NAME, OUTPUTS_DIR
from dnd_agent.game.loader import load_character, load_world
from dnd_agent.utils.json_io import save_json
from evaluation.test_cases import CASES, EvalCase

console = Console()
TEST_RESULTS_FILE = "test_results.json"


def summarize(turn: dict) -> dict:
    """Фактичний результат ходу з траєкторії: інструменти, події, зупинка, стан і відповідь."""
    steps = turn.get("steps", [])
    tool_calls = [call["name"] for step in steps for call in step.get("tool_calls", [])]
    final = turn.get("final") or {}
    answer = next((step["answer"] for step in reversed(steps) if "answer" in step), None)
    return {
        "tools": sorted(set(tool_calls)),
        "tool_calls": tool_calls,
        "events": [step["event"]["kind"] for step in steps if "event" in step],
        "stop": turn.get("stop"),
        "position": final.get("position"),
        "hp": final.get("hp"),
        "answer": answer,
    }


def check(case: EvalCase, actual: dict) -> dict[str, bool]:
    """Перевірки кейсу окремо: видно, яка саме умова не збіглася."""
    checks = {
        "tools": case.expect_tools <= set(actual["tools"]),
        "no_stop": actual["stop"] is None,
    }
    if case.expect_position is not None:
        checks["position"] = actual["position"] == list(case.expect_position)
    if case.expect_hp is not None:
        checks["hp"] = actual["hp"] == case.expect_hp
    if case.expect_event is not None:
        checks["event"] = case.expect_event in actual["events"]
    return checks


def _step_label(step: dict) -> str:
    """Короткий опис кроку для звіту: вузол і що він зробив."""
    if "tool_calls" in step:
        return f"{step['node']}: " + ", ".join(call["name"] for call in step["tool_calls"])
    if "event" in step:
        return f"{step['node']}: {step['event']['kind']}"
    if "answer" in step:
        return f"{step['node']}: відповідь"
    return step["node"]


def _run_case(app, logger: TrajectoryLogger, case: EvalCase) -> dict:
    """Один кейс: нова гра, один хід, перевірки за траєкторією."""
    world = load_world("forest")
    character = load_character("priest", world.start)
    if case.start_position is not None:
        character = character.model_copy(update={"position": case.start_position})
    config = {"configurable": {"thread_id": f"eval-{case.id}"}}
    payload = {
        "messages": [HumanMessage(content=case.query)],
        "character": character,
        "world": world,
        "step_count": 0,
    }

    logger.start_turn(case.query, label=case.id)
    try:
        for update in app.stream(payload, config, stream_mode="updates"):
            for node, values in update.items():
                logger.record(node, values)
        logger.end_turn(app.get_state(config).values)
    except Exception as error:
        # Помилка - теж результат кейсу, а не привід зупинити весь прогін
        logger.end_turn(app.get_state(config).values, error=repr(error))

    turn = logger.turns[-1]
    actual = summarize(turn)
    checks = check(case, actual)
    return {
        "id": case.id,
        "query": case.query,
        "description": case.description,
        "expected": {
            "tools": sorted(case.expect_tools),
            "position": list(case.expect_position) if case.expect_position else None,
            "hp": case.expect_hp,
            "event": case.expect_event,
        },
        "actual": actual,
        "checks": checks,
        "passed": "error" not in turn and all(checks.values()),
        "steps": [_step_label(step) for step in turn["steps"]],
        "time_seconds": turn["seconds"],
        "error": turn.get("error"),
    }


def _print_report(report: dict) -> None:
    table = Table(title=f"Оцінювання: {report['passed']}/{report['total']} пройдено")
    table.add_column("Кейс")
    table.add_column("Результат")
    table.add_column("Інструменти")
    table.add_column("Час, с", justify="right")
    table.add_column("Не збіглося")
    for result in report["cases"]:
        failed = [name for name, ok in result["checks"].items() if not ok]
        table.add_row(
            result["id"],
            "[green]OK[/green]" if result["passed"] else "[red]ПОМИЛКА[/red]",
            ", ".join(result["actual"]["tools"]) or "-",
            f"{result['time_seconds']:.1f}",
            ", ".join(failed) or "-",
        )
    console.print(table)


def run_eval(
    cases: list[EvalCase] = CASES,
    model: BaseChatModel | None = None,
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict:
    """Прогоняє кейси, записує test_results.json і траєкторію в папку запуску.

    model - None означає реальну модель; у тестах сюди передають фейкову.
    """
    logger = TrajectoryLogger(command="eval", outputs_dir=outputs_dir)
    logger.start_run()
    app = build_graph(model=model, checkpointer=MemorySaver(serde=SERDE))
    results = [_run_case(app, logger, case) for case in cases]
    passed = sum(result["passed"] for result in results)
    report = {
        "run_id": logger.run.run_id,
        "model": MODEL_NAME if model is None else type(model).__name__,
        "total": len(results),
        "passed": passed,
        "cases": results,
    }
    save_json(report, logger.run.artifact(TEST_RESULTS_FILE))
    logger.finish_run(passed=passed, total=len(results))
    _print_report(report)
    return report

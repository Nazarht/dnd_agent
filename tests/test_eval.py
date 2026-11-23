"""Тести оцінювання: набір кейсів, підсумок ходу, перевірки і повний прогін з фейковою моделлю."""

from langchain_core.messages import AIMessage

from dnd_agent.safety.demo import ScriptedModel
from dnd_agent.utils.json_io import load_json
from evaluation.runner import check, run_eval, summarize
from evaluation.test_cases import CASES, EvalCase


def call(name: str, args: dict, index: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{index}", "type": "tool_call"}])


def fake_turn(**overrides) -> dict:
    turn = {
        "steps": [
            {"node": "agent", "tool_calls": [{"name": "move_character", "args": {}}]},
            {"node": "tools", "tool_results": [{"tool": "move_character", "status": "success", "content": ""}]},
            {"node": "world_events", "event": {"kind": "world_event", "text": "..."}},
            {"node": "formatter", "answer": "Готово", "confidence": 1.0, "sources": ["move_character"]},
        ],
        "stop": None,
        "final": {"hp": 16, "position": [2, 3]},
    }
    turn.update(overrides)
    return turn


def test_cases_are_three_to_five_with_unique_ids():
    assert 3 <= len(CASES) <= 5
    assert len({case.id for case in CASES}) == len(CASES)


def test_summarize_extracts_tools_event_and_state():
    actual = summarize(fake_turn())
    assert actual["tools"] == ["move_character"]
    assert actual["events"] == ["world_event"]
    assert actual["position"] == [2, 3]
    assert actual["hp"] == 16
    assert actual["answer"] == "Готово"
    assert actual["stop"] is None


def test_check_reports_each_mismatch_separately():
    case = EvalCase(id="x", query="q", description="d", expect_tools=frozenset({"modify_hp"}), expect_hp=11)
    checks = check(case, summarize(fake_turn()))
    assert checks["tools"] is False  # немає modify_hp
    assert checks["hp"] is False  # HP 16, очікувалось 11
    assert checks["no_stop"] is True


def test_check_passes_when_expectations_match():
    case = EvalCase(id="x", query="q", description="d", expect_tools=frozenset({"move_character"}),
                    expect_position=(2, 3), expect_event="world_event")
    checks = check(case, summarize(fake_turn()))
    assert all(checks.values())


def test_run_eval_writes_results_file(tmp_path):
    case = next(c for c in CASES if c.id == "monster_attacks")
    model = ScriptedModel(responses=[
        AIMessage(content="Бачу гобліна поруч."),
        call("roll_dice", {"dice": "1d20", "attribute": "strength", "modifier": 0, "purpose": "атака монстра"}, 0),
        AIMessage(content="Гоблін промахнувся."),
    ])
    report = run_eval(cases=[case], model=model, outputs_dir=tmp_path)

    assert report["total"] == 1 and report["passed"] == 1
    saved = load_json(tmp_path / report["run_id"] / "test_results.json")
    result = saved["cases"][0]
    assert result["id"] == "monster_attacks"
    assert result["passed"] is True
    assert result["actual"]["tools"] == ["roll_dice"]
    assert result["actual"]["events"] == ["world_event"]
    assert "world_events: world_event" in result["steps"]
    assert result["time_seconds"] >= 0
    # Траєкторія тієї самої папки містить той самий хід, позначений id кейсу
    trajectory = load_json(tmp_path / report["run_id"] / "trajectory.json")
    assert trajectory["turns"][0]["label"] == "monster_attacks"
    run_meta = load_json(tmp_path / report["run_id"] / "run_meta.json")
    assert run_meta["status"] == "finished" and run_meta["passed"] == 1

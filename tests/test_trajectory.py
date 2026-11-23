"""Тести траєкторії: файл trajectory.json після ходу в грі з фейковою моделлю."""

from langchain_core.messages import AIMessage

from dnd_agent.agent.graph import build_graph
from dnd_agent.audit.trajectory import TRAJECTORY_FILE, TrajectoryLogger
from dnd_agent.safety.demo import ScriptedModel
from dnd_agent.safety.loop_detector import LOOP
from dnd_agent.ui.play import Session
from dnd_agent.utils.json_io import load_json


def call(name: str, args: dict, index: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{index}", "type": "tool_call"}])


def play_one_turn(tmp_path, responses, text):
    """Ведe один хід через Session і повертає логер (запуск ще відкритий)."""
    logger = TrajectoryLogger(command="play", outputs_dir=tmp_path)
    session = Session("forest", "priest", build_graph(model=ScriptedModel(responses=responses)), logger)
    session.play_turn(text)
    return logger


def read_trajectory(logger):
    path = logger.run.artifact(TRAJECTORY_FILE)
    return load_json(path)


def test_move_turn_is_recorded(tmp_path):
    logger = play_one_turn(
        tmp_path,
        [call("move_character", {"direction": "east", "steps": 2}, 0), AIMessage(content="Пройшов.")],
        "Іду на схід два кроки",
    )
    trajectory = read_trajectory(logger)
    logger.finish_run()

    turn = trajectory["turns"][0]
    assert turn["player"] == "Іду на схід два кроки"
    assert turn["model_calls"] == 2
    assert turn["tool_calls"] == 1
    assert turn["stop"] is None
    assert turn["final"] == {"hp": 16, "position": [2, 3]}

    nodes = [step["node"] for step in turn["steps"]]
    assert nodes[0] == "agent" and "tools" in nodes and nodes[-1] == "formatter"
    tool_step = next(step for step in turn["steps"] if step["node"] == "tools")
    assert tool_step["position"] == [2, 3]
    assert tool_step["tool_results"][0]["status"] == "success"
    # Відповідь формувача з фейкової моделі (див. ScriptedModel)
    answer_step = turn["steps"][-1]
    assert answer_step["answer"] == "GM зупинив хід"
    assert answer_step["confidence"] == 0.5


def test_loop_stop_is_recorded(tmp_path):
    responses = [call("move_character", {"direction": "west", "steps": 1}, i) for i in range(10)]
    logger = play_one_turn(tmp_path, responses, "Іду на захід")
    trajectory = read_trajectory(logger)
    logger.finish_run()

    turn = trajectory["turns"][0]
    assert turn["stop"] == LOOP
    assert turn["tool_calls"] == 3
    events = [step["event"]["kind"] for step in turn["steps"] if "event" in step]
    assert LOOP in events


def test_text_from_content_blocks_is_plain_string(tmp_path):
    logger = TrajectoryLogger(command="play", outputs_dir=tmp_path)
    logger.start_turn("Огляньмося")
    message = AIMessage(content=[{"type": "text", "text": "Привіт", "annotations": []}])
    logger.record("agent", {"messages": [message]})
    assert logger._turn["steps"][0]["text"] == "Привіт"


def test_run_meta_is_finished_after_run(tmp_path):
    logger = play_one_turn(tmp_path, [AIMessage(content="Тихо.")], "Огляньмося")
    run_dir = logger.run.path
    logger.finish_run()

    meta = load_json(run_dir / "run_meta.json")
    assert meta["status"] == "finished"
    assert meta["turns"] == 1
    assert (run_dir / TRAJECTORY_FILE).exists()

"""Тести таймауту ходу: перевірка дедлайну і зупинка повільного ходу."""

import time

from dnd_agent.safety.demo import DEMO_MODEL_DELAY, DEMO_TURN_TIMEOUT, run_timeout_turn
from dnd_agent.safety.limits import MAX_MODEL_CALLS_PER_TURN, model_calls_this_turn
from dnd_agent.safety.timeout import TIMEOUT, TIMEOUT_MESSAGE, deadline_passed


def test_no_deadline_never_expires():
    assert not deadline_passed(None)


def test_past_deadline_is_expired():
    assert deadline_passed(time.monotonic() - 1)


def test_future_deadline_is_not_expired():
    assert not deadline_passed(time.monotonic() + 60)


def test_slow_turn_stops_on_deadline():
    out, elapsed = run_timeout_turn()
    calls = model_calls_this_turn(out["messages"])
    # Зупинка за часом, а не за лімітом кроків
    assert 1 <= calls < MAX_MODEL_CALLS_PER_TURN
    stop = out["messages"][-1]
    assert stop.name == TIMEOUT
    assert stop.content == TIMEOUT_MESSAGE
    # Хід не має тривати значно довше за таймаут плюс один запит
    assert elapsed < DEMO_TURN_TIMEOUT + 2 * DEMO_MODEL_DELAY + 1.0
    assert out["response"].answer

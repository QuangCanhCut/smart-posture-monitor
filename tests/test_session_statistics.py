import src.session_statistics as session_module
from src.session_statistics import SessionStatistics


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def test_bad_shorter_than_threshold_does_not_alert(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(session_module.time, "monotonic", clock.monotonic)
    stats = SessionStatistics(alert_threshold_seconds=2.0)

    assert stats.update("lean_right") is False
    clock.now = 1.99
    assert stats.update("lean_right") is False
    assert stats.alert_count == 0


def test_continuous_bad_alerts_only_once(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(session_module.time, "monotonic", clock.monotonic)
    stats = SessionStatistics(alert_threshold_seconds=2.0)

    assert stats.update("forward_slouch") is False
    clock.now = 2.0
    assert stats.update("forward_slouch") is True
    clock.now = 6.0
    assert stats.update("forward_slouch") is False
    assert stats.alert_count == 1
    assert stats.bad_seconds == 6.0


def test_correct_then_bad_creates_new_episode(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(session_module.time, "monotonic", clock.monotonic)
    stats = SessionStatistics(alert_threshold_seconds=2.0)

    stats.update("lean_left")
    clock.now = 2.0
    assert stats.update("lean_left") is True
    clock.now = 3.0
    assert stats.update("correct") is False
    clock.now = 4.0
    assert stats.update("lean_left") is False
    clock.now = 6.0
    assert stats.update("lean_left") is True
    assert stats.alert_count == 2


def test_neutral_state_ends_bad_episode(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(session_module.time, "monotonic", clock.monotonic)
    stats = SessionStatistics(alert_threshold_seconds=2.0)

    stats.update("lean_right")
    clock.now = 1.0
    assert stats.update("no_person", status="NO_PERSON") is False
    clock.now = 2.0
    stats.update("lean_right")
    clock.now = 4.0
    assert stats.update("lean_right") is True
    assert stats.alert_count == 1


def test_reset_clears_timers_and_durations(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(session_module.time, "monotonic", clock.monotonic)
    stats = SessionStatistics(alert_threshold_seconds=2.0)
    stats.update("correct")
    clock.now = 1.0
    stats.update("correct")

    stats.reset()

    assert stats.to_dict()["correct_seconds"] == 0.0
    assert stats.start_time is None
    assert stats.alert_count == 0

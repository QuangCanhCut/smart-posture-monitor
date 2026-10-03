from src.temporal_monitor import TemporalMonitor


def _monitor() -> TemporalMonitor:
    return TemporalMonitor(
        window_size=7,
        min_samples=3,
        hysteresis_frames=2,
        yaw_mode="conservative",
    )


def test_single_noisy_prediction_does_not_flicker() -> None:
    monitor = _monitor()
    outputs = [
        monitor.update(label)
        for label in (
            "correct",
            "correct",
            "correct",
            "lean_right",
            "correct",
            "correct",
            "correct",
        )
    ]

    assert outputs[:2] == ["evaluating", "evaluating"]
    assert outputs[2:] == ["correct"] * 5


def test_majority_plus_hysteresis_changes_stable_posture() -> None:
    monitor = _monitor()
    outputs = [
        monitor.update(label)
        for label in (
            "correct",
            "correct",
            "lean_right",
            "lean_right",
            "lean_right",
            "lean_right",
            "lean_right",
        )
    ]

    assert outputs[4] == "correct"
    assert outputs[5:] == ["lean_right", "lean_right"]


def test_invalid_states_do_not_enter_vote_window() -> None:
    monitor = _monitor()
    monitor.update("correct")
    monitor.update("no_person")
    monitor.update("evaluating")
    monitor.update("correct")
    result = monitor.update("correct")

    stats = monitor.get_stats()
    assert result == "correct"
    assert stats["window_distribution"] == {"correct": 3}
    assert stats["valid_samples"] == 3


def test_conservative_head_turn_keeps_state_and_window() -> None:
    monitor = _monitor()
    for _ in range(3):
        monitor.update("correct", face_rotation_proxy=1.0)

    before = monitor.get_stats()["window_distribution"]
    result = monitor.update("lean_right", face_rotation_proxy=2.0)

    assert result == "correct"
    assert monitor.get_stats()["window_distribution"] == before
    assert monitor.get_stats()["yaw_gated"] is True


def test_reset_clears_window_and_pending_transition() -> None:
    monitor = _monitor()
    for label in ("correct", "correct", "correct", "lean_right", "lean_right"):
        monitor.update(label)

    monitor.reset()
    stats = monitor.get_stats()

    assert stats["current_label"] == "evaluating"
    assert stats["valid_samples"] == 0
    assert stats["pending_label"] is None
    assert stats["hysteresis_counter"] == 0


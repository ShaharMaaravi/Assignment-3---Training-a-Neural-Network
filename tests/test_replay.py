from pathlib import Path

import pytest

from evo_racer.controller import NeuralController
from evo_racer.fitness import EpisodeConfig
from evo_racer.replay import ReplaySession
from evo_racer.track import Track


@pytest.fixture
def square_track() -> Track:
    return Track(((100, 100), (300, 100), (300, 300), (100, 300)), width=100)


def constant_controller(steering: float, throttle: float) -> NeuralController:
    controller = NeuralController(seed=23)
    genome = list(controller.get_genome())
    for genes in genome:
        genes.fill(0)
    genome[-1][:] = (steering, throttle)
    controller.set_genome(genome)
    return controller


def test_replay_loads_saved_weights(tmp_path: Path, square_track: Track) -> None:
    original = constant_controller(0.25, 0.75)
    weights_path = original.save_weights(tmp_path / "champion.weights.h5")

    replay = ReplaySession.from_weights(weights_path, track=square_track)

    assert replay.weights_path == weights_path
    assert replay.controller.act((0.0,) * 10) == original.act((0.0,) * 10)


def test_replay_advances_and_resets(square_track: Track) -> None:
    replay = ReplaySession(
        constant_controller(0.0, 1.0),
        track=square_track,
        episode_config=EpisodeConfig(maximum_seconds=1.0, idle_timeout_seconds=0.3),
    )

    replay.step()
    replay.step()

    assert replay.car.distance_travelled > 0
    assert replay.maximum_progress > 0
    assert replay.fitness > 0
    assert replay.last_action.throttle > 0

    replay.reset()
    assert replay.car.distance_travelled == 0
    assert replay.elapsed_seconds == 0
    assert replay.maximum_progress == 0
    assert replay.termination_reason is None
    assert replay.last_action.throttle == 0


def test_idle_champion_stops_as_stalled(square_track: Track) -> None:
    replay = ReplaySession(
        constant_controller(0.0, 0.0),
        track=square_track,
        episode_config=EpisodeConfig(
            time_step=0.1,
            maximum_seconds=1.0,
            idle_timeout_seconds=0.3,
        ),
    )

    while replay.active:
        replay.step()

    assert replay.termination_reason == "stalled"
    assert replay.elapsed_seconds == pytest.approx(0.3)
    assert replay.fitness == 0


def test_missing_weights_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        ReplaySession.from_weights(tmp_path / "missing.weights.h5")

import pytest

from evo_racer.car import Car
from evo_racer.controller import NeuralController
from evo_racer.fitness import (
    EpisodeConfig,
    EpisodeEvaluator,
    fractional_checkpoint_progress,
    race_progress,
)
from evo_racer.track import Track


@pytest.fixture
def square_track() -> Track:
    return Track(((100, 100), (300, 100), (300, 300), (100, 300)), width=100)


def constant_controller(steering: float, throttle: float) -> NeuralController:
    controller = NeuralController(seed=7)
    genome = list(controller.get_genome())
    for genes in genome:
        genes.fill(0)
    genome[-1][:] = (steering, throttle)
    controller.set_genome(genome)
    return controller


def test_fractional_progress_follows_the_current_checkpoint_segment(
    square_track: Track,
) -> None:
    car = Car.at_track_start(square_track)
    assert fractional_checkpoint_progress(car, square_track) == pytest.approx(0.0)
    assert race_progress(car, square_track) == pytest.approx(0.0)

    car.position.x = 200
    assert fractional_checkpoint_progress(car, square_track) == pytest.approx(0.5)
    assert race_progress(car, square_track) == pytest.approx(0.5)


def test_idle_controller_stops_early(square_track: Track) -> None:
    evaluator = EpisodeEvaluator(
        square_track,
        EpisodeConfig(
            time_step=0.1,
            maximum_seconds=2.0,
            idle_timeout_seconds=0.3,
        ),
    )

    result = evaluator.evaluate(constant_controller(0.0, 0.0))

    assert result.termination_reason == "stalled"
    assert result.elapsed_seconds == pytest.approx(0.3)
    assert result.maximum_progress == pytest.approx(0.0)
    assert result.fitness == 0.0


def test_forward_progress_scores_higher_than_stalling(square_track: Track) -> None:
    evaluator = EpisodeEvaluator(
        square_track,
        EpisodeConfig(
            time_step=0.05,
            maximum_seconds=1.0,
            idle_timeout_seconds=0.3,
        ),
    )

    idle_result = evaluator.evaluate(constant_controller(0.0, 0.0))
    moving_result = evaluator.evaluate(constant_controller(0.0, 1.0))

    assert moving_result.maximum_progress > idle_result.maximum_progress
    assert moving_result.fitness > idle_result.fitness
    assert moving_result.distance_travelled > 0


@pytest.mark.parametrize(
    ("keyword", "values"),
    [
        ("timing", {"time_step": 0.0}),
        ("Idle", {"idle_timeout_seconds": 0.0}),
        ("Target", {"target_laps": 0}),
        ("rewards", {"crash_penalty": -1.0}),
    ],
)
def test_episode_config_rejects_invalid_values(keyword: str, values: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=keyword):
        EpisodeConfig(**values)

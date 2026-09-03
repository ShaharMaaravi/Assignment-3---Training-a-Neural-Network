from pathlib import Path

import numpy as np
import pytest

from evo_racer.car import Car
from evo_racer.controller import NeuralController
from evo_racer.track import Track


@pytest.fixture
def observation() -> tuple[float, ...]:
    return (0.4, 0.7, 0.9, 1.0, 0.8, 0.6, 0.3, 0.25, -0.2, 0.98)


def test_controller_has_expected_architecture_and_bounded_outputs(
    observation: tuple[float, ...],
) -> None:
    controller = NeuralController(seed=11)

    action = controller.act(observation)

    assert controller.model.input_shape == (None, 10)
    assert controller.model.output_shape == (None, 2)
    assert controller.parameter_count == 406
    assert -1 <= action.steering <= 1
    assert -1 <= action.throttle <= 1


def test_seeded_controllers_are_reproducible(observation: tuple[float, ...]) -> None:
    first = NeuralController(seed=27)
    second = NeuralController(seed=27)

    assert first.act(observation) == second.act(observation)


def test_genome_copy_and_clone_are_independent(observation: tuple[float, ...]) -> None:
    original = NeuralController(seed=4)
    exported = original.get_genome()
    clone = original.clone()

    exported[0][0, 0] += 100

    assert clone.act(observation) == original.act(observation)
    assert original.get_genome()[0][0, 0] != exported[0][0, 0]


def test_set_genome_rejects_wrong_shapes_and_non_finite_values() -> None:
    controller = NeuralController(seed=3)
    wrong_shape = list(controller.get_genome())
    wrong_shape[0] = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        controller.set_genome(wrong_shape)

    non_finite = list(controller.get_genome())
    non_finite[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        controller.set_genome(non_finite)


def test_invalid_observations_are_rejected() -> None:
    controller = NeuralController(seed=8)

    with pytest.raises(ValueError, match="shape"):
        controller.act((0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        controller.act((0.0,) * 9 + (float("inf"),))


def test_saved_weights_restore_identical_prediction(
    tmp_path: Path,
    observation: tuple[float, ...],
) -> None:
    original = NeuralController(seed=14)
    weights_path = original.save_weights(tmp_path / "driver.weights.h5")
    restored = NeuralController(seed=99)
    restored.load_weights(weights_path)

    assert restored.act(observation).steering == pytest.approx(original.act(observation).steering)
    assert restored.act(observation).throttle == pytest.approx(original.act(observation).throttle)


def test_controller_can_drive_the_simulated_car() -> None:
    track = Track(((100, 100), (300, 100), (300, 300), (100, 300)), width=80)
    car = Car.at_track_start(track)
    controller = NeuralController(seed=2)
    genome = list(controller.get_genome())
    genome[-2].fill(0)
    genome[-1][:] = (0.0, 1.0)
    controller.set_genome(genome)

    start = car.position.copy()
    action = controller.drive(car, track, delta_seconds=0.2)

    assert action.throttle > 0
    assert car.position.distance_to(start) > 0

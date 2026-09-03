import pytest

from evo_racer.car import Car
from evo_racer.track import Track


@pytest.fixture
def square_track() -> Track:
    return Track(((100, 100), (300, 100), (300, 300), (100, 300)), width=80)


def test_car_accelerates_and_moves_forward(square_track: Track) -> None:
    car = Car.at_track_start(square_track)
    start = car.position.copy()

    car.update(steering=0, throttle=1, delta_seconds=0.2, track=square_track)

    assert car.speed > 0
    assert car.position.distance_to(start) > 0
    assert not car.crashed


def test_car_detects_collision_using_its_corners(square_track: Track) -> None:
    car = Car.at_track_start(square_track)
    car.position.update(200, 100)
    car.heading_degrees = -90
    car.speed = 120

    car.update(steering=0, throttle=0, delta_seconds=0.4, track=square_track)

    assert car.crashed
    assert car.speed == 0


def test_car_advances_ordered_checkpoint(square_track: Track) -> None:
    car = Car.at_track_start(square_track)
    car.position.update(270, 100)
    car.heading_degrees = 0
    car.speed = 120

    car.update(steering=0, throttle=0, delta_seconds=0.4, track=square_track)

    assert car.checkpoints_passed == 1
    assert car.next_checkpoint_index == 2


def test_observation_has_normalized_sensor_and_navigation_values(square_track: Track) -> None:
    car = Car.at_track_start(square_track)

    observation = car.observation(square_track)

    assert len(observation) == 10
    assert all(0 <= value <= 1 for value in observation[:7])
    assert -1 <= observation[7] <= 1
    assert -1 <= observation[8] <= 1
    assert -1 <= observation[9] <= 1

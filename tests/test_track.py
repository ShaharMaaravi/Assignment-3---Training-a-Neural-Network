import pytest
from pygame import Vector2

from evo_racer.track import Track


@pytest.fixture
def square_track() -> Track:
    return Track(((100, 100), (300, 100), (300, 300), (100, 300)), width=80)


def test_track_is_a_ring_around_its_centerline(square_track: Track) -> None:
    assert square_track.contains((200, 100))
    assert square_track.contains((300, 240))
    assert not square_track.contains((200, 200))
    assert not square_track.contains((20, 20))


def test_track_location_reports_normalized_progress(square_track: Track) -> None:
    location = square_track.locate((200, 100))

    assert location.segment_index == 0
    assert location.distance_from_center == pytest.approx(0)
    assert location.normalized_progress == pytest.approx(0.125)


def test_raycast_finds_track_edge(square_track: Track) -> None:
    distance = square_track.raycast(Vector2(200, 100), Vector2(0, -1), max_distance=100)

    assert distance == pytest.approx(40, abs=0.1)


def test_checkpoint_must_be_crossed_in_forward_direction(square_track: Track) -> None:
    assert square_track.crossed_checkpoint(Vector2(260, 100), Vector2(310, 100), 1)
    assert not square_track.crossed_checkpoint(Vector2(310, 100), Vector2(260, 100), 1)

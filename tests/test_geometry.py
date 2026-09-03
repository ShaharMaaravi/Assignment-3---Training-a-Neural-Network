import pytest
from pygame import Vector2

from evo_racer.geometry import project_point_to_segment, segments_intersect


def test_point_projection_is_clamped_to_segment() -> None:
    projection = project_point_to_segment(Vector2(15, 4), Vector2(0, 0), Vector2(10, 0))

    assert projection.point == Vector2(10, 0)
    assert projection.fraction == 1.0
    assert projection.distance == pytest.approx(41**0.5)


def test_segment_intersection_detects_crossing_and_rejects_separation() -> None:
    assert segments_intersect(Vector2(0, 5), Vector2(10, 5), Vector2(5, 0), Vector2(5, 10))
    assert not segments_intersect(Vector2(0, 0), Vector2(2, 0), Vector2(5, 0), Vector2(8, 0))

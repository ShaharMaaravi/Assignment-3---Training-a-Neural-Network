from __future__ import annotations

from dataclasses import dataclass

from pygame import Vector2

EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class SegmentProjection:
    point: Vector2
    distance: float
    fraction: float


def project_point_to_segment(point: Vector2, start: Vector2, end: Vector2) -> SegmentProjection:
    """Return the closest point and normalized position on a finite line segment."""

    segment = end - start
    length_squared = segment.length_squared()
    if length_squared <= EPSILON:
        return SegmentProjection(start.copy(), point.distance_to(start), 0.0)

    fraction = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
    projected = start + segment * fraction
    return SegmentProjection(projected, point.distance_to(projected), fraction)


def segments_intersect(
    first_start: Vector2,
    first_end: Vector2,
    second_start: Vector2,
    second_end: Vector2,
) -> bool:
    """Return whether two finite 2D segments intersect, including their endpoints."""

    first = first_end - first_start
    second = second_end - second_start
    denominator = first.cross(second)
    offset = second_start - first_start

    if abs(denominator) <= EPSILON:
        if abs(offset.cross(first)) > EPSILON:
            return False
        first_length_squared = first.length_squared()
        if first_length_squared <= EPSILON:
            return first_start.distance_squared_to(second_start) <= EPSILON
        start_fraction = offset.dot(first) / first_length_squared
        end_fraction = (second_end - first_start).dot(first) / first_length_squared
        low, high = sorted((start_fraction, end_fraction))
        return high >= 0.0 and low <= 1.0

    first_fraction = offset.cross(second) / denominator
    second_fraction = offset.cross(first) / denominator
    return -EPSILON <= first_fraction <= 1.0 + EPSILON and -EPSILON <= second_fraction <= 1.0 + EPSILON

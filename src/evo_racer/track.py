from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from pygame import Vector2

from .geometry import project_point_to_segment, segments_intersect


@dataclass(frozen=True, slots=True)
class TrackPosition:
    distance_from_center: float
    distance_along_track: float
    normalized_progress: float
    segment_index: int
    closest_point: Vector2
    tangent: Vector2


@dataclass(slots=True)
class Track:
    centerline: tuple[Vector2, ...]
    width: float = 130.0
    _segment_lengths: tuple[float, ...] = field(init=False, repr=False)
    _cumulative_lengths: tuple[float, ...] = field(init=False, repr=False)
    total_length: float = field(init=False)

    def __init__(self, centerline: Iterable[tuple[float, float] | Vector2], width: float = 130.0):
        points = tuple(Vector2(point) for point in centerline)
        if len(points) < 3:
            raise ValueError("A closed track needs at least three centerline points")
        if width <= 0:
            raise ValueError("Track width must be positive")
        self.centerline = points
        self.width = float(width)

        lengths = tuple(
            points[index].distance_to(points[(index + 1) % len(points)])
            for index in range(len(points))
        )
        if any(length <= 0 for length in lengths):
            raise ValueError("Consecutive track points must be different")
        self._segment_lengths = lengths
        cumulative = [0.0]
        for length in lengths[:-1]:
            cumulative.append(cumulative[-1] + length)
        self._cumulative_lengths = tuple(cumulative)
        self.total_length = sum(lengths)

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def checkpoint_count(self) -> int:
        return len(self.centerline)

    @property
    def spawn_position(self) -> Vector2:
        return self.centerline[0].copy()

    @property
    def spawn_heading_degrees(self) -> float:
        tangent = self.segment_tangent(0)
        return math.degrees(math.atan2(tangent.y, tangent.x))

    def segment_tangent(self, index: int) -> Vector2:
        start = self.centerline[index % len(self.centerline)]
        end = self.centerline[(index + 1) % len(self.centerline)]
        return (end - start).normalize()

    def checkpoint_tangent(self, index: int) -> Vector2:
        previous = self.centerline[(index - 1) % len(self.centerline)]
        following = self.centerline[(index + 1) % len(self.centerline)]
        tangent = following - previous
        return tangent.normalize() if tangent.length_squared() else self.segment_tangent(index)

    def locate(self, point: Vector2 | tuple[float, float]) -> TrackPosition:
        target = Vector2(point)
        best: TrackPosition | None = None
        for index, start in enumerate(self.centerline):
            end = self.centerline[(index + 1) % len(self.centerline)]
            projection = project_point_to_segment(target, start, end)
            along = self._cumulative_lengths[index] + self._segment_lengths[index] * projection.fraction
            candidate = TrackPosition(
                distance_from_center=projection.distance,
                distance_along_track=along,
                normalized_progress=along / self.total_length,
                segment_index=index,
                closest_point=projection.point,
                tangent=self.segment_tangent(index),
            )
            if best is None or candidate.distance_from_center < best.distance_from_center:
                best = candidate
        assert best is not None
        return best

    def contains(self, point: Vector2 | tuple[float, float], margin: float = 0.0) -> bool:
        usable_half_width = max(0.0, self.half_width - margin)
        return self.locate(point).distance_from_center <= usable_half_width

    def checkpoint_segment(self, index: int, inset: float = 4.0) -> tuple[Vector2, Vector2]:
        checkpoint = self.centerline[index % len(self.centerline)]
        tangent = self.checkpoint_tangent(index)
        normal = Vector2(-tangent.y, tangent.x)
        radius = max(1.0, self.half_width - inset)
        return checkpoint - normal * radius, checkpoint + normal * radius

    def crossed_checkpoint(
        self,
        movement_start: Vector2,
        movement_end: Vector2,
        checkpoint_index: int,
    ) -> bool:
        gate_start, gate_end = self.checkpoint_segment(checkpoint_index)
        if not segments_intersect(movement_start, movement_end, gate_start, gate_end):
            return False
        movement = movement_end - movement_start
        return movement.dot(self.checkpoint_tangent(checkpoint_index)) > 0

    def raycast(
        self,
        origin: Vector2,
        direction: Vector2,
        max_distance: float,
        step: float = 4.0,
    ) -> float:
        """Measure distance to the closest track edge along a ray."""

        if max_distance <= 0 or step <= 0:
            raise ValueError("Ray distance and step must be positive")
        if not self.contains(origin):
            return 0.0
        ray = direction.normalize()
        previous_distance = 0.0
        distance = step
        while distance <= max_distance:
            if not self.contains(origin + ray * distance):
                low, high = previous_distance, distance
                for _ in range(7):
                    middle = (low + high) / 2.0
                    if self.contains(origin + ray * middle):
                        low = middle
                    else:
                        high = middle
                return high
            previous_distance = distance
            distance += step
        return max_distance


def default_track() -> Track:
    return Track(
        (
            (170, 390),
            (215, 210),
            (410, 115),
            (680, 125),
            (930, 105),
            (1090, 245),
            (1100, 475),
            (990, 640),
            (735, 675),
            (525, 590),
            (345, 660),
            (175, 555),
        ),
        width=126.0,
    )

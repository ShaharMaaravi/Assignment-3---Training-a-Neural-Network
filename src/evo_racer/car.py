from __future__ import annotations

import math
from dataclasses import dataclass, field

from pygame import Vector2

from .track import Track

SENSOR_ANGLES = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)


@dataclass(frozen=True, slots=True)
class CarConfig:
    length: float = 38.0
    width: float = 21.0
    maximum_forward_speed: float = 255.0
    maximum_reverse_speed: float = 70.0
    acceleration: float = 175.0
    braking_acceleration: float = 235.0
    drag: float = 0.55
    steering_speed_degrees: float = 150.0
    sensor_range: float = 230.0


@dataclass(slots=True)
class Car:
    position: Vector2
    heading_degrees: float
    config: CarConfig = field(default_factory=CarConfig)
    speed: float = 0.0
    crashed: bool = False
    next_checkpoint_index: int = 1
    checkpoints_passed: int = 0
    laps_completed: int = 0
    distance_travelled: float = 0.0
    idle_seconds: float = 0.0

    @classmethod
    def at_track_start(cls, track: Track, config: CarConfig | None = None) -> Car:
        return cls(
            position=track.spawn_position,
            heading_degrees=track.spawn_heading_degrees,
            config=config or CarConfig(),
        )

    @property
    def heading_vector(self) -> Vector2:
        return Vector2(1.0, 0.0).rotate(self.heading_degrees)

    def reset(self, track: Track) -> None:
        self.position = track.spawn_position
        self.heading_degrees = track.spawn_heading_degrees
        self.speed = 0.0
        self.crashed = False
        self.next_checkpoint_index = 1
        self.checkpoints_passed = 0
        self.laps_completed = 0
        self.distance_travelled = 0.0
        self.idle_seconds = 0.0

    def update(self, steering: float, throttle: float, delta_seconds: float, track: Track) -> None:
        if self.crashed or delta_seconds <= 0:
            return

        steering = max(-1.0, min(1.0, steering))
        throttle = max(-1.0, min(1.0, throttle))
        previous_position = self.position.copy()

        acceleration = self.config.acceleration if throttle >= 0 else self.config.braking_acceleration
        self.speed += throttle * acceleration * delta_seconds
        self.speed -= self.speed * self.config.drag * delta_seconds
        self.speed = max(
            -self.config.maximum_reverse_speed,
            min(self.config.maximum_forward_speed, self.speed),
        )

        speed_ratio = min(1.0, abs(self.speed) / 55.0)
        reverse_direction = -1.0 if self.speed < 0 else 1.0
        self.heading_degrees += (
            steering
            * self.config.steering_speed_degrees
            * speed_ratio
            * reverse_direction
            * delta_seconds
        )
        self.heading_degrees = ((self.heading_degrees + 180.0) % 360.0) - 180.0

        movement = self.heading_vector * self.speed * delta_seconds
        self.position += movement
        self.distance_travelled += movement.length()
        self.idle_seconds = self.idle_seconds + delta_seconds if abs(self.speed) < 8.0 else 0.0

        if track.crossed_checkpoint(previous_position, self.position, self.next_checkpoint_index):
            crossed = self.next_checkpoint_index
            self.checkpoints_passed += 1
            self.next_checkpoint_index = (crossed + 1) % track.checkpoint_count
            if self.next_checkpoint_index == 1:
                self.laps_completed += 1

        if not all(track.contains(corner) for corner in self.corners()):
            self.crashed = True
            self.speed = 0.0

    def corners(self) -> tuple[Vector2, Vector2, Vector2, Vector2]:
        half_length = self.config.length / 2.0
        half_width = self.config.width / 2.0
        forward = self.heading_vector
        side = Vector2(-forward.y, forward.x)
        return (
            self.position + forward * half_length + side * half_width,
            self.position + forward * half_length - side * half_width,
            self.position - forward * half_length - side * half_width,
            self.position - forward * half_length + side * half_width,
        )

    def sensor_distances(self, track: Track) -> tuple[float, ...]:
        distances = []
        for relative_angle in SENSOR_ANGLES:
            direction = self.heading_vector.rotate(relative_angle)
            distances.append(track.raycast(self.position, direction, self.config.sensor_range))
        return tuple(distances)

    def observation(self, track: Track) -> tuple[float, ...]:
        """Return ten normalized values suitable for the future Keras controller."""

        sensors = tuple(distance / self.config.sensor_range for distance in self.sensor_distances(track))
        normalized_speed = self.speed / self.config.maximum_forward_speed
        target = track.centerline[self.next_checkpoint_index] - self.position
        target_angle = math.degrees(math.atan2(target.y, target.x))
        difference = math.radians(((target_angle - self.heading_degrees + 180.0) % 360.0) - 180.0)
        return (*sensors, normalized_speed, math.sin(difference), math.cos(difference))

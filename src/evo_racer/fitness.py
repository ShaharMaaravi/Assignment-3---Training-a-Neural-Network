from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .car import Car
from .controller import NeuralController
from .track import Track, default_track

TerminationReason = Literal["completed", "crashed", "stalled", "timeout"]


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    time_step: float = 1.0 / 30.0
    maximum_seconds: float = 30.0
    idle_timeout_seconds: float = 3.0
    target_laps: int = 1
    checkpoint_reward: float = 1_000.0
    completion_bonus: float = 5_000.0
    remaining_second_bonus: float = 25.0
    crash_penalty: float = 50.0
    elapsed_second_penalty: float = 0.25

    def __post_init__(self) -> None:
        if self.time_step <= 0 or self.maximum_seconds <= 0:
            raise ValueError("Episode timing values must be positive")
        if self.idle_timeout_seconds <= 0:
            raise ValueError("Idle timeout must be positive")
        if self.target_laps <= 0:
            raise ValueError("Target laps must be positive")
        rewards = (
            self.checkpoint_reward,
            self.completion_bonus,
            self.remaining_second_bonus,
            self.crash_penalty,
            self.elapsed_second_penalty,
        )
        if not all(math.isfinite(value) and value >= 0 for value in rewards):
            raise ValueError("Fitness rewards and penalties must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    fitness: float
    maximum_progress: float
    checkpoints_passed: int
    laps_completed: int
    elapsed_seconds: float
    distance_travelled: float
    termination_reason: TerminationReason


def fractional_checkpoint_progress(car: Car, track: Track) -> float:
    """Return progress from the last checkpoint to the next one in the range [0, 1]."""

    next_index = car.next_checkpoint_index
    previous_index = (next_index - 1) % track.checkpoint_count
    segment_start = track.centerline[previous_index]
    segment = track.centerline[next_index] - segment_start
    fraction = (car.position - segment_start).dot(segment) / segment.length_squared()
    return max(0.0, min(1.0, fraction))


def race_progress(car: Car, track: Track) -> float:
    """Measure ordered checkpoint progress with a smooth partial-checkpoint signal."""

    return car.checkpoints_passed + fractional_checkpoint_progress(car, track)


class EpisodeEvaluator:
    """Run one neural driver headlessly and calculate its genetic fitness."""

    def __init__(self, track: Track | None = None, config: EpisodeConfig | None = None):
        self.track = track or default_track()
        self.config = config or EpisodeConfig()

    def evaluate(self, controller: NeuralController) -> EpisodeResult:
        car = Car.at_track_start(self.track)
        elapsed_seconds = 0.0
        maximum_progress = race_progress(car, self.track)
        termination_reason: TerminationReason = "timeout"

        while elapsed_seconds < self.config.maximum_seconds:
            delta_seconds = min(
                self.config.time_step,
                self.config.maximum_seconds - elapsed_seconds,
            )
            controller.drive(car, self.track, delta_seconds)
            elapsed_seconds += delta_seconds
            maximum_progress = max(maximum_progress, race_progress(car, self.track))

            if car.laps_completed >= self.config.target_laps:
                termination_reason = "completed"
                break
            if car.crashed:
                termination_reason = "crashed"
                break
            if car.idle_seconds >= self.config.idle_timeout_seconds:
                termination_reason = "stalled"
                break

        fitness = maximum_progress * self.config.checkpoint_reward
        fitness -= elapsed_seconds * self.config.elapsed_second_penalty
        if termination_reason == "completed":
            remaining = max(0.0, self.config.maximum_seconds - elapsed_seconds)
            fitness += self.config.completion_bonus
            fitness += remaining * self.config.remaining_second_bonus
        elif termination_reason == "crashed":
            fitness -= self.config.crash_penalty

        return EpisodeResult(
            fitness=max(0.0, fitness),
            maximum_progress=maximum_progress,
            checkpoints_passed=car.checkpoints_passed,
            laps_completed=car.laps_completed,
            elapsed_seconds=elapsed_seconds,
            distance_travelled=car.distance_travelled,
            termination_reason=termination_reason,
        )

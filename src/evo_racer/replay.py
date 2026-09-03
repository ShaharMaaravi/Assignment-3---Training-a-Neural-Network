from __future__ import annotations

from pathlib import Path

from .car import Car
from .controller import ControlAction, ControllerConfig, NeuralController
from .fitness import EpisodeConfig, TerminationReason, calculate_fitness, race_progress
from .track import Track, default_track


class ReplaySession:
    """Run a saved champion in the deterministic racing simulation."""

    def __init__(
        self,
        controller: NeuralController,
        *,
        track: Track | None = None,
        episode_config: EpisodeConfig | None = None,
        weights_path: str | Path | None = None,
    ):
        self.controller = controller
        self.track = track or default_track()
        self.episode_config = episode_config or EpisodeConfig(maximum_seconds=60.0)
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.car = Car.at_track_start(self.track)
        self.elapsed_seconds = 0.0
        self.maximum_progress = 0.0
        self.termination_reason: TerminationReason | None = None
        self.last_action = ControlAction(steering=0.0, throttle=0.0)

    @classmethod
    def from_weights(
        cls,
        weights_path: str | Path,
        *,
        track: Track | None = None,
        episode_config: EpisodeConfig | None = None,
        controller_config: ControllerConfig | None = None,
    ) -> ReplaySession:
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"Champion weights were not found: {path}")
        controller = NeuralController(controller_config)
        controller.load_weights(path)
        return cls(
            controller,
            track=track,
            episode_config=episode_config,
            weights_path=path,
        )

    @property
    def active(self) -> bool:
        return self.termination_reason is None

    @property
    def fitness(self) -> float:
        reason = self.termination_reason or "timeout"
        return calculate_fitness(
            self.maximum_progress,
            self.elapsed_seconds,
            reason,
            self.episode_config,
        )

    def step(self) -> bool:
        if not self.active:
            return False

        delta_seconds = min(
            self.episode_config.time_step,
            self.episode_config.maximum_seconds - self.elapsed_seconds,
        )
        self.last_action = self.controller.drive(self.car, self.track, delta_seconds)
        self.elapsed_seconds += delta_seconds
        self.maximum_progress = max(
            self.maximum_progress,
            race_progress(self.car, self.track),
        )

        if self.car.laps_completed >= self.episode_config.target_laps:
            self.termination_reason = "completed"
        elif self.car.crashed:
            self.termination_reason = "crashed"
        elif self.car.idle_seconds >= self.episode_config.idle_timeout_seconds:
            self.termination_reason = "stalled"
        elif self.elapsed_seconds >= self.episode_config.maximum_seconds:
            self.termination_reason = "timeout"
        return self.active

    def reset(self) -> None:
        self.car = Car.at_track_start(self.track)
        self.elapsed_seconds = 0.0
        self.maximum_progress = 0.0
        self.termination_reason = None
        self.last_action = ControlAction(steering=0.0, throttle=0.0)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .car import Car
from .controller import NeuralController
from .evolution import EvolutionEngine, GenerationStats, ScoredController, TrainingResult
from .fitness import (
    EpisodeConfig,
    EpisodeResult,
    TerminationReason,
    build_episode_result,
    race_progress,
)
from .reporting import TrainingReportPaths, write_training_report
from .track import Track, default_track


@dataclass(slots=True)
class DriverRun:
    controller: NeuralController
    car: Car
    maximum_progress: float = 0.0
    finished_at_seconds: float | None = None
    termination_reason: TerminationReason | None = None

    @property
    def active(self) -> bool:
        return self.termination_reason is None


class TrainingSession:
    """Advance genetic training one simulation tick at a time for live rendering."""

    def __init__(
        self,
        engine: EvolutionEngine,
        *,
        generations: int = 25,
        track: Track | None = None,
        episode_config: EpisodeConfig | None = None,
        champion_path: str | Path = "artifacts/champion.weights.h5",
        report_directory: str | Path | None = None,
    ):
        if generations <= 0:
            raise ValueError("Training must include at least one generation")
        self.engine = engine
        self.total_generations = generations
        self.track = track or default_track()
        self.episode_config = episode_config or EpisodeConfig()
        self.champion_path = Path(champion_path)
        self.report_directory = (
            Path(report_directory) if report_directory is not None else self.champion_path.parent
        )
        self.report_paths: TrainingReportPaths | None = None
        self.generation = 1
        self.elapsed_seconds = 0.0
        self.history: list[GenerationStats] = []
        self.last_scored_population: tuple[ScoredController, ...] = ()
        self.champion: NeuralController | None = None
        self.champion_episode: EpisodeResult | None = None
        self.complete = False
        self.runs = self._new_runs(self.engine.create_population())

    @property
    def active_count(self) -> int:
        return sum(run.active for run in self.runs)

    @property
    def leader(self) -> DriverRun:
        return max(self.runs, key=lambda run: run.maximum_progress)

    def step(self) -> bool:
        """Advance one fixed simulation tick; return False once training is complete."""

        if self.complete:
            return False

        delta_seconds = min(
            self.episode_config.time_step,
            self.episode_config.maximum_seconds - self.elapsed_seconds,
        )
        for run in self.runs:
            if run.active:
                run.controller.drive(run.car, self.track, delta_seconds)
                run.maximum_progress = max(
                    run.maximum_progress,
                    race_progress(run.car, self.track),
                )

        self.elapsed_seconds += delta_seconds
        for run in self.runs:
            if not run.active:
                continue
            reason = self._termination_reason(run.car)
            if reason is not None:
                run.termination_reason = reason
                run.finished_at_seconds = self.elapsed_seconds

        if self.elapsed_seconds >= self.episode_config.maximum_seconds:
            for run in self.runs:
                if run.active:
                    run.termination_reason = "timeout"
                    run.finished_at_seconds = self.elapsed_seconds

        if self.active_count == 0:
            self._finish_generation()
        return not self.complete

    def result(self) -> TrainingResult:
        if not self.complete or self.champion is None or self.champion_episode is None:
            raise RuntimeError("Training has not finished")
        return TrainingResult(
            champion=self.champion.clone(),
            champion_episode=self.champion_episode,
            history=tuple(self.history),
        )

    def _new_runs(self, population: list[NeuralController]) -> list[DriverRun]:
        return [
            DriverRun(
                controller=controller,
                car=Car.at_track_start(self.track),
                maximum_progress=0.0,
            )
            for controller in population
        ]

    def _termination_reason(self, car: Car) -> TerminationReason | None:
        if car.laps_completed >= self.episode_config.target_laps:
            return "completed"
        if car.crashed:
            return "crashed"
        if car.idle_seconds >= self.episode_config.idle_timeout_seconds:
            return "stalled"
        return None

    def _finish_generation(self) -> None:
        scored = tuple(
            ScoredController(
                controller=run.controller,
                episode=build_episode_result(
                    run.car,
                    run.maximum_progress,
                    run.finished_at_seconds or self.elapsed_seconds,
                    run.termination_reason or "timeout",
                    self.episode_config,
                ),
            )
            for run in self.runs
        )
        self.last_scored_population = scored
        stats = self.engine.generation_stats(self.generation, scored)
        self.history.append(stats)
        generation_best = max(scored, key=lambda candidate: candidate.fitness)
        if (
            self.champion_episode is None
            or generation_best.fitness > self.champion_episode.fitness
        ):
            self.champion = generation_best.controller.clone()
            self.champion_episode = generation_best.episode
            self.champion.save_weights(self.champion_path)

        is_complete = self.generation >= self.total_generations
        if is_complete:
            self.complete = True

        assert self.champion_episode is not None
        self.report_paths = write_training_report(
            self.report_directory,
            self.history,
            self.champion_episode,
            self.engine.config,
            self.episode_config,
            total_generations=self.total_generations,
            complete=is_complete,
            champion_weights=self.champion_path,
        )

        if is_complete:
            return

        population = self.engine.next_population(scored)
        self.generation += 1
        self.elapsed_seconds = 0.0
        self.runs = self._new_runs(population)

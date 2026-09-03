from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pygame

from .evolution import EvolutionConfig, GenerationStats
from .fitness import EpisodeConfig, EpisodeResult

BACKGROUND = (18, 13, 38)
PANEL = (32, 24, 63)
GRID = (72, 61, 98)
WHITE = (247, 245, 252)
MUTED = (171, 165, 191)
LIME = (200, 255, 88)
CYAN = (91, 225, 213)


@dataclass(frozen=True, slots=True)
class TrainingReportPaths:
    history_csv: Path
    summary_json: Path
    fitness_chart: Path


def write_training_report(
    output_directory: str | Path,
    history: list[GenerationStats],
    champion_episode: EpisodeResult,
    evolution_config: EvolutionConfig,
    episode_config: EpisodeConfig,
    *,
    total_generations: int,
    complete: bool,
    champion_weights: str | Path,
) -> TrainingReportPaths:
    if not history:
        raise ValueError("A training report requires at least one completed generation")

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = TrainingReportPaths(
        history_csv=directory / "training_history.csv",
        summary_json=directory / "training_summary.json",
        fitness_chart=directory / "fitness_progress.png",
    )
    _write_history_csv(paths.history_csv, history)
    _write_summary_json(
        paths.summary_json,
        history,
        champion_episode,
        evolution_config,
        episode_config,
        total_generations=total_generations,
        complete=complete,
        champion_weights=champion_weights,
    )
    _write_fitness_chart(paths.fitness_chart, history, complete)
    return paths


def _write_history_csv(path: Path, history: list[GenerationStats]) -> None:
    fieldnames = (
        "generation",
        "best_fitness",
        "mean_fitness",
        "median_fitness",
        "worst_fitness",
        "best_checkpoints",
        "best_laps",
        "best_termination_reason",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for stats in history:
            writer.writerow(asdict(stats))


def _write_summary_json(
    path: Path,
    history: list[GenerationStats],
    champion_episode: EpisodeResult,
    evolution_config: EvolutionConfig,
    episode_config: EpisodeConfig,
    *,
    total_generations: int,
    complete: bool,
    champion_weights: str | Path,
) -> None:
    best_stats = max(history, key=lambda stats: stats.best_fitness)
    payload: dict[str, Any] = {
        "project": "EvoRacer",
        "status": "complete" if complete else "in_progress",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "completed_generations": len(history),
        "planned_generations": total_generations,
        "best_generation": best_stats.generation,
        "champion_weights": str(champion_weights),
        "champion_episode": asdict(champion_episode),
        "latest_generation": asdict(history[-1]),
        "evolution_config": asdict(evolution_config),
        "episode_config": asdict(episode_config),
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _write_fitness_chart(
    path: Path,
    history: list[GenerationStats],
    complete: bool,
) -> None:
    pygame.font.init()
    surface = pygame.Surface((1000, 600))
    surface.fill(BACKGROUND)
    title_font = pygame.font.SysFont("Arial", 34, bold=True)
    body_font = pygame.font.SysFont("Arial", 18)
    label_font = pygame.font.SysFont("Arial", 15)

    _blit_text(surface, title_font, "EVORACER FITNESS PROGRESS", (58, 36), LIME)
    state = "Training complete" if complete else "Training in progress"
    _blit_text(
        surface,
        body_font,
        f"{state} - {len(history)} generations recorded",
        (60, 78),
        MUTED,
    )

    chart = pygame.Rect(88, 130, 840, 350)
    pygame.draw.rect(surface, PANEL, chart, border_radius=12)
    maximum = max(stats.best_fitness for stats in history)
    maximum = max(1.0, maximum * 1.08)
    for index in range(6):
        fraction = index / 5
        y = round(chart.bottom - chart.height * fraction)
        pygame.draw.line(surface, GRID, (chart.left, y), (chart.right, y), 1)
        label = _format_axis_value(maximum * fraction, maximum)
        rendered = label_font.render(label, True, MUTED)
        surface.blit(rendered, rendered.get_rect(midright=(chart.left - 12, y)))

    best_points = _series_points(chart, history, maximum, "best_fitness")
    mean_points = _series_points(chart, history, maximum, "mean_fitness")
    _draw_series(surface, best_points, LIME)
    _draw_series(surface, mean_points, CYAN)

    first_generation = label_font.render("1", True, MUTED)
    last_generation = label_font.render(str(history[-1].generation), True, MUTED)
    surface.blit(first_generation, first_generation.get_rect(midtop=(chart.left, chart.bottom + 10)))
    surface.blit(last_generation, last_generation.get_rect(midtop=(chart.right, chart.bottom + 10)))

    pygame.draw.line(surface, LIME, (620, 535), (650, 535), 4)
    _blit_text(surface, body_font, "Best fitness", (660, 524), WHITE)
    pygame.draw.line(surface, CYAN, (790, 535), (820, 535), 4)
    _blit_text(surface, body_font, "Mean fitness", (830, 524), WHITE)
    best = max(history, key=lambda stats: stats.best_fitness)
    _blit_text(
        surface,
        body_font,
        f"Peak {best.best_fitness:,.1f} at generation {best.generation}",
        (60, 524),
        WHITE,
    )
    pygame.image.save(surface, path)


def _series_points(
    chart: pygame.Rect,
    history: list[GenerationStats],
    maximum: float,
    attribute: str,
) -> list[tuple[int, int]]:
    denominator = max(1, len(history) - 1)
    return [
        (
            round(chart.left + chart.width * index / denominator),
            round(chart.bottom - chart.height * getattr(stats, attribute) / maximum),
        )
        for index, stats in enumerate(history)
    ]


def _draw_series(
    surface: pygame.Surface,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    if len(points) >= 2:
        pygame.draw.aalines(surface, color, False, points)
        pygame.draw.lines(surface, color, False, points, 3)
    for point in points:
        pygame.draw.circle(surface, color, point, 5)


def _blit_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    value: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    surface.blit(font.render(value, True, color), position)


def _format_axis_value(value: float, maximum: float) -> str:
    if maximum < 10:
        return f"{value:.1f}"
    if maximum < 1_000:
        return f"{value:,.0f}"
    return f"{value / 1_000:.1f}k"

import csv
import json
from pathlib import Path

import pygame

from evo_racer.evolution import EvolutionConfig, GenerationStats
from evo_racer.fitness import EpisodeConfig, EpisodeResult
from evo_racer.reporting import write_training_report


def generation(number: int, best: float, mean: float) -> GenerationStats:
    return GenerationStats(
        generation=number,
        best_fitness=best,
        mean_fitness=mean,
        median_fitness=mean - 5,
        worst_fitness=max(0, mean - 30),
        best_checkpoints=number,
        best_laps=0,
        best_termination_reason="timeout",
    )


def test_training_report_writes_csv_json_and_chart(tmp_path: Path) -> None:
    history = [generation(1, 120.0, 60.0), generation(2, 240.0, 110.0)]
    champion = EpisodeResult(
        fitness=240.0,
        maximum_progress=2.4,
        checkpoints_passed=2,
        laps_completed=0,
        elapsed_seconds=10.0,
        distance_travelled=500.0,
        termination_reason="timeout",
    )

    paths = write_training_report(
        tmp_path,
        history,
        champion,
        EvolutionConfig(),
        EpisodeConfig(),
        total_generations=2,
        complete=True,
        champion_weights=tmp_path / "champion.weights.h5",
    )

    with paths.history_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["generation"] for row in rows] == ["1", "2"]
    assert rows[-1]["best_fitness"] == "240.0"

    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["completed_generations"] == 2
    assert summary["best_generation"] == 2
    assert summary["champion_episode"]["fitness"] == 240.0
    assert summary["evolution_config"]["population_size"] == 30

    chart = pygame.image.load(paths.fitness_chart)
    assert chart.get_size() == (1000, 600)


def test_training_report_rejects_empty_history(tmp_path: Path) -> None:
    champion = EpisodeResult(0, 0, 0, 0, 0, 0, "timeout")

    try:
        write_training_report(
            tmp_path,
            [],
            champion,
            EvolutionConfig(),
            EpisodeConfig(),
            total_generations=1,
            complete=False,
            champion_weights="champion.weights.h5",
        )
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("An empty report should fail")

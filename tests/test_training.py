from pathlib import Path

import pytest

from evo_racer.evolution import EvolutionConfig, EvolutionEngine
from evo_racer.fitness import EpisodeConfig
from evo_racer.track import Track
from evo_racer.training import TrainingSession


def test_live_session_runs_generations_and_saves_the_champion(tmp_path: Path) -> None:
    track = Track(((100, 100), (300, 100), (300, 300), (100, 300)), width=100)
    engine = EvolutionEngine(
        EvolutionConfig(
            population_size=3,
            elite_count=1,
            tournament_size=2,
            seed=31,
        )
    )
    champion_path = tmp_path / "champion.weights.h5"
    session = TrainingSession(
        engine,
        generations=2,
        track=track,
        episode_config=EpisodeConfig(
            time_step=0.05,
            maximum_seconds=0.05,
            idle_timeout_seconds=1.0,
        ),
        champion_path=champion_path,
    )

    while not session.complete:
        session.step()

    result = session.result()
    assert champion_path.is_file()
    assert len(result.history) == 2
    assert [stats.generation for stats in result.history] == [1, 2]
    assert result.champion_episode.fitness == pytest.approx(
        max(stats.best_fitness for stats in result.history)
    )
    assert session.active_count == 0


def test_training_result_is_unavailable_before_completion(tmp_path: Path) -> None:
    engine = EvolutionEngine(
        EvolutionConfig(population_size=2, elite_count=1, tournament_size=2)
    )
    session = TrainingSession(engine, generations=1, champion_path=tmp_path / "driver.weights.h5")

    with pytest.raises(RuntimeError, match="not finished"):
        session.result()

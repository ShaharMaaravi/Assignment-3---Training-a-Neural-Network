import numpy as np
import pytest

from evo_racer.controller import ControllerConfig, NeuralController
from evo_racer.evolution import (
    EvolutionConfig,
    EvolutionEngine,
    ScoredController,
    crossover_genomes,
    mutate_genome,
)
from evo_racer.fitness import EpisodeResult


def episode(fitness: float, checkpoints: int = 0) -> EpisodeResult:
    return EpisodeResult(
        fitness=fitness,
        maximum_progress=float(checkpoints),
        checkpoints_passed=checkpoints,
        laps_completed=0,
        elapsed_seconds=1.0,
        distance_travelled=10.0,
        termination_reason="timeout",
    )


def assert_same_genome(first: NeuralController, second: NeuralController) -> None:
    for first_array, second_array in zip(
        first.get_genome(),
        second.get_genome(),
        strict=True,
    ):
        np.testing.assert_array_equal(first_array, second_array)


def test_uniform_crossover_uses_only_parent_values() -> None:
    first = (np.ones((100,), dtype=np.float32),)
    second = (np.full((100,), 2.0, dtype=np.float32),)

    child = crossover_genomes(first, second, np.random.default_rng(12))

    assert set(np.unique(child[0])) == {1.0, 2.0}


def test_mutation_can_change_every_gene_without_changing_the_parent() -> None:
    parent = (np.zeros((50,), dtype=np.float32),)

    child = mutate_genome(
        parent,
        np.random.default_rng(4),
        mutation_rate=1.0,
        standard_deviation=0.2,
    )

    assert np.all(parent[0] == 0)
    assert np.any(child[0] != 0)


def test_next_population_preserves_the_best_elites() -> None:
    config = EvolutionConfig(population_size=6, elite_count=2, tournament_size=2, seed=5)
    engine = EvolutionEngine(config)
    population = engine.create_population()
    scored = [
        ScoredController(controller=controller, episode=episode(float(index)))
        for index, controller in enumerate(population)
    ]

    next_population = engine.next_population(scored)

    assert len(next_population) == config.population_size
    assert_same_genome(next_population[0], population[5])
    assert_same_genome(next_population[1], population[4])
    assert next_population[0] is not population[5]


def test_training_records_non_decreasing_best_fitness() -> None:
    controller_config = ControllerConfig(input_size=2, hidden_layers=(3,))
    evolution_config = EvolutionConfig(
        population_size=8,
        elite_count=2,
        tournament_size=3,
        mutation_rate=0.4,
        mutation_standard_deviation=0.2,
        seed=19,
    )

    def genome_score(controller: NeuralController) -> EpisodeResult:
        score = sum(float(np.sum(genes)) for genes in controller.get_genome())
        return episode(score)

    observed_generations = []
    engine = EvolutionEngine(evolution_config, controller_config, genome_score)
    result = engine.train(4, callback=observed_generations.append)

    best_values = [stats.best_fitness for stats in result.history]
    assert [stats.generation for stats in result.history] == [1, 2, 3, 4]
    assert observed_generations == list(result.history)
    assert best_values == sorted(best_values)
    assert result.champion_episode.fitness == pytest.approx(best_values[-1])


@pytest.mark.parametrize(
    "values",
    [
        {"population_size": 1},
        {"population_size": 4, "elite_count": 4},
        {"population_size": 4, "tournament_size": 5},
        {"crossover_rate": 1.1},
        {"mutation_rate": -0.1},
        {"mutation_standard_deviation": -0.1},
    ],
)
def test_evolution_config_rejects_invalid_values(values: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        EvolutionConfig(**values)

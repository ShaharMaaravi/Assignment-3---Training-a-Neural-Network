from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .controller import ControllerConfig, Genome, NeuralController
from .fitness import EpisodeEvaluator, EpisodeResult

FitnessFunction = Callable[[NeuralController], EpisodeResult]
GenerationCallback = Callable[["GenerationStats"], None]


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    population_size: int = 30
    elite_count: int = 4
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_rate: float = 0.08
    mutation_standard_deviation: float = 0.15
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("Population size must be at least two")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("Elite count must be between one and population size minus one")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("Tournament size must fit inside the population")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between zero and one")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between zero and one")
        if self.mutation_standard_deviation < 0:
            raise ValueError("Mutation standard deviation cannot be negative")


@dataclass(frozen=True, slots=True)
class ScoredController:
    controller: NeuralController
    episode: EpisodeResult

    @property
    def fitness(self) -> float:
        return self.episode.fitness


@dataclass(frozen=True, slots=True)
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    worst_fitness: float
    best_checkpoints: int
    best_laps: int
    best_termination_reason: str


@dataclass(frozen=True, slots=True)
class TrainingResult:
    champion: NeuralController
    champion_episode: EpisodeResult
    history: tuple[GenerationStats, ...]


def crossover_genomes(
    first: Sequence[NDArray[np.floating]],
    second: Sequence[NDArray[np.floating]],
    rng: np.random.Generator,
) -> Genome:
    """Combine two genomes using independent uniform crossover masks."""

    if len(first) != len(second):
        raise ValueError("Parent genomes must contain the same number of arrays")

    child: list[NDArray[np.float32]] = []
    for index, (first_array, second_array) in enumerate(zip(first, second, strict=True)):
        left = np.asarray(first_array, dtype=np.float32)
        right = np.asarray(second_array, dtype=np.float32)
        if left.shape != right.shape:
            raise ValueError(f"Parent genome array {index} shapes do not match")
        mask = rng.random(left.shape) < 0.5
        child.append(np.where(mask, left, right).astype(np.float32, copy=False))
    return tuple(child)


def mutate_genome(
    genome: Sequence[NDArray[np.floating]],
    rng: np.random.Generator,
    mutation_rate: float,
    standard_deviation: float,
) -> Genome:
    """Apply zero-mean Gaussian mutations to independently selected parameters."""

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("Mutation rate must be between zero and one")
    if standard_deviation < 0:
        raise ValueError("Mutation standard deviation cannot be negative")

    mutated: list[NDArray[np.float32]] = []
    for genes in genome:
        values = np.asarray(genes, dtype=np.float32).copy()
        mask = rng.random(values.shape) < mutation_rate
        noise = rng.normal(0.0, standard_deviation, values.shape).astype(np.float32)
        values[mask] += noise[mask]
        mutated.append(values)
    return tuple(mutated)


class EvolutionEngine:
    """Evolve Keras controller weights with elitism and tournament selection."""

    def __init__(
        self,
        evolution_config: EvolutionConfig | None = None,
        controller_config: ControllerConfig | None = None,
        fitness_function: FitnessFunction | None = None,
    ):
        self.config = evolution_config or EvolutionConfig()
        self.controller_config = controller_config or ControllerConfig()
        self._fitness_function = fitness_function or EpisodeEvaluator().evaluate
        self._rng = np.random.default_rng(self.config.seed)

    def create_population(self) -> list[NeuralController]:
        maximum_seed = np.iinfo(np.int32).max
        return [
            NeuralController(
                self.controller_config,
                seed=int(self._rng.integers(0, maximum_seed)),
            )
            for _ in range(self.config.population_size)
        ]

    def evaluate_population(
        self,
        population: Sequence[NeuralController],
    ) -> list[ScoredController]:
        if len(population) != self.config.population_size:
            raise ValueError(
                f"Expected {self.config.population_size} controllers, received {len(population)}"
            )
        return [
            ScoredController(controller=controller, episode=self._fitness_function(controller))
            for controller in population
        ]

    def generation_stats(
        self,
        generation: int,
        scored_population: Sequence[ScoredController],
    ) -> GenerationStats:
        if generation < 1:
            raise ValueError("Generation numbers start at one")
        if not scored_population:
            raise ValueError("Cannot summarize an empty population")

        fitness_values = np.asarray(
            [candidate.fitness for candidate in scored_population],
            dtype=np.float64,
        )
        best = max(scored_population, key=lambda candidate: candidate.fitness)
        return GenerationStats(
            generation=generation,
            best_fitness=float(np.max(fitness_values)),
            mean_fitness=float(np.mean(fitness_values)),
            median_fitness=float(np.median(fitness_values)),
            worst_fitness=float(np.min(fitness_values)),
            best_checkpoints=best.episode.checkpoints_passed,
            best_laps=best.episode.laps_completed,
            best_termination_reason=best.episode.termination_reason,
        )

    def next_population(
        self,
        scored_population: Sequence[ScoredController],
    ) -> list[NeuralController]:
        if len(scored_population) != self.config.population_size:
            raise ValueError(
                f"Expected {self.config.population_size} scored controllers, "
                f"received {len(scored_population)}"
            )

        ranked = sorted(scored_population, key=lambda candidate: candidate.fitness, reverse=True)
        population = [
            candidate.controller.clone() for candidate in ranked[: self.config.elite_count]
        ]
        while len(population) < self.config.population_size:
            first_parent = self._select_parent(scored_population)
            second_parent = self._select_parent(scored_population)
            first_genome = first_parent.controller.get_genome()
            second_genome = second_parent.controller.get_genome()

            if self._rng.random() < self.config.crossover_rate:
                child_genome = crossover_genomes(first_genome, second_genome, self._rng)
            else:
                source = first_genome if self._rng.random() < 0.5 else second_genome
                child_genome = tuple(array.copy() for array in source)
            child_genome = mutate_genome(
                child_genome,
                self._rng,
                self.config.mutation_rate,
                self.config.mutation_standard_deviation,
            )

            child = NeuralController(self.controller_config)
            child.set_genome(child_genome)
            population.append(child)
        return population

    def train(
        self,
        generations: int,
        callback: GenerationCallback | None = None,
    ) -> TrainingResult:
        if generations <= 0:
            raise ValueError("Training must include at least one generation")

        population = self.create_population()
        history: list[GenerationStats] = []
        champion: NeuralController | None = None
        champion_episode: EpisodeResult | None = None

        for generation in range(1, generations + 1):
            scored = self.evaluate_population(population)
            stats = self.generation_stats(generation, scored)
            history.append(stats)
            generation_best = max(scored, key=lambda candidate: candidate.fitness)
            if champion_episode is None or generation_best.fitness > champion_episode.fitness:
                champion = generation_best.controller.clone()
                champion_episode = generation_best.episode
            if callback is not None:
                callback(stats)
            if generation < generations:
                population = self.next_population(scored)

        assert champion is not None and champion_episode is not None
        return TrainingResult(
            champion=champion,
            champion_episode=champion_episode,
            history=tuple(history),
        )

    def _select_parent(
        self,
        scored_population: Sequence[ScoredController],
    ) -> ScoredController:
        indices = self._rng.choice(
            len(scored_population),
            size=self.config.tournament_size,
            replace=False,
        )
        return max((scored_population[index] for index in indices), key=lambda item: item.fitness)

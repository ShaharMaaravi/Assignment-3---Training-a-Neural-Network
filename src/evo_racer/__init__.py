"""EvoRacer neural-network racing project."""

from .car import Car, CarConfig
from .controller import ControlAction, ControllerConfig, NeuralController
from .evolution import EvolutionConfig, EvolutionEngine, GenerationStats, TrainingResult
from .fitness import EpisodeConfig, EpisodeEvaluator, EpisodeResult
from .track import Track, default_track

__all__ = [
    "Car",
    "CarConfig",
    "ControlAction",
    "ControllerConfig",
    "EpisodeConfig",
    "EpisodeEvaluator",
    "EpisodeResult",
    "EvolutionConfig",
    "EvolutionEngine",
    "GenerationStats",
    "NeuralController",
    "Track",
    "TrainingResult",
    "default_track",
]

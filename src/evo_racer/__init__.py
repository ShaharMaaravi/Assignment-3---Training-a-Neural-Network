"""EvoRacer neural-network racing project."""

from .car import Car, CarConfig
from .controller import ControlAction, ControllerConfig, NeuralController
from .evolution import EvolutionConfig, EvolutionEngine, GenerationStats, TrainingResult
from .fitness import EpisodeConfig, EpisodeEvaluator, EpisodeResult
from .replay import ReplaySession
from .reporting import TrainingReportPaths, write_training_report
from .track import Track, default_track
from .training import DriverRun, TrainingSession

__all__ = [
    "Car",
    "CarConfig",
    "ControlAction",
    "ControllerConfig",
    "DriverRun",
    "EpisodeConfig",
    "EpisodeEvaluator",
    "EpisodeResult",
    "EvolutionConfig",
    "EvolutionEngine",
    "GenerationStats",
    "NeuralController",
    "ReplaySession",
    "Track",
    "TrainingReportPaths",
    "TrainingResult",
    "TrainingSession",
    "default_track",
    "write_training_report",
]

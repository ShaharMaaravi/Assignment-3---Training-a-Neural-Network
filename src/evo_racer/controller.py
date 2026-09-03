from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from numpy.typing import NDArray

from .car import Car
from .track import Track

Genome = tuple[NDArray[np.float32], ...]


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    input_size: int = 10
    hidden_layers: tuple[int, ...] = (16, 12)
    output_size: int = 2

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.output_size != 2:
            raise ValueError("The controller needs positive inputs and exactly two outputs")
        if not self.hidden_layers or any(units <= 0 for units in self.hidden_layers):
            raise ValueError("At least one positive-size hidden layer is required")


@dataclass(frozen=True, slots=True)
class ControlAction:
    steering: float
    throttle: float


class NeuralController:
    """A compact Keras policy whose weights form a genetic-algorithm genome."""

    def __init__(self, config: ControllerConfig | None = None, *, seed: int | None = None):
        self.config = config or ControllerConfig()
        self.seed = seed
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        inputs = keras.Input(shape=(self.config.input_size,), name="car_observation")
        hidden = inputs
        for index, units in enumerate(self.config.hidden_layers):
            layer_seed = None if self.seed is None else self.seed + index
            hidden = keras.layers.Dense(
                units,
                activation="relu",
                kernel_initializer=keras.initializers.GlorotUniform(seed=layer_seed),
                bias_initializer="zeros",
                name=f"hidden_{index + 1}",
            )(hidden)
        output_seed = None if self.seed is None else self.seed + len(self.config.hidden_layers)
        outputs = keras.layers.Dense(
            self.config.output_size,
            activation="tanh",
            kernel_initializer=keras.initializers.GlorotUniform(seed=output_seed),
            bias_initializer="zeros",
            name="controls",
        )(hidden)
        return keras.Model(inputs=inputs, outputs=outputs, name="evo_racer_controller")

    @property
    def parameter_count(self) -> int:
        return self.model.count_params()

    def act(self, observation: Sequence[float]) -> ControlAction:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.config.input_size,):
            raise ValueError(
                f"Expected an observation with shape ({self.config.input_size},), "
                f"received {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Controller observations must contain only finite numbers")

        prediction = np.asarray(self.model(values[None, :], training=False))[0]
        return ControlAction(
            steering=float(np.clip(prediction[0], -1.0, 1.0)),
            throttle=float(np.clip(prediction[1], -1.0, 1.0)),
        )

    def drive(self, car: Car, track: Track, delta_seconds: float) -> ControlAction:
        action = self.act(car.observation(track))
        car.update(action.steering, action.throttle, delta_seconds, track)
        return action

    def get_genome(self) -> Genome:
        return tuple(np.asarray(weight, dtype=np.float32).copy() for weight in self.model.get_weights())

    def set_genome(self, genome: Sequence[NDArray[np.floating]]) -> None:
        expected = self.model.get_weights()
        if len(genome) != len(expected):
            raise ValueError(f"Expected {len(expected)} genome arrays, received {len(genome)}")

        copied: list[NDArray[np.float32]] = []
        for index, (candidate, reference) in enumerate(zip(genome, expected, strict=True)):
            array = np.asarray(candidate, dtype=np.float32)
            if array.shape != reference.shape:
                raise ValueError(
                    f"Genome array {index} has shape {array.shape}; expected {reference.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Genome array {index} contains non-finite values")
            copied.append(array.copy())
        self.model.set_weights(copied)

    def clone(self) -> NeuralController:
        controller = NeuralController(self.config, seed=self.seed)
        controller.set_genome(self.get_genome())
        return controller

    def save_weights(self, path: str | Path) -> Path:
        destination = Path(path)
        if not destination.name.endswith(".weights.h5"):
            raise ValueError("Keras weight files must end in .weights.h5")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(destination)
        return destination

    def load_weights(self, path: str | Path) -> None:
        self.model.load_weights(Path(path))

    def save_model(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix != ".keras":
            raise ValueError("Complete Keras model files must end in .keras")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(destination)
        return destination

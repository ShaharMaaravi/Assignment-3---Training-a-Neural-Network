# EvoRacer

EvoRacer is a top-down racing simulator that will train a Keras neural-network driver with a
genetic algorithm. The current milestone provides the deterministic simulation foundation:

- closed racetrack geometry;
- manual car physics;
- ordered checkpoints and lap counting;
- collision detection using all four car corners;
- seven normalized wall-distance sensors;
- a ten-value observation vector;
- a Keras neural controller with two bounded outputs for steering and throttle;
- immutable genome export, validation, cloning, and weight restoration.

## Setup

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the simulation

```bash
evo-racer
```

Controls:

- Arrow keys or `WASD`: steer, accelerate, and brake
- `V`: show or hide sensor rays
- `R`: reset after a crash
- `Esc`: quit

## Neural controller

The controller architecture is intentionally compact so a genetic algorithm can evaluate many
drivers efficiently:

```text
10 observations -> Dense(16, ReLU) -> Dense(12, ReLU) -> Dense(2, tanh)
```

The two outputs are steering and throttle values in the range `[-1, 1]`. No gradient optimizer is
used: the upcoming genetic algorithm will evolve the 406 Keras parameters directly through
selection, crossover, and mutation.

Keras-compatible champion artifacts will be saved as `*.weights.h5` and complete models as
`*.keras`.

## Genetic training core

Each neural driver runs in a deterministic, headless episode. Its fitness primarily measures
ordered progress toward checkpoints, including partial progress between gates. Completing a lap
adds a large bonus, and finishing sooner adds a smaller time bonus. Crashes receive a penalty;
crashed, stalled, completed, and timed-out drivers stop immediately so training does not waste
simulation time.

The evolution engine supports:

- seeded random populations;
- elitism, which copies the best drivers unchanged;
- tournament selection;
- uniform crossover across both Keras parents' weights;
- Gaussian mutation of individual parameters;
- best, mean, median, and worst fitness statistics for every generation;
- a callback that the upcoming live chart can consume.

## Test

```bash
pytest
ruff check .
```

The next milestone will connect the trainer to a live Pygame visualization, plot fitness by
generation, and save the champion weights automatically.

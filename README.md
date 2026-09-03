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

## Watch the neural drivers evolve

Run the visual trainer:

```bash
evo-racer-train
```

The app races the whole population simultaneously, plots best and mean fitness after each
generation, highlights the current leader, and automatically saves every improved champion to
`artifacts/champion.weights.h5`.

Controls:

- `Space`: pause or resume
- `F`: cycle between 1x, 4x, and 12x training speed
- `V`: show or hide driver trails
- `Esc`: quit (the latest completed champion remains saved)

Training settings can be changed from the command line:

```bash
evo-racer-train --generations 40 --population 50 --max-seconds 35 --seed 2026
```

The next milestone will add a champion replay mode and polish the final training experience.

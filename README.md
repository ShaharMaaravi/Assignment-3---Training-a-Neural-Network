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
- a callback consumed by the live training chart.

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

After every generation, the `artifacts/` directory also receives presentation-ready evidence:

- `training_history.csv`: complete best/mean/median/worst fitness history;
- `training_summary.json`: champion result and reproducible training configuration;
- `fitness_progress.png`: a standalone best-versus-mean fitness graph.

The report files are updated even before training finishes, so safely quitting after a completed
generation preserves the progress collected so far.

Controls:

- `Space`: pause or resume
- `F`: cycle between 1x, 4x, and 12x training speed
- `V`: show or hide driver trails
- `Esc`: quit (the latest completed champion remains saved)

Training settings can be changed from the command line:

```bash
evo-racer-train --generations 40 --population 50 --max-seconds 35 --seed 2026
```

Use `--report-dir another-folder` to place reports somewhere other than the weight directory.
When keeping an existing champion, give the next experiment its own paths:

```bash
evo-racer-train --output artifacts/run-2/champion.weights.h5 --report-dir artifacts/run-2
```

## Replay the champion

After training completes, launch the saved neural driver on its own:

```bash
evo-racer-replay
```

Replay mode loads `artifacts/champion.weights.h5`, exposes the network's live steering and
throttle outputs, renders its seven sensor rays, and reports speed, fitness, checkpoints, laps,
and the final outcome.

Controls:

- `Space`: pause or resume
- `R`: restart the champion from the starting line
- `V`: show or hide sensor rays
- `Esc`: quit

Use a different weight file or a longer challenge when needed:

```bash
evo-racer-replay --weights artifacts/champion.weights.h5 --max-seconds 90 --target-laps 2
```

The trained champion completes the default track. Remaining work is final documentation and demo
preparation.

# EvoRacer

EvoRacer is a top-down racing simulator that will train a Keras neural-network driver with a
genetic algorithm. The current milestone provides the deterministic simulation foundation:

- closed racetrack geometry;
- manual car physics;
- ordered checkpoints and lap counting;
- collision detection using all four car corners;
- seven normalized wall-distance sensors;
- a ten-value observation vector ready for the neural network.

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

## Test

```bash
pytest
ruff check .
```

The next milestone will add the Keras controller, genetic evolution, fitness scoring, and a live
training chart.

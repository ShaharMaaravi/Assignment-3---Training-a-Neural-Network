from __future__ import annotations

import argparse
import json
from pathlib import Path

from evo_racer.car import SENSOR_ANGLES
from evo_racer.replay import ReplaySession


def snapshot(replay: ReplaySession) -> dict[str, object]:
    action = replay.last_action
    car = replay.car
    return {
        "time": round(replay.elapsed_seconds, 4),
        "x": round(car.position.x, 3),
        "y": round(car.position.y, 3),
        "heading": round(car.heading_degrees, 3),
        "speed": round(car.speed, 3),
        "steering": round(action.steering, 5),
        "throttle": round(action.throttle, 5),
        "sensors": [round(value, 3) for value in car.sensor_distances(replay.track)],
        "next_checkpoint": car.next_checkpoint_index,
        "checkpoints": car.checkpoints_passed,
        "laps": car.laps_completed,
        "fitness": round(replay.fitness, 3),
    }


def export(weights: Path, output: Path) -> None:
    replay = ReplaySession.from_weights(weights)
    frames = [snapshot(replay)]
    while replay.active:
        replay.step()
        frames.append(snapshot(replay))

    payload = {
        "version": 1,
        "sample_rate": round(1 / replay.episode_config.time_step),
        "track": {
            "centerline": [
                [round(point.x, 3), round(point.y, 3)] for point in replay.track.centerline
            ],
            "width": replay.track.width,
            "checkpoint_count": replay.track.checkpoint_count,
        },
        "car": {
            "length": replay.car.config.length,
            "width": replay.car.config.width,
            "sensor_range": replay.car.config.sensor_range,
            "sensor_angles": list(SENSOR_ANGLES),
        },
        "result": {
            "outcome": replay.termination_reason,
            "elapsed_seconds": round(replay.elapsed_seconds, 3),
            "fitness": round(replay.fitness, 3),
            "checkpoints": replay.car.checkpoints_passed,
            "laps": replay.car.laps_completed,
            "frame_count": len(frames),
        },
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a champion replay for the web demo")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/champion.weights.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo/champion-telemetry.json"),
    )
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"weights file not found: {args.weights}")
    return args


def main() -> None:
    args = parse_args()
    export(args.weights, args.output)
    print(f"Exported champion telemetry to {args.output}")


if __name__ == "__main__":
    main()

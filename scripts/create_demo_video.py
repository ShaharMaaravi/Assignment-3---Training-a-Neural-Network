from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import imageio_ffmpeg
import pygame

from evo_racer.evolution import EvolutionConfig, EvolutionEngine
from evo_racer.fitness import EpisodeConfig
from evo_racer.replay import ReplaySession
from evo_racer.replay_app import ReplayApp
from evo_racer.training import TrainingSession
from evo_racer.training_app import TrainingApp

VIDEO_SIZE = (1280, 720)
FRAMES_PER_SECOND = 30
BACKGROUND = (14, 10, 31)
PANEL = (32, 24, 63)
GRID = (41, 32, 70)
WHITE = (247, 245, 252)
MUTED = (171, 165, 191)
LIME = (200, 255, 88)
CYAN = (91, 225, 213)
PINK = (255, 104, 151)


class VideoWriter:
    def __init__(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
            "-r",
            str(FRAMES_PER_SECOND),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_count = 0

    def write(self, surface: pygame.Surface, copies: int = 1) -> None:
        if surface.get_size() != VIDEO_SIZE:
            raise ValueError(f"Video frames must have size {VIDEO_SIZE}")
        assert self._process.stdin is not None
        pixels = pygame.image.tobytes(surface, "RGB")
        for _ in range(copies):
            self._process.stdin.write(pixels)
            self.frame_count += 1

    def close(self) -> None:
        assert self._process.stdin is not None
        self._process.stdin.close()
        stderr = self._process.stderr.read().decode("utf-8", errors="replace")
        return_code = self._process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}:\n{stderr}")


def fonts() -> dict[str, pygame.font.Font]:
    return {
        "hero": pygame.font.SysFont("Arial", 58, bold=True),
        "title": pygame.font.SysFont("Arial", 32, bold=True),
        "body": pygame.font.SysFont("Arial", 23),
        "small": pygame.font.SysFont("Arial", 18),
        "metric": pygame.font.SysFont("Arial", 42, bold=True),
    }


def base_frame() -> pygame.Surface:
    surface = pygame.Surface(VIDEO_SIZE)
    surface.fill(BACKGROUND)
    for x in range(0, VIDEO_SIZE[0], 48):
        pygame.draw.line(surface, GRID, (x, 0), (x, VIDEO_SIZE[1]), 1)
    for y in range(0, VIDEO_SIZE[1], 48):
        pygame.draw.line(surface, GRID, (0, y), (VIDEO_SIZE[0], y), 1)
    return surface


def text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    value: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
) -> None:
    surface.blit(font.render(value, True, color), position)


def card(
    title: str,
    subtitle: str,
    lines: Iterable[str],
    app_fonts: dict[str, pygame.font.Font],
) -> pygame.Surface:
    surface = base_frame()
    pygame.draw.rect(surface, PANEL, (72, 82, 1136, 556), border_radius=24)
    pygame.draw.rect(surface, LIME, (72, 82, 12, 556), border_radius=6)
    text(surface, app_fonts["hero"], title, (122, 132), LIME)
    text(surface, app_fonts["body"], subtitle, (126, 214), MUTED)
    y = 306
    for line in lines:
        pygame.draw.circle(surface, CYAN, (137, y + 13), 6)
        text(surface, app_fonts["body"], line, (162, y), WHITE)
        y += 58
    return surface


def framed_app(
    app_surface: pygame.Surface,
    heading: str,
    caption: str,
    app_fonts: dict[str, pygame.font.Font],
) -> pygame.Surface:
    frame = base_frame()
    text(frame, app_fonts["title"], heading, (28, 18), LIME)
    text(frame, app_fonts["small"], caption, (30, 57), MUTED)
    available = pygame.Rect(10, 92, 1260, 610)
    scale = min(
        available.width / app_surface.get_width(),
        available.height / app_surface.get_height(),
    )
    size = (
        round(app_surface.get_width() * scale),
        round(app_surface.get_height() * scale),
    )
    scaled = pygame.transform.smoothscale(app_surface, size)
    frame.blit(scaled, scaled.get_rect(center=available.center))
    return frame


def render_training(writer: VideoWriter, app_fonts: dict[str, pygame.font.Font]) -> None:
    with tempfile.TemporaryDirectory(prefix="evo-racer-demo-") as directory_name:
        directory = Path(directory_name)
        engine = EvolutionEngine(
            EvolutionConfig(
                population_size=10,
                elite_count=2,
                tournament_size=3,
                seed=2026,
            )
        )
        session = TrainingSession(
            engine,
            generations=5,
            episode_config=EpisodeConfig(
                maximum_seconds=1.2,
                idle_timeout_seconds=0.5,
            ),
            champion_path=directory / "champion.weights.h5",
            report_directory=directory,
        )
        app = TrainingApp(session)
        completed_generations = 0
        while not session.complete:
            previous_generation = session.generation
            session.step()
            if session.generation != previous_generation:
                app.trails = app._new_trails()
            else:
                app._record_trails()
            app._draw()
            frame = framed_app(
                app.screen,
                "1. GENETIC TRAINING",
                "Every visible car is controlled by a different Keras genome.",
                app_fonts,
            )
            writer.write(frame)
            if len(session.history) > completed_generations:
                completed_generations = len(session.history)
                writer.write(frame, copies=12)

        app._draw()
        complete_frame = framed_app(
            app.screen,
            "1. GENETIC TRAINING",
            "The chart records real best and mean fitness after every generation.",
            app_fonts,
        )
        writer.write(complete_frame, copies=45)

        assert session.report_paths is not None
        report_frame = report_card(session, app_fonts)
        chart = pygame.image.load(session.report_paths.fitness_chart)
        chart = pygame.transform.smoothscale(chart, (610, 366))
        report_frame.blit(chart, (622, 230))
        writer.write(report_frame, copies=4 * FRAMES_PER_SECOND)
        pygame.display.quit()


def report_card(
    session: TrainingSession,
    app_fonts: dict[str, pygame.font.Font],
) -> pygame.Surface:
    surface = base_frame()
    text(surface, app_fonts["hero"], "2. MODEL + EVIDENCE SAVED", (58, 40), LIME)
    text(
        surface,
        app_fonts["body"],
        "Each completed generation updates reproducible artifacts.",
        (62, 116),
        MUTED,
    )
    pygame.draw.rect(surface, PANEL, (48, 190, 530, 430), border_radius=18)
    outputs = (
        ("champion.weights.h5", "Keras model weights"),
        ("training_history.csv", "Fitness for every generation"),
        ("training_summary.json", "Configuration + champion result"),
        ("fitness_progress.png", "Standalone progress chart"),
    )
    y = 230
    for filename, description in outputs:
        pygame.draw.circle(surface, LIME, (78, y + 14), 7)
        text(surface, app_fonts["body"], filename, (100, y), WHITE)
        text(surface, app_fonts["small"], description, (100, y + 31), MUTED)
        y += 86
    first = session.history[0].best_fitness
    last = session.history[-1].best_fitness
    text(
        surface,
        app_fonts["body"],
        f"Demo best fitness: {first:,.1f}  ->  {last:,.1f}",
        (64, 658),
        CYAN,
    )
    return surface


def render_replay(
    writer: VideoWriter,
    champion_path: Path,
    app_fonts: dict[str, pygame.font.Font],
) -> ReplaySession:
    replay = ReplaySession.from_weights(champion_path)
    app = ReplayApp(replay)
    while replay.active:
        replay.step()
        app._draw()
        frame = framed_app(
            app.screen,
            "3. CHAMPION REPLAY",
            "The saved network drives alone; sensor rays and control outputs are live.",
            app_fonts,
        )
        writer.write(frame)
    app._draw()
    frame = framed_app(
        app.screen,
        "3. CHAMPION REPLAY",
        "All 12 checkpoints were crossed in order.",
        app_fonts,
    )
    writer.write(frame, copies=3 * FRAMES_PER_SECOND)
    pygame.display.quit()
    return replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the EvoRacer demonstration video")
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("artifacts/champion.weights.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo/evoracer-demonstration.mp4"),
    )
    args = parser.parse_args()
    if not args.champion.is_file():
        parser.error(f"champion weights not found: {args.champion}")
    return args


def main() -> None:
    args = parse_args()
    pygame.init()
    app_fonts = fonts()
    writer = VideoWriter(args.output)
    try:
        intro = card(
            "EVORACER",
            "A Keras neural racing driver evolved with a genetic algorithm",
            (
                "Train a population without labelled examples",
                "Measure progress with a purpose-built fitness function",
                "Save and replay the strongest neural driver",
            ),
            app_fonts,
        )
        writer.write(intro, copies=3 * FRAMES_PER_SECOND)
        render_training(writer, app_fonts)

        transition = card(
            "THE CHAMPION",
            "Loading the best saved weights from the full training run",
            (
                "10 sensor and navigation inputs",
                "406 evolved Keras parameters",
                "2 continuous outputs: steering and throttle",
            ),
            app_fonts,
        )
        writer.write(transition, copies=2 * FRAMES_PER_SECOND)
        replay = render_replay(writer, args.champion, app_fonts)

        outcome = "LAP COMPLETE" if replay.termination_reason == "completed" else "RUN COMPLETE"
        outro = card(
            outcome,
            "The saved neural network completed the default EvoRacer track",
            (
                f"Checkpoints crossed: {replay.car.checkpoints_passed}",
                f"Lap time: {replay.elapsed_seconds:.2f} simulated seconds",
                f"Final fitness: {replay.fitness:,.2f}",
            ),
            app_fonts,
        )
        writer.write(outro, copies=3 * FRAMES_PER_SECOND)
    finally:
        try:
            writer.close()
        finally:
            pygame.quit()

    duration = writer.frame_count / FRAMES_PER_SECOND
    print(f"Created {args.output} ({duration:.1f} seconds, {writer.frame_count} frames)")


if __name__ == "__main__":
    main()

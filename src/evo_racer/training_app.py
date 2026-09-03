from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from .evolution import EvolutionConfig, EvolutionEngine, GenerationStats
from .fitness import EpisodeConfig
from .simulation import (
    LIME,
    MUTED,
    PURPLE,
    TRACK_BORDER,
    WHITE,
    draw_background,
    draw_track,
)
from .training import DriverRun, TrainingSession

TRACK_VIEW_WIDTH = 1280
SIDEBAR_WIDTH = 360
WINDOW_HEIGHT = 780
WINDOW_SIZE = (TRACK_VIEW_WIDTH + SIDEBAR_WIDTH, WINDOW_HEIGHT)
CYAN = (91, 225, 213)
PINK = (255, 104, 151)
RED = (232, 69, 94)
PANEL = (32, 24, 63)
GRID = (67, 56, 94)
SPEEDS = (1, 4, 12)


class TrainingApp:
    def __init__(
        self,
        session: TrainingSession,
        *,
        exit_on_complete: bool = False,
    ):
        pygame.init()
        pygame.display.set_caption("EvoRacer - Genetic Neural Network Training")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.session = session
        self.exit_on_complete = exit_on_complete
        self.paused = False
        self.show_trails = True
        self.speed_index = 1
        self.running = True
        self.fonts = {
            "hero": pygame.font.SysFont("Arial", 32, bold=True),
            "title": pygame.font.SysFont("Arial", 21, bold=True),
            "body": pygame.font.SysFont("Arial", 16),
            "small": pygame.font.SysFont("Arial", 13),
            "metric": pygame.font.SysFont("Arial", 27, bold=True),
        }
        self.trails = self._new_trails()

    def run(self) -> None:
        while self.running:
            self.clock.tick(30)
            self._handle_events()
            if not self.paused and not self.session.complete:
                previous_generation = self.session.generation
                for _ in range(SPEEDS[self.speed_index]):
                    self.session.step()
                    if self.session.generation != previous_generation or self.session.complete:
                        break
                if self.session.generation != previous_generation:
                    self.trails = self._new_trails()
                else:
                    self._record_trails()
            self._draw()
            pygame.display.flip()
            if self.exit_on_complete and self.session.complete:
                self.running = False
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_f:
                    self.speed_index = (self.speed_index + 1) % len(SPEEDS)
                elif event.key == pygame.K_v:
                    self.show_trails = not self.show_trails

    def _new_trails(self) -> list[list[pygame.Vector2]]:
        return [[run.car.position.copy()] for run in self.session.runs]

    def _record_trails(self) -> None:
        for trail, run in zip(self.trails, self.session.runs, strict=True):
            if trail[-1].distance_to(run.car.position) >= 5.0:
                trail.append(run.car.position.copy())
                if len(trail) > 55:
                    del trail[0]

    def _draw(self) -> None:
        track_surface = self.screen.subsurface((0, 0, TRACK_VIEW_WIDTH, WINDOW_HEIGHT))
        draw_background(track_surface)
        draw_track(track_surface, self.session.track, self.session.leader.car.next_checkpoint_index)
        self._draw_trails(track_surface)
        self._draw_cars(track_surface)
        self._draw_track_hud(track_surface)
        self._draw_sidebar()
        if self.session.complete:
            self._draw_complete_overlay(track_surface)

    def _draw_trails(self, surface: pygame.Surface) -> None:
        if not self.show_trails:
            return
        overlay = pygame.Surface((TRACK_VIEW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        for trail in self.trails:
            if len(trail) >= 2:
                pygame.draw.lines(overlay, (*CYAN, 65), False, trail, 2)
        surface.blit(overlay, (0, 0))

    def _draw_cars(self, surface: pygame.Surface) -> None:
        leader_index = max(
            range(len(self.session.runs)),
            key=lambda index: self.session.runs[index].maximum_progress,
        )
        for index, run in enumerate(self.session.runs):
            highlighted = index == leader_index
            color = LIME if highlighted else PURPLE
            if run.termination_reason == "crashed":
                color = RED
            elif not run.active:
                color = MUTED
            self._draw_driver(surface, run, color, highlighted)

    @staticmethod
    def _draw_driver(
        surface: pygame.Surface,
        run: DriverRun,
        color: tuple[int, int, int],
        highlighted: bool,
    ) -> None:
        points = [(round(point.x), round(point.y)) for point in run.car.corners()]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, WHITE if highlighted else TRACK_BORDER, points, 3 if highlighted else 1)
        nose = run.car.position + run.car.heading_vector * (run.car.config.length * 0.34)
        pygame.draw.circle(surface, WHITE, nose, 3)

    def _draw_track_hud(self, surface: pygame.Surface) -> None:
        panel = pygame.Surface((1220, 82), pygame.SRCALPHA)
        panel.fill((*PANEL, 240))
        surface.blit(panel, (30, 680))
        self._text(surface, "LIVE EVOLUTION", (55, 698), "title", LIME)
        status = "PAUSED" if self.paused else "TRAINING"
        if self.session.complete:
            status = "COMPLETE"
        self._text(surface, status, (55, 728), "small", MUTED)

        metrics = (
            ("GENERATION", f"{self.session.generation}/{self.session.total_generations}"),
            ("ACTIVE", f"{self.session.active_count}/{len(self.session.runs)}"),
            ("TIME", f"{self.session.elapsed_seconds:04.1f}s"),
            ("LEADER", f"{self.session.leader.maximum_progress:.2f} gates"),
        )
        x = 300
        for label, value in metrics:
            self._text(surface, label, (x, 696), "small", MUTED)
            self._text(surface, value, (x, 719), "title", WHITE)
            x += 175

        self._text(
            surface,
            f"SPACE pause   F speed: {SPEEDS[self.speed_index]}x   V trails   ESC quit",
            (890, 742),
            "small",
            MUTED,
        )

    def _draw_sidebar(self) -> None:
        sidebar = pygame.Rect(TRACK_VIEW_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, (22, 16, 44), sidebar)
        pygame.draw.line(
            self.screen,
            TRACK_BORDER,
            (TRACK_VIEW_WIDTH, 0),
            (TRACK_VIEW_WIDTH, WINDOW_HEIGHT),
            3,
        )
        left = TRACK_VIEW_WIDTH + 28
        self._text(self.screen, "EVORACER AI LAB", (left, 30), "hero", LIME)
        self._text(self.screen, "Neural evolution telemetry", (left, 70), "body", MUTED)

        best = self.session.champion_episode
        best_fitness = best.fitness if best is not None else 0.0
        cards = (
            ("BEST FITNESS", f"{best_fitness:,.1f}", LIME),
            ("COMPLETED GENS", str(len(self.session.history)), CYAN),
            ("CHAMPION GATES", str(best.checkpoints_passed if best else 0), PINK),
        )
        y = 112
        for label, value, color in cards:
            pygame.draw.rect(self.screen, PANEL, (left, y, 304, 66), border_radius=10)
            self._text(self.screen, label, (left + 14, y + 10), "small", MUTED)
            self._text(self.screen, value, (left + 14, y + 29), "metric", color)
            y += 78

        self._draw_chart((left, 362, 304, 218))
        self._text(self.screen, "BEST", (left + 12, 592), "small", LIME)
        self._text(self.screen, "MEAN", (left + 87, 592), "small", CYAN)

        if self.session.history:
            latest = self.session.history[-1]
            self._draw_latest_stats(left, 626, latest)
        else:
            self._text(self.screen, "The first generation is racing...", (left, 632), "body", MUTED)

        self._text(self.screen, "AUTO-SAVED CHAMPION", (left, 716), "small", MUTED)
        self._text(self.screen, self.session.champion_path.name, (left, 738), "body", WHITE)

    def _draw_chart(self, bounds: tuple[int, int, int, int]) -> None:
        rect = pygame.Rect(bounds)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        self._text(self.screen, "FITNESS BY GENERATION", (rect.x + 12, rect.y + 10), "small", WHITE)
        chart = pygame.Rect(rect.x + 14, rect.y + 38, rect.width - 28, rect.height - 54)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = round(chart.bottom - chart.height * fraction)
            pygame.draw.line(self.screen, GRID, (chart.left, y), (chart.right, y), 1)

        if not self.session.history:
            return
        maximum = max(stats.best_fitness for stats in self.session.history)
        maximum = max(1.0, maximum * 1.08)
        best_points = self._chart_points(chart, self.session.history, maximum, "best_fitness")
        mean_points = self._chart_points(chart, self.session.history, maximum, "mean_fitness")
        self._draw_series(best_points, LIME)
        self._draw_series(mean_points, CYAN)

    @staticmethod
    def _chart_points(
        chart: pygame.Rect,
        history: list[GenerationStats],
        maximum: float,
        attribute: str,
    ) -> list[tuple[int, int]]:
        denominator = max(1, len(history) - 1)
        return [
            (
                round(chart.left + chart.width * index / denominator),
                round(chart.bottom - chart.height * getattr(stats, attribute) / maximum),
            )
            for index, stats in enumerate(history)
        ]

    def _draw_series(self, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
        if len(points) >= 2:
            pygame.draw.lines(self.screen, color, False, points, 3)
        for point in points:
            pygame.draw.circle(self.screen, color, point, 4)

    def _draw_latest_stats(self, left: int, top: int, latest: GenerationStats) -> None:
        self._text(self.screen, f"Generation {latest.generation} result", (left, top), "title", WHITE)
        self._text(
            self.screen,
            f"Best {latest.best_fitness:,.1f}  •  Mean {latest.mean_fitness:,.1f}",
            (left, top + 30),
            "body",
            MUTED,
        )
        self._text(
            self.screen,
            f"{latest.best_checkpoints} gates  •  {latest.best_termination_reason}",
            (left, top + 54),
            "body",
            MUTED,
        )

    def _draw_complete_overlay(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((TRACK_VIEW_WIDTH, 680), pygame.SRCALPHA)
        overlay.fill((12, 7, 25, 155))
        surface.blit(overlay, (0, 0))
        title = self.fonts["hero"].render("TRAINING COMPLETE", True, LIME)
        subtitle = self.fonts["title"].render(
            f"Champion saved to {self.session.champion_path}",
            True,
            WHITE,
        )
        surface.blit(title, title.get_rect(center=(640, 304)))
        surface.blit(subtitle, subtitle.get_rect(center=(640, 350)))

    def _text(
        self,
        surface: pygame.Surface,
        value: str,
        position: tuple[int, int],
        font: str,
        color: tuple[int, int, int],
    ) -> None:
        surface.blit(self.fonts[font].render(value, True, color), position)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch EvoRacer neural drivers evolve live")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/champion.weights.h5"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="report directory (defaults to the champion weight directory)",
    )
    parser.add_argument("--exit-on-complete", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.generations <= 0:
        parser.error("--generations must be positive")
    if args.population < 2:
        parser.error("--population must be at least 2")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    elite_count = max(1, min(args.population - 1, args.population // 8))
    engine = EvolutionEngine(
        EvolutionConfig(
            population_size=args.population,
            elite_count=elite_count,
            tournament_size=min(3, args.population),
            seed=args.seed,
        )
    )
    session = TrainingSession(
        engine,
        generations=args.generations,
        episode_config=EpisodeConfig(maximum_seconds=args.max_seconds),
        champion_path=args.output,
        report_directory=args.report_dir,
    )
    TrainingApp(session, exit_on_complete=args.exit_on_complete).run()


if __name__ == "__main__":
    main()

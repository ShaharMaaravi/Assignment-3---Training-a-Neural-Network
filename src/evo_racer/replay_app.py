from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from .fitness import EpisodeConfig
from .replay import ReplaySession
from .simulation import (
    LIME,
    MUTED,
    WHITE,
    WINDOW_SIZE,
    draw_background,
    draw_car,
    draw_sensor_rays,
    draw_track,
)

CYAN = (91, 225, 213)
PINK = (255, 104, 151)
RED = (255, 104, 124)
PANEL = (32, 24, 63)


class ReplayApp:
    def __init__(self, session: ReplaySession, *, exit_on_finish: bool = False):
        pygame.init()
        pygame.display.set_caption("EvoRacer - Champion Replay")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.session = session
        self.exit_on_finish = exit_on_finish
        self.paused = False
        self.show_sensors = True
        self.running = True
        self.fonts = {
            "hero": pygame.font.SysFont("Arial", 38, bold=True),
            "title": pygame.font.SysFont("Arial", 22, bold=True),
            "body": pygame.font.SysFont("Arial", 16),
            "small": pygame.font.SysFont("Arial", 13),
            "metric": pygame.font.SysFont("Arial", 26, bold=True),
        }

    def run(self) -> None:
        while self.running:
            self.clock.tick(30)
            self._handle_events()
            if not self.paused and self.session.active:
                self.session.step()
            self._draw()
            pygame.display.flip()
            if self.exit_on_finish and not self.session.active:
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
                elif event.key == pygame.K_r:
                    self.session.reset()
                    self.paused = False
                elif event.key == pygame.K_v:
                    self.show_sensors = not self.show_sensors

    def _draw(self) -> None:
        draw_background(self.screen)
        draw_track(self.screen, self.session.track, self.session.car.next_checkpoint_index)
        if self.show_sensors:
            draw_sensor_rays(self.screen, self.session.car, self.session.track)
        draw_car(self.screen, self.session.car)
        self._draw_network_panel()
        self._draw_hud()
        if not self.session.active:
            self._draw_result()

    def _draw_network_panel(self) -> None:
        panel = pygame.Surface((330, 152), pygame.SRCALPHA)
        panel.fill((*PANEL, 238))
        self.screen.blit(panel, (30, 28))
        self._text("CHAMPION BRAIN", (49, 44), "title", LIME)
        filename = self.session.weights_path.name if self.session.weights_path else "in-memory weights"
        self._text(filename, (49, 72), "small", MUTED)
        self._draw_action_bar("STEERING", self.session.last_action.steering, 49, 105, CYAN)
        self._draw_action_bar("THROTTLE", self.session.last_action.throttle, 49, 140, PINK)

    def _draw_action_bar(
        self,
        label: str,
        value: float,
        left: int,
        top: int,
        color: tuple[int, int, int],
    ) -> None:
        self._text(label, (left, top - 13), "small", MUTED)
        bounds = pygame.Rect(left + 90, top - 8, 168, 10)
        pygame.draw.rect(self.screen, (69, 59, 92), bounds, border_radius=5)
        center_x = bounds.centerx
        value_x = round(center_x + value * bounds.width / 2)
        fill = pygame.Rect(min(center_x, value_x), bounds.y, abs(value_x - center_x), bounds.height)
        if fill.width:
            pygame.draw.rect(self.screen, color, fill, border_radius=5)
        pygame.draw.line(self.screen, WHITE, (center_x, bounds.y - 2), (center_x, bounds.bottom + 2), 1)
        self._text(f"{value:+.2f}", (left + 266, top - 13), "small", color)

    def _draw_hud(self) -> None:
        panel = pygame.Surface((1220, 82), pygame.SRCALPHA)
        panel.fill((*PANEL, 240))
        self.screen.blit(panel, (30, 680))
        self._text("CHAMPION REPLAY", (55, 698), "title", LIME)
        status = "PAUSED" if self.paused else "AUTONOMOUS"
        if not self.session.active:
            status = self.session.termination_reason.upper()
        self._text(status, (55, 728), "small", MUTED)

        metrics = (
            ("SPEED", f"{abs(self.session.car.speed):.0f}"),
            ("FITNESS", f"{self.session.fitness:,.1f}"),
            ("GATES", str(self.session.car.checkpoints_passed)),
            ("LAPS", str(self.session.car.laps_completed)),
            ("TIME", f"{self.session.elapsed_seconds:.1f}s"),
        )
        x = 285
        for label, value in metrics:
            self._text(label, (x, 696), "small", MUTED)
            self._text(value, (x, 717), "metric", WHITE)
            x += 140
        self._text("SPACE pause   R replay   V sensors   ESC quit", (890, 740), "small", MUTED)

    def _draw_result(self) -> None:
        overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((12, 7, 25, 110))
        self.screen.blit(overlay, (0, 0))
        completed = self.session.termination_reason == "completed"
        title_value = "LAP COMPLETE!" if completed else self.session.termination_reason.upper()
        title_color = LIME if completed else RED
        title = self.fonts["hero"].render(title_value, True, title_color)
        details = self.fonts["title"].render(
            f"Fitness {self.session.fitness:,.1f}  |  "
            f"Gates {self.session.car.checkpoints_passed}  |  Press R to replay",
            True,
            WHITE,
        )
        self.screen.blit(title, title.get_rect(center=(640, 315)))
        self.screen.blit(details, details.get_rect(center=(640, 365)))

    def _text(
        self,
        value: str,
        position: tuple[int, int],
        font: str,
        color: tuple[int, int, int],
    ) -> None:
        self.screen.blit(self.fonts[font].render(value, True, color), position)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay the evolved EvoRacer champion")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/champion.weights.h5"),
    )
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--exit-on-finish", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"weights file not found: {args.weights}")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.target_laps <= 0:
        parser.error("--target-laps must be positive")
    session = ReplaySession.from_weights(
        args.weights,
        episode_config=EpisodeConfig(
            maximum_seconds=args.max_seconds,
            target_laps=args.target_laps,
        ),
    )
    ReplayApp(session, exit_on_finish=args.exit_on_finish).run()


if __name__ == "__main__":
    main()

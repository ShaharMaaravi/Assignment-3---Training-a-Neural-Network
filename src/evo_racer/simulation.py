from __future__ import annotations

import pygame

from .car import SENSOR_ANGLES, Car
from .track import Track, default_track

WINDOW_SIZE = (1280, 780)
BACKGROUND = (18, 13, 38)
TRACK_BORDER = (109, 78, 190)
TRACK_SURFACE = (57, 55, 71)
TRACK_CENTER = (126, 121, 145)
LIME = (200, 255, 88)
PURPLE = (118, 78, 238)
WHITE = (247, 245, 252)
MUTED = (171, 165, 191)


def _draw_thick_closed_line(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    points: tuple[pygame.Vector2, ...],
    width: int,
) -> None:
    integer_points = [(round(point.x), round(point.y)) for point in points]
    pygame.draw.lines(surface, color, True, integer_points, width)
    radius = width // 2
    for point in integer_points:
        pygame.draw.circle(surface, color, point, radius)


def _draw_dashed_centerline(surface: pygame.Surface, track: Track) -> None:
    for index, start in enumerate(track.centerline):
        end = track.centerline[(index + 1) % track.checkpoint_count]
        segment = end - start
        length = segment.length()
        direction = segment.normalize()
        cursor = 0.0
        while cursor < length:
            dash_end = min(cursor + 13.0, length)
            pygame.draw.line(
                surface,
                TRACK_CENTER,
                start + direction * cursor,
                start + direction * dash_end,
                2,
            )
            cursor += 28.0


def draw_track(surface: pygame.Surface, track: Track, next_checkpoint: int) -> None:
    _draw_thick_closed_line(surface, TRACK_BORDER, track.centerline, round(track.width + 14))
    _draw_thick_closed_line(surface, TRACK_SURFACE, track.centerline, round(track.width))
    _draw_dashed_centerline(surface, track)

    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for index in range(track.checkpoint_count):
        start, end = track.checkpoint_segment(index, inset=9)
        color = (*LIME, 180) if index == next_checkpoint else (255, 255, 255, 32)
        pygame.draw.line(overlay, color, start, end, 4 if index == next_checkpoint else 2)
        label = pygame.font.Font(None, 20).render(str(index), True, color[:3])
        overlay.blit(label, track.centerline[index] + pygame.Vector2(7, 7))
    surface.blit(overlay, (0, 0))


def draw_sensor_rays(surface: pygame.Surface, car: Car, track: Track) -> None:
    for relative_angle, distance in zip(SENSOR_ANGLES, car.sensor_distances(track), strict=True):
        end = car.position + car.heading_vector.rotate(relative_angle) * distance
        pygame.draw.line(surface, (125, 244, 217), car.position, end, 2)
        pygame.draw.circle(surface, LIME, end, 4)


def draw_car(surface: pygame.Surface, car: Car) -> None:
    points = [(round(point.x), round(point.y)) for point in car.corners()]
    shadow = [(x + 5, y + 6) for x, y in points]
    pygame.draw.polygon(surface, (16, 10, 29), shadow)
    pygame.draw.polygon(surface, (232, 69, 94) if car.crashed else PURPLE, points)
    pygame.draw.polygon(surface, WHITE, points, 2)
    nose = car.position + car.heading_vector * (car.config.length * 0.34)
    pygame.draw.circle(surface, LIME, nose, 4)


def _text(
    surface: pygame.Surface,
    value: str,
    position: tuple[int, int],
    size: int,
    color: tuple[int, int, int] = WHITE,
    bold: bool = False,
) -> None:
    font = pygame.font.SysFont("Arial", size, bold=bold)
    surface.blit(font.render(value, True, color), position)


def draw_hud(surface: pygame.Surface, car: Car, track: Track, sensors_visible: bool) -> None:
    panel = pygame.Surface((1220, 82), pygame.SRCALPHA)
    panel.fill((32, 24, 63, 235))
    surface.blit(panel, (30, 680))
    _text(surface, "EVORACER", (55, 697), 25, LIME, True)
    _text(surface, "Simulation foundation", (55, 727), 16, MUTED)

    stats = (
        ("SPEED", f"{abs(car.speed):.0f}"),
        ("NEXT GATE", f"{car.next_checkpoint_index}/{track.checkpoint_count - 1}"),
        ("PASSED", str(car.checkpoints_passed)),
        ("LAPS", str(car.laps_completed)),
    )
    x = 335
    for label, value in stats:
        _text(surface, label, (x, 696), 13, MUTED, True)
        _text(surface, value, (x, 718), 25, WHITE, True)
        x += 140

    sensor_label = "ON" if sensors_visible else "OFF"
    _text(surface, f"ARROWS/WASD drive   V sensors: {sensor_label}   R reset   ESC quit", (870, 713), 15, MUTED)


def draw_background(surface: pygame.Surface) -> None:
    surface.fill(BACKGROUND)
    for x in range(0, WINDOW_SIZE[0], 40):
        pygame.draw.line(surface, (26, 20, 49), (x, 0), (x, 680), 1)
    for y in range(0, 680, 40):
        pygame.draw.line(surface, (26, 20, 49), (0, y), (WINDOW_SIZE[0], y), 1)


def draw_crash_message(surface: pygame.Surface) -> None:
    overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
    overlay.fill((12, 7, 25, 115))
    surface.blit(overlay, (0, 0))
    title = pygame.font.SysFont("Arial", 52, bold=True).render("CRASHED", True, (255, 104, 124))
    subtitle = pygame.font.SysFont("Arial", 22, bold=True).render("Press R to reset", True, WHITE)
    surface.blit(title, title.get_rect(center=(640, 325)))
    surface.blit(subtitle, subtitle.get_rect(center=(640, 370)))


def run() -> None:
    pygame.init()
    pygame.display.set_caption("EvoRacer - Simulation Foundation")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    track = default_track()
    car = Car.at_track_start(track)
    sensors_visible = True
    running = True

    while running:
        delta_seconds = min(clock.tick(60) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    car.reset(track)
                elif event.key == pygame.K_v:
                    sensors_visible = not sensors_visible

        keys = pygame.key.get_pressed()
        throttle = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(
            keys[pygame.K_DOWN] or keys[pygame.K_s]
        )
        steering = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(
            keys[pygame.K_LEFT] or keys[pygame.K_a]
        )
        car.update(steering, throttle, delta_seconds, track)

        draw_background(screen)
        draw_track(screen, track, car.next_checkpoint_index)
        if sensors_visible:
            draw_sensor_rays(screen, car, track)
        draw_car(screen, car)
        draw_hud(screen, car, track, sensors_visible)
        if car.crashed:
            draw_crash_message(screen)
        pygame.display.flip()

    pygame.quit()


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()

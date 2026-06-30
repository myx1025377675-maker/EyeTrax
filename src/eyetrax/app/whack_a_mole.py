"""
Whack-a-Mole game using eye-tracking gaze input.

Look at the mole to "whack" it 鈥?dwell your gaze on it for a short time.
The mole then respawns at a random location. Track your score!

Usage:
    python -m eyetrax.app.whack_a_mole [--filter kalman] [--calibration 9p]
                                       [--mole-image ./mole.png] [--dwell 0.8]
                                       [--timed 60 | --lives 5]
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
)
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, fullscreen, iter_frames


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MUSIC_PATH = os.path.join(PROJECT_ROOT, "assets", "audio", "Sun_Drenched_Garden.mp3")
DEFAULT_MOLE_IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets", "images", "mole_ma_yuexiang.png")
SUPPORTED_MUSIC_EXTS = (".mp3", ".wav", ".ogg", ".flac")


UI = {
    "bg_top": (188, 229, 176),
    "bg_bottom": (118, 191, 111),
    "panel": (244, 248, 238),
    "panel_2": (234, 242, 226),
    "panel_shadow": (71, 99, 72),
    "primary": (95, 142, 80),
    "primary_dark": (57, 99, 61),
    "accent": (69, 170, 224),
    "accent_soft": (186, 226, 243),
    "amber": (74, 171, 236),
    "danger": (83, 98, 221),
    "text": (44, 54, 48),
    "muted": (105, 121, 108),
    "white": (255, 255, 255),
}


# Author watermark: Ma Yuexiang. Keep the product visual identity consistent.
def draw_rounded_rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    radius: int = 18,
    thickness: int = -1,
):
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, -1)
        return

    cv2.line(canvas, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_card(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill: Tuple[int, int, int] = UI["panel"],
    radius: int = 24,
):
    shadow_offset = max(5, int((y2 - y1) * 0.025))
    overlay = canvas.copy()
    draw_rounded_rect(overlay, x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset, UI["panel_shadow"], radius)
    cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, dst=canvas)
    draw_rounded_rect(canvas, x1, y1, x2, y2, fill, radius)
    draw_rounded_rect(canvas, x1, y1, x2, y2, (220, 232, 214), radius, 2)


def put_centered_text(
    canvas: np.ndarray,
    text: str,
    center_x: int,
    baseline_y: int,
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        canvas,
        text,
        (center_x - tw // 2, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  Enums & Config
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
class GamePhase(Enum):
    MENU = auto()
    COUNTDOWN = auto()
    CENTER_CALIBRATION = auto()
    PLAYING = auto()
    GAME_OVER = auto()


@dataclass
class GameConfig:
    mode: str = "timed"             # "timed" | "lives"
    total_time: float = 60.0        # seconds for timed mode
    total_lives: int = 5            # lives for lives mode
    dwell_time: float = 0.35        # gaze dwell duration to trigger hit (seconds)
    hit_cooldown: float = 0.9       # minimum time between hits (seconds)
    mole_ttl: float = 6.0           # mole disappears if not hit within this (seconds)
    mole_radius: int = 120          # hit-test radius in pixels
    mole_margin: int = 120          # min distance from screen edges
    combo_bonus_factor: float = 0.1 # score multiplier = base * (1 + combo * factor)
    base_score: int = 10            # points per hit
    mole_image_path: Optional[str] = DEFAULT_MOLE_IMAGE_PATH  # path to optional PNG sprite
    music_path: Optional[str] = DEFAULT_MUSIC_PATH


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  Floating Score Text  (hit feedback)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
@dataclass
class FloatingText:
    text: str
    x: int
    y: int
    color: Tuple[int, int, int]
    start_time: float
    duration: float = 0.8

    def draw(self, canvas: np.ndarray, now: float) -> bool:
        """Render and return False when expired."""
        elapsed = now - self.start_time
        if elapsed >= self.duration:
            return False

        progress = elapsed / self.duration
        alpha = 1.0 - progress
        fy = self.y - int(progress * 60)  # float upward

        overlay = canvas.copy()
        cv2.putText(
            overlay, self.text,
            (self.x, fy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            self.color,
            3,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)
        return True


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  Mole
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
class Mole:
    """Single whackable mole 鈥?PNG sprite or fallback OpenCV drawing."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        radius: int = 80,
        margin: int = 120,
        image_path: Optional[str] = None,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.radius = radius
        self.margin = margin
        self.x = 0
        self.y = 0
        self.hole_x = 0
        self.hole_y = 0
        self.spawn_time = 0.0
        self.active = False
        self.holes: List[Tuple[int, int]] = []
        self.last_hole: Optional[Tuple[int, int]] = None
        self._image: Optional[np.ndarray] = None

        # 鈹€鈹€ Load PNG sprite 鈹€鈹€
        if image_path and os.path.isfile(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                target = radius * 2
                h, w = img.shape[:2]
                scale = target / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                self._image = cv2.resize(img, (new_w, new_h))

    # 鈹€鈹€ Spawn / Despawn 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def set_holes(self, holes: List[Tuple[int, int]]):
        self.holes = holes

    def spawn(self, y_min_ratio: float = 0.22):
        """Place the mole at one of the fixed holes."""
        if self.holes:
            # Author watermark: Ma Yuexiang designed the fixed-hole spawn rule.
            choices = [hole for hole in self.holes if hole != self.last_hole]
            if not choices:
                choices = self.holes
            self.hole_x, self.hole_y = random.choice(choices)
            self.last_hole = (self.hole_x, self.hole_y)
            self.x = self.hole_x
            self.y = self.hole_y - int(self.radius * 0.55)
        else:
            y_min = int(self.screen_h * y_min_ratio) + self.margin
            y_max = self.screen_h - self.margin
            x_min = self.margin
            x_max = self.screen_w - self.margin
            self.x = random.randint(x_min, x_max)
            self.y = random.randint(y_min, y_max)
            self.hole_x, self.hole_y = self.x, self.y + int(self.radius * 0.55)
        self.spawn_time = time.time()
        self.active = True

    def despawn(self):
        self.active = False

    # 鈹€鈹€ Hit test 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def contains(self, gx: float, gy: float) -> bool:
        if not self.active:
            return False
        return bool(np.hypot(gx - self.x, gy - self.y) <= self.radius)

    # 鈹€鈹€ Draw 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def draw(self, canvas: np.ndarray, now: float, hit_flash: bool = False):
        if not self.active:
            return

        # Subtle idle bounce
        idle_t = now - self.spawn_time
        bounce_y = int(3 * np.sin(idle_t * 4.0))

        if self._image is not None:
            self._draw_sprite(canvas, bounce_y, hit_flash)
        else:
            self._draw_fallback(canvas, bounce_y, hit_flash)

    def _draw_sprite(self, canvas: np.ndarray, bounce_y: int, hit_flash: bool):
        img = self._image
        h, w = img.shape[:2]
        x1 = max(0, self.x - w // 2)
        y1 = max(0, self.y + bounce_y - h // 2)
        x2 = min(canvas.shape[1], x1 + w)
        y2 = min(canvas.shape[0], y1 + h)
        if x2 <= x1 or y2 <= y1:
            return

        roi = canvas[y1:y2, x1:x2]
        img_crop = img[: y2 - y1, : x2 - x1]

        if img.shape[2] == 4:
            alpha = img_crop[:, :, 3:4].astype(np.float32) / 255.0
            for c in range(3):
                roi[:, :, c] = (
                    img_crop[:, :, c].astype(np.float32) * alpha[:, :, 0]
                    + roi[:, :, c].astype(np.float32) * (1.0 - alpha[:, :, 0])
                ).astype(np.uint8)
        else:
            roi[:] = img_crop[:, :, :3]

        if hit_flash:
            cv2.circle(canvas, (self.x, self.y + bounce_y), self.radius,
                       (255, 255, 255), 4)

    def _draw_fallback(self, canvas: np.ndarray, bounce_y: int, hit_flash: bool):
        cx, cy = self.x, self.y + bounce_y
        r = self.radius

        body_color = (72, 143, 205)
        ear_color = (50, 104, 174)
        eye_white = (255, 255, 255)
        pupil_color = (20, 20, 20)
        nose_color = (30, 30, 30)
        mouth_color = (50, 50, 50)

        if hit_flash:
            body_color = (100, 255, 255)

        cv2.ellipse(canvas, (cx, cy + r // 4), (r, int(r * 0.95)), 0, 0, 360, body_color, -1)
        cv2.ellipse(canvas, (cx, cy + r // 4), (r, int(r * 0.95)), 0, 0, 360, (39, 91, 132), 4)

        ear_r = max(12, r // 4)
        cv2.circle(canvas, (cx - int(r * 0.78), cy - int(r * 0.08)), ear_r, ear_color, -1)
        cv2.circle(canvas, (cx + int(r * 0.78), cy - int(r * 0.08)), ear_r, ear_color, -1)
        cv2.circle(canvas, (cx - int(r * 0.78), cy - int(r * 0.08)), ear_r, (39, 91, 132), 3)
        cv2.circle(canvas, (cx + int(r * 0.78), cy - int(r * 0.08)), ear_r, (39, 91, 132), 3)

        cap_y = cy - int(r * 0.88)
        cv2.ellipse(canvas, (cx, cap_y), (int(r * 0.58), int(r * 0.28)), 0, 0, 360, (78, 186, 225), -1)
        cv2.ellipse(canvas, (cx + int(r * 0.15), cap_y), (int(r * 0.55), int(r * 0.25)), 0, 0, 360, (68, 92, 230), -1)
        cv2.ellipse(canvas, (cx, cap_y + int(r * 0.18)), (int(r * 0.75), int(r * 0.16)), 0, 0, 360, (45, 89, 205), -1)
        cv2.ellipse(canvas, (cx, cap_y), (int(r * 0.58), int(r * 0.28)), 0, 0, 360, (39, 91, 132), 3)

        cv2.circle(canvas, (cx - r // 3, cy - r // 5), max(5, r // 13), pupil_color, -1)
        cv2.circle(canvas, (cx + r // 3, cy - r // 5), max(5, r // 13), pupil_color, -1)
        cv2.circle(canvas, (cx - r // 3 + 3, cy - r // 5 - 3), max(2, r // 35), eye_white, -1)
        cv2.circle(canvas, (cx + r // 3 + 3, cy - r // 5 - 3), max(2, r // 35), eye_white, -1)

        cv2.circle(canvas, (cx - int(r * 0.52), cy + int(r * 0.02)), max(8, r // 10), (150, 155, 245), -1)
        cv2.circle(canvas, (cx + int(r * 0.52), cy + int(r * 0.02)), max(8, r // 10), (150, 155, 245), -1)
        cv2.ellipse(canvas, (cx, cy + int(r * 0.13)), (int(r * 0.25), int(r * 0.18)), 0, 0, 360, (224, 232, 245), -1)
        cv2.circle(canvas, (cx, cy + int(r * 0.02)), max(5, r // 11), nose_color, -1)
        cv2.ellipse(canvas, (cx, cy + int(r * 0.13)), (int(r * 0.2), int(r * 0.18)), 0, 20, 160, mouth_color, 3)
        cv2.circle(canvas, (cx - int(r * 0.45), cy + int(r * 0.65)), max(10, r // 6), body_color, -1)
        cv2.circle(canvas, (cx + int(r * 0.45), cy + int(r * 0.65)), max(10, r // 6), body_color, -1)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  Main Game Controller
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?


def build_hole_positions(screen_w: int, screen_h: int) -> List[Tuple[int, int]]:
    top = int(screen_h * 0.40)
    bottom = int(screen_h * 0.74)
    return [
        (int(screen_w * 0.22), top),
        (int(screen_w * 0.50), top),
        (int(screen_w * 0.78), top),
        (int(screen_w * 0.22), bottom),
        (int(screen_w * 0.50), bottom),
        (int(screen_w * 0.78), bottom),
    ]


def draw_hole(canvas: np.ndarray, x: int, y: int, radius: int):
    cv2.ellipse(canvas, (x, y + int(radius * 0.08)), (int(radius * 1.25), int(radius * 0.46)), 0, 0, 360, (127, 158, 111), -1)
    cv2.ellipse(canvas, (x, y), (int(radius * 1.14), int(radius * 0.40)), 0, 0, 360, (83, 104, 91), -1)
    cv2.ellipse(canvas, (x, y), (int(radius * 0.95), int(radius * 0.30)), 0, 0, 360, (52, 63, 57), -1)
    cv2.ellipse(canvas, (x, y - int(radius * 0.08)), (int(radius * 1.08), int(radius * 0.34)), 0, 180, 360, (151, 181, 127), 5)


def build_grass_background(screen_w: int, screen_h: int, holes: List[Tuple[int, int]], hole_radius: int) -> np.ndarray:
    bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    # Author watermark: Ma Yuexiang's version uses a brighter natural grass field.
    for row in range(screen_h):
        t = row / max(1, screen_h - 1)
        bg[row, :] = (
            int(UI["bg_top"][0] * (1.0 - t) + UI["bg_bottom"][0] * t),
            int(UI["bg_top"][1] * (1.0 - t) + UI["bg_bottom"][1] * t),
            int(UI["bg_top"][2] * (1.0 - t) + UI["bg_bottom"][2] * t),
        )

    rng = random.Random(7)
    for _ in range(120):
        x = rng.randint(40, max(41, screen_w - 40))
        y = rng.randint(int(screen_h * 0.18), max(int(screen_h * 0.18) + 1, screen_h - 36))
        color = rng.choice([(92, 170, 84), (112, 190, 96), (76, 153, 76), (135, 203, 110)])
        length = rng.randint(8, 18)
        cv2.line(bg, (x, y), (x - length // 2, y - length), color, 2, cv2.LINE_AA)
        cv2.line(bg, (x, y), (x + length // 2, y - length), color, 2, cv2.LINE_AA)

    for x, y in holes:
        draw_hole(bg, x, y, hole_radius)
    return bg
class WhackAMoleGame:
    """Orchestrates calibration, menu, gameplay, and game-over."""

    def __init__(
        self,
        camera_index: int = 0,
        filter_method: str = "none",
        calibration_method: str = "9p",
        background_path: Optional[str] = None,
        confidence_level: float = 0.5,
        model_name: str = "ridge",
        model_file: Optional[str] = None,
        config: Optional[GameConfig] = None,
    ):
        self.camera_index = camera_index
        self.filter_method = filter_method
        self.calibration_method = calibration_method
        self.background_path = background_path
        self.confidence_level = confidence_level
        self.model_name = model_name
        self.model_file = model_file
        self.config = config or GameConfig()

        # These are set up during _setup()
        self.estimator: Optional[GazeEstimator] = None
        self.smoother = None
        self.screen_w: int = 0
        self.screen_h: int = 0
        self.background: Optional[np.ndarray] = None
        self.mole: Optional[Mole] = None

        # Runtime state
        self.phase: GamePhase = GamePhase.MENU
        self.score: int = 0
        self.combo: int = 0
        self.lives: int = 0
        self.timer_start: float = 0.0
        self.timer_remaining: float = 0.0
        self.dwell_start: Optional[float] = None
        self.last_hit_time: float = 0.0
        self.floating_texts: List[FloatingText] = []
        self._gaze_smooth_x: Optional[float] = None
        self._gaze_smooth_y: Optional[float] = None
        self._last_gaze_raw_pos: Optional[Tuple[int, int]] = None
        self._max_gaze_jump_px: int = 0
        self._gaze_offset_x: float = 0.0
        self._gaze_offset_y: float = 0.0
        self._center_samples: List[Tuple[int, int]] = []
        self._center_start: Optional[float] = None
        self._locked_on_mole: bool = False
        self._lock_last_seen: float = 0.0
        self._music = None
        self._music_available: bool = False
        self._music_playing: bool = False
        self._music_path: Optional[str] = None

    # 鈹€鈹€ Public entry point 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def run(self):
        self._setup()

        with camera(self.camera_index) as cap, fullscreen("Whack-a-Mole"):
            prev_time = time.time()

            for frame in iter_frames(cap):
                now = time.time()
                dt = now - prev_time
                prev_time = now

                # 鈹€鈹€ Gaze prediction 鈹€鈹€
                features, blink = self.estimator.extract_features(frame)
                if features is not None and not blink:
                    gx, gy = map(int, self.estimator.predict(np.array([features]))[0])
                    gx, gy = self.smoother.step(gx, gy)
                    if self._gaze_smooth_x is None or self._gaze_smooth_y is None:
                        self._gaze_smooth_x = float(gx)
                        self._gaze_smooth_y = float(gy)
                    else:
                        if self._last_gaze_raw_pos is not None and self._max_gaze_jump_px > 0:
                            lx, ly = self._last_gaze_raw_pos
                            dx, dy = gx - lx, gy - ly
                            dist = float(np.hypot(dx, dy))
                            if dist > self._max_gaze_jump_px:
                                ratio = self._max_gaze_jump_px / dist
                                gx = int(lx + dx * ratio)
                                gy = int(ly + dy * ratio)
                        alpha = 0.12
                        self._gaze_smooth_x = (1.0 - alpha) * self._gaze_smooth_x + alpha * gx
                        self._gaze_smooth_y = (1.0 - alpha) * self._gaze_smooth_y + alpha * gy
                    self._last_gaze_raw_pos = (int(self._gaze_smooth_x), int(self._gaze_smooth_y))
                    gx = int(self._gaze_smooth_x + self._gaze_offset_x)
                    gy = int(self._gaze_smooth_y + self._gaze_offset_y)
                    gx = max(0, min(self.screen_w - 1, gx))
                    gy = max(0, min(self.screen_h - 1, gy))
                else:
                    gx = gy = None

                # 鈹€鈹€ Dispatch per phase 鈹€鈹€
                canvas = self.background.copy()

                if self.phase == GamePhase.MENU:
                    self._update_menu(canvas, gx, gy, now)
                elif self.phase == GamePhase.COUNTDOWN:
                    self._update_countdown(canvas, now)
                elif self.phase == GamePhase.CENTER_CALIBRATION:
                    self._update_center_calibration(canvas, gx, gy, now)
                elif self.phase == GamePhase.PLAYING:
                    self._update_playing(canvas, gx, gy, now)
                elif self.phase == GamePhase.GAME_OVER:
                    self._update_game_over(canvas, gx, gy, now)

                # 鈹€鈹€ Draw floating texts 鈹€鈹€
                self.floating_texts = [
                    ft for ft in self.floating_texts if ft.draw(canvas, now)
                ]

                # 鈹€鈹€ Gaze cursor (small green dot) 鈹€鈹€
                if gx is not None:
                    cv2.circle(canvas, (gx, gy), 18, UI["white"], 3, cv2.LINE_AA)
                    cv2.circle(canvas, (gx, gy), 9, UI["primary"], -1, cv2.LINE_AA)
                    cv2.circle(canvas, (gx, gy), 4, UI["primary_dark"], -1, cv2.LINE_AA)

                cv2.imshow("Whack-a-Mole", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key == ord("m") or key == ord("M"):
                    self._toggle_music()
                if self.phase == GamePhase.GAME_OVER:
                    if key == ord("r") or key == ord("R"):
                        self._restart()
                    elif key == ord("q") or key == ord("Q"):
                        break

        self._stop_music()
        cv2.destroyAllWindows()

    # 鈹€鈹€ Setup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _setup(self):
        self.estimator = GazeEstimator(model_name=self.model_name)

        if self.model_file and os.path.isfile(self.model_file):
            self.estimator.load_model(self.model_file)
            print(f"[whack-a-mole] Loaded model from {self.model_file}")
        else:
            calib = self.calibration_method
            if calib == "9p":
                run_9_point_calibration(self.estimator, camera_index=self.camera_index)
            elif calib == "5p":
                run_5_point_calibration(self.estimator, camera_index=self.camera_index)
            else:
                run_lissajous_calibration(self.estimator, camera_index=self.camera_index)

        self.screen_w, self.screen_h = get_screen_size()

        # Smoother
        fm = self.filter_method
        if fm == "kalman":
            kf = make_kalman()
            self.smoother = KalmanSmoother(kf)
            self.smoother.tune(self.estimator, camera_index=self.camera_index)
        elif fm == "kde":
            self.smoother = KDESmoother(self.screen_w, self.screen_h,
                                        confidence=self.confidence_level)
        else:
            self.smoother = NoSmoother()

        hole_radius = max(90, int(self.config.mole_radius * 0.95))
        self._holes = build_hole_positions(self.screen_w, self.screen_h)

        # Background
        if self.background_path and os.path.isfile(self.background_path):
            bg = cv2.imread(self.background_path)
            self.background = cv2.resize(bg, (self.screen_w, self.screen_h))
            for hx, hy in self._holes:
                draw_hole(self.background, hx, hy, hole_radius)
        else:
            self.background = build_grass_background(
                self.screen_w,
                self.screen_h,
                self._holes,
                hole_radius,
            )

        # Mole
        self.mole = Mole(
            self.screen_w, self.screen_h,
            radius=self.config.mole_radius,
            margin=self.config.mole_margin,
            image_path=self.config.mole_image_path,
        )
        self.mole.set_holes(self._holes)

        # Init state
        self.phase = GamePhase.MENU
        self.score = 0
        self.combo = 0
        self._init_menu_dwell()
        self._gaze_smooth_x = None
        self._gaze_smooth_y = None
        self._last_gaze_raw_pos = None
        self._max_gaze_jump_px = int(min(self.screen_w, self.screen_h) * 0.15)
        self._init_music()

    def _resolve_music_path(self) -> Optional[str]:
        path = self.config.music_path
        if not path:
            return None
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in SUPPORTED_MUSIC_EXTS:
            return path

        base, ext = os.path.splitext(path)
        if ext.lower() == ".ncm":
            for supported_ext in SUPPORTED_MUSIC_EXTS:
                candidate = base + supported_ext
                if os.path.isfile(candidate):
                    return candidate
            print(
                "[whack-a-mole] Music file is .ncm, which cannot be played directly. "
                "Please convert it to mp3/wav/ogg/flac with the same name."
            )
            return None

        if os.path.isfile(path):
            print(f"[whack-a-mole] Unsupported music format: {path}")
        else:
            print(f"[whack-a-mole] Music file not found: {path}")
        return None

    def _init_music(self):
        self._music_path = self._resolve_music_path()
        if self._music_path is None:
            return
        try:
            import pygame

            pygame.mixer.init()
            pygame.mixer.music.load(self._music_path)
            pygame.mixer.music.set_volume(0.45)
            self._music = pygame.mixer.music
            self._music_available = True
            print(f"[whack-a-mole] Music ready: {self._music_path}")
        except Exception as exc:
            self._music_available = False
            self._music = None
            print(f"[whack-a-mole] Music unavailable: {exc}")

    def _play_music(self):
        if not self._music_available or self._music is None:
            return
        try:
            if not self._music_playing:
                self._music.play(-1)
                self._music_playing = True
                print("[whack-a-mole] Music on.")
        except Exception as exc:
            print(f"[whack-a-mole] Could not play music: {exc}")

    def _stop_music(self):
        if self._music is None:
            return
        try:
            self._music.stop()
        except Exception:
            pass
        self._music_playing = False

    def _toggle_music(self):
        if not self._music_available or self._music is None:
            print("[whack-a-mole] Music is not available. Use mp3/wav/ogg/flac, not .ncm.")
            return
        if self._music_playing:
            self._music.pause()
            self._music_playing = False
            print("[whack-a-mole] Music off.")
        else:
            self._music.unpause()
            if not self._music.get_busy():
                self._music.play(-1)
            self._music_playing = True
            print("[whack-a-mole] Music on.")

    # 鈹€鈹€ Menu 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _init_menu_dwell(self):
        self._menu_btn_labels = ["Timed"]   # only Timed for simplicity; lives can be added
        self._menu_last_btn: Optional[str] = None
        self._menu_dwell_start: Optional[float] = None
        self._menu_btn_rects: List[Tuple[str, int, int, int, int]] = []
        self._build_menu_buttons()

    def _build_menu_buttons(self):
        """Construct two centered buttons."""
        btn_w = min(520, max(360, int(self.screen_w * 0.26)))
        btn_h = min(104, max(76, int(self.screen_h * 0.095)))
        gap = max(24, int(self.screen_h * 0.035))
        cx = self.screen_w // 2
        top = int(self.screen_h * 0.44)

        labels = ["Timed (60s)", "Lives (5)"]
        self._menu_btn_labels = labels
        self._menu_btn_rects.clear()
        for i, label in enumerate(labels):
            x1 = cx - btn_w // 2
            y1 = top + i * (btn_h + gap)
            self._menu_btn_rects.append((label, x1, y1, x1 + btn_w, y1 + btn_h))

    def _find_menu_button(self, gx: float, gy: float) -> Optional[str]:
        for label, x1, y1, x2, y2 in self._menu_btn_rects:
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                return label
        return None

    def _draw_menu(self, canvas: np.ndarray,
                   highlighted_label: Optional[str],
                   dwell_progress: float):
        # Author watermark: Ma Yuexiang appears subtly on the start screen.
        panel_w = min(int(self.screen_w * 0.78), 980)
        panel_h = min(int(self.screen_h * 0.68), 700)
        x1 = (self.screen_w - panel_w) // 2
        y1 = max(48, (self.screen_h - panel_h) // 2)
        x2 = x1 + panel_w
        y2 = y1 + panel_h
        draw_card(canvas, x1, y1, x2, y2, UI["panel"], 28)

        tag_w, tag_h = 210, 42
        draw_rounded_rect(canvas, x1 + 54, y1 + 46, x1 + 54 + tag_w, y1 + 46 + tag_h, UI["panel_2"], 18)
        cv2.putText(canvas, "GAZE GAME", (x1 + 82, y1 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, UI["primary_dark"], 2, cv2.LINE_AA)

        put_centered_text(canvas, "WHACK-A-MOLE", self.screen_w // 2, y1 + 160, 2.45, UI["text"], 4)
        put_centered_text(canvas, "Look at a mode button to start", self.screen_w // 2, y1 + 220, 1.0, UI["muted"], 2)
        put_centered_text(canvas, "Author: Ma Yuexiang", self.screen_w // 2, y2 - 44, 0.72, UI["muted"], 2)

        for label, x1, y1, x2, y2 in self._menu_btn_rects:
            is_hl = highlighted_label is not None and highlighted_label == label
            fill = UI["primary"] if is_hl else UI["white"]
            text_color = UI["white"] if is_hl else UI["text"]
            border_color = UI["primary_dark"] if is_hl else (214, 226, 207)
            draw_rounded_rect(canvas, x1, y1, x2, y2, fill, 22)
            draw_rounded_rect(canvas, x1, y1, x2, y2, border_color, 22, 3 if is_hl else 2)

            (bw, bh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.28, 2)
            bx = x1 + (x2 - x1 - bw) // 2
            by = y1 + (y2 - y1 + bh) // 2
            cv2.putText(canvas, label, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.28,
                        text_color, 2, cv2.LINE_AA)

        # Dwell ring
        if highlighted_label is not None and dwell_progress > 0:
            for label, x1, y1, x2, y2 in self._menu_btn_rects:
                if label == highlighted_label:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    radius = int(min(x2 - x1, y2 - y1) * 0.3)
                    cv2.circle(canvas, (cx, cy), radius, UI["accent_soft"], 3)
                    angle = int(360 * min(1.0, dwell_progress))
                    cv2.ellipse(canvas, (cx, cy), (radius, radius),
                                -90, 0, angle, UI["amber"], 5)
                    break

    def _update_menu(self, canvas: np.ndarray,
                     gx: Optional[int], gy: Optional[int], now: float):
        dwell_progress = 0.0
        highlighted: Optional[str] = None

        if gx is not None and gy is not None:
            highlighted = self._find_menu_button(gx, gy)

            if highlighted is not None:
                if self._menu_last_btn == highlighted:
                    if self._menu_dwell_start is None:
                        self._menu_dwell_start = now
                    elapsed = now - self._menu_dwell_start
                    dwell_progress = elapsed / self.config.dwell_time
                else:
                    self._menu_last_btn = highlighted
                    self._menu_dwell_start = now
            else:
                self._menu_last_btn = None
                self._menu_dwell_start = None
        else:
            self._menu_last_btn = None
            self._menu_dwell_start = None

        # Activate selection
        if (highlighted is not None and self._menu_dwell_start is not None
                and (now - self._menu_dwell_start) >= self.config.dwell_time):
            if "Lives" in highlighted:
                self.config.mode = "lives"
                self.lives = self.config.total_lives
            else:
                self.config.mode = "timed"
                self.timer_remaining = self.config.total_time

            self._start_countdown(now)

        self._draw_menu(canvas, highlighted, dwell_progress)

    # 鈹€鈹€ Countdown 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _start_countdown(self, now: float):
        self.phase = GamePhase.COUNTDOWN
        self._cd_start = now
        self._cd_duration = 3.0  # 3-2-1

    def _update_countdown(self, canvas: np.ndarray, now: float):
        elapsed = now - self._cd_start
        remaining = self._cd_duration - elapsed

        if remaining <= 0:
            self._start_center_calibration(now)
            return

        number = int(remaining) + 1  # 3鈫?鈫?
        text = str(number)
        frac = remaining - int(remaining)  # 0鈫? within each number
        pulse = 1.0 + 0.2 * np.sin(frac * np.pi)
        scale = 7.5 * pulse

        overlay = canvas.copy()
        draw_card(overlay, self.screen_w // 2 - 210, self.screen_h // 2 - 210,
                  self.screen_w // 2 + 210, self.screen_h // 2 + 210, UI["panel"], 42)
        (tw2, th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, int(12 * pulse))
        cx2 = (self.screen_w - tw2) // 2
        cy2 = (self.screen_h + th2) // 2
        cv2.putText(overlay, text, (cx2, cy2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    UI["primary_dark"], int(12 * pulse), cv2.LINE_AA)
        alpha = 0.92
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

    def _start_center_calibration(self, now: float):
        self.phase = GamePhase.CENTER_CALIBRATION
        self._center_start = now
        self._center_samples = []
        self._gaze_offset_x = 0.0
        self._gaze_offset_y = 0.0

    def _update_center_calibration(self, canvas: np.ndarray,
                                   gx: Optional[int], gy: Optional[int], now: float):
        center = (self.screen_w // 2, self.screen_h // 2)
        elapsed = now - (self._center_start or now)
        duration = 1.5

        draw_card(canvas, center[0] - 360, center[1] - 220, center[0] + 360, center[1] + 250, UI["panel"], 30)
        cv2.circle(canvas, center, 32, UI["white"], -1)
        cv2.circle(canvas, center, 54, UI["accent"], 4)
        cv2.line(canvas, (center[0] - 82, center[1]), (center[0] + 82, center[1]), UI["primary_dark"], 3)
        cv2.line(canvas, (center[0], center[1] - 82), (center[0], center[1] + 82), UI["primary_dark"], 3)

        if gx is not None and gy is not None and elapsed > 0.35:
            self._center_samples.append((gx, gy))

        progress = min(1.0, elapsed / duration)
        cv2.ellipse(canvas, center, (76, 76), -90, 0, int(360 * progress), UI["primary"], 7)
        text = "Look at center"
        put_centered_text(canvas, text, center[0], center[1] + 150, 1.35, UI["text"], 3)
        put_centered_text(canvas, "Keep your head still for a moment", center[0], center[1] + 198, 0.82, UI["muted"], 2)

        if elapsed >= duration:
            if self._center_samples:
                sample = np.median(np.array(self._center_samples), axis=0)
                self._gaze_offset_x = float(center[0] - sample[0])
                self._gaze_offset_y = float(center[1] - sample[1])
            self._start_game(now)

    # 鈹€鈹€ Game Start 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _start_game(self, now: float):
        self.phase = GamePhase.PLAYING
        self.score = 0
        self.combo = 0
        self.dwell_start = None
        self.last_hit_time = 0.0
        self._locked_on_mole = False
        self._lock_last_seen = 0.0
        self.timer_start = now
        self.floating_texts.clear()
        self.mole.spawn()
        self._play_music()
        print(f"[whack-a-mole] Game started! Mode: {self.config.mode}")

    # 鈹€鈹€ Main Game Loop 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _update_playing(self, canvas: np.ndarray,
                        gx: Optional[int], gy: Optional[int], now: float):
        # Timer
        if self.config.mode == "timed":
            self.timer_remaining = self.config.total_time - (now - self.timer_start)
            if self.timer_remaining <= 0:
                self.timer_remaining = 0
                self._end_game()
                return

        # Lock onto a nearby mole and tolerate short gaze dropouts.
        # Author watermark: Ma Yuexiang tuned this lock behavior for gaze input.
        dwell_progress = 0.0
        on_mole = self._locked_on_mole
        if gx is not None and gy is not None and self.mole.active:
            dist = float(np.hypot(gx - self.mole.x, gy - self.mole.y))
            capture_radius = self.mole.radius * 1.75
            hold_radius = self.mole.radius * 2.2
            if dist <= capture_radius or (self._locked_on_mole and dist <= hold_radius):
                self._locked_on_mole = True
                self._lock_last_seen = now
                on_mole = True
            elif self._locked_on_mole and (now - self._lock_last_seen) <= 0.35:
                on_mole = True
            else:
                self._locked_on_mole = False
                on_mole = False
        elif self._locked_on_mole and (now - self._lock_last_seen) <= 0.35:
            on_mole = True
        else:
            self._locked_on_mole = False

        if on_mole:
            if self.dwell_start is None:
                self.dwell_start = now
            dwell_progress = (now - self.dwell_start) / self.config.dwell_time
        else:
            self.dwell_start = None

        # Hit!
        if (on_mole and self.dwell_start is not None
                and (now - self.dwell_start) >= self.config.dwell_time):
            if now - self.last_hit_time >= self.config.hit_cooldown:
                self._on_hit(now)

        # Mole expired (lives mode penalty)
        if self.mole.active:
            mole_age = now - self.mole.spawn_time
            if mole_age >= self.config.mole_ttl:
                self._on_miss(now)

        # 鈹€鈹€ Render 鈹€鈹€
        # Draw mole with hit flash in the 0.2s after hit
        hit_flash = (now - self.last_hit_time) < 0.2
        self.mole.draw(canvas, now, hit_flash=hit_flash)

        # Dwell progress ring on mole
        if on_mole and self.mole.active and dwell_progress > 0:
            mx, my = self.mole.x, self.mole.y
            ring_r = self.mole.radius + 10
            cv2.circle(canvas, (mx, my), int(self.mole.radius * 1.75), UI["accent_soft"], 3)
            cv2.circle(canvas, (mx, my), ring_r, UI["white"], 4)
            angle = int(360 * min(1.0, dwell_progress))
            cv2.ellipse(canvas, (mx, my), (ring_r, ring_r),
                        -90, 0, angle, UI["amber"], 8)

        # HUD
        self._draw_hud(canvas, now)

    # 鈹€鈹€ Hit / Miss handlers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _on_hit(self, now: float):
        self.last_hit_time = now
        self.combo += 1
        bonus = 1.0 + self.combo * self.config.combo_bonus_factor
        points = int(self.config.base_score * bonus)
        self.score += points

        # Floating text
        combo_str = f" x{self.combo}" if self.combo > 1 else ""
        ft = FloatingText(
            text=f"+{points}{combo_str}",
            x=self.mole.x - 40,
            y=self.mole.y - self.mole.radius,
            color=(0, 255, 255),
            start_time=now,
        )
        self.floating_texts.append(ft)

        if self.combo > 1 and self.combo % 5 == 0:
            ft2 = FloatingText(
                text=f"COMBO {self.combo}!",
                x=self.screen_w // 2 - 80,
                y=self.screen_h // 3,
                color=(255, 215, 0),
                start_time=now,
                duration=1.2,
            )
            self.floating_texts.append(ft2)

        print(f"[whack-a-mole] HIT! Score: {self.score}  Combo: {self.combo}")

        # Respawn mole
        self.mole.despawn()
        self.dwell_start = None
        self._locked_on_mole = False
        self._lock_last_seen = 0.0
        self._gaze_smooth_x = None
        self._gaze_smooth_y = None
        self._last_gaze_raw_pos = None
        self.mole.spawn()

    def _on_miss(self, now: float):
        self.combo = 0
        self.dwell_start = None
        self._locked_on_mole = False
        self._lock_last_seen = 0.0
        self._gaze_smooth_x = None
        self._gaze_smooth_y = None
        self._last_gaze_raw_pos = None

        if self.config.mode == "lives":
            self.lives -= 1
            print(f"[whack-a-mole] MISS! Lives: {self.lives}")
            if self.lives <= 0:
                self._end_game()
                return

        # Respawn mole
        self.mole.despawn()
        self.mole.spawn()

    # 鈹€鈹€ HUD 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _draw_hud(self, canvas: np.ndarray, now: float):
        # Author watermark: Ma Yuexiang is shown quietly in the live HUD.
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = max(18, int(self.screen_w * 0.018))
        card_h = min(96, max(74, int(self.screen_h * 0.09)))
        y1 = margin
        y2 = y1 + card_h
        card_w = min(360, max(250, int(self.screen_w * 0.20)))

        draw_card(canvas, margin, y1, margin + card_w, y2, UI["panel"], 20)
        cv2.putText(canvas, "SCORE", (margin + 26, y1 + 30),
                    font, 0.62, UI["muted"], 2, cv2.LINE_AA)
        cv2.putText(canvas, str(self.score), (margin + 26, y1 + 68),
                    font, 1.32, UI["text"], 3, cv2.LINE_AA)

        center_w = min(330, max(230, int(self.screen_w * 0.18)))
        center_x1 = (self.screen_w - center_w) // 2
        draw_card(canvas, center_x1, y1, center_x1 + center_w, y2, UI["panel"], 20)
        mode_txt = "TIMED MODE" if self.config.mode == "timed" else "LIVES MODE"
        put_centered_text(canvas, mode_txt, self.screen_w // 2, y1 + 34, 0.72, UI["muted"], 2)
        combo_txt = f"COMBO x{self.combo}" if self.combo > 1 else "FOCUS TO WHACK"
        put_centered_text(canvas, combo_txt, self.screen_w // 2, y1 + 70, 0.82, UI["primary_dark"], 2)
        put_centered_text(canvas, "Author: Ma Yuexiang", self.screen_w // 2, y2 + 28, 0.56, UI["muted"], 1)
        music_label = "Music ON  M" if self._music_playing else "Music OFF  M"
        put_centered_text(canvas, music_label, self.screen_w // 2, y2 + 52, 0.45, UI["muted"], 1)

        right_x1 = self.screen_w - margin - card_w
        draw_card(canvas, right_x1, y1, right_x1 + card_w, y2, UI["panel"], 20)
        if self.config.mode == "timed":
            label = "TIME"
            value = f"{int(self.timer_remaining)}s"
            color = UI["primary_dark"] if self.timer_remaining > 10 else UI["danger"]
        else:
            label = "LIVES"
            value = str(self.lives)
            color = UI["danger"]
        cv2.putText(canvas, label, (right_x1 + 26, y1 + 30),
                    font, 0.62, UI["muted"], 2, cv2.LINE_AA)
        (vw, _), _ = cv2.getTextSize(value, font, 1.32, 3)
        cv2.putText(canvas, value, (right_x1 + card_w - vw - 28, y1 + 68),
                    font, 1.32, color, 3, cv2.LINE_AA)

    # 鈹€鈹€ Game Over 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _end_game(self):
        self.phase = GamePhase.GAME_OVER
        self._go_start = time.time()
        self.mole.despawn()
        print(f"[whack-a-mole] GAME OVER! Final Score: {self.score}")

    def _update_game_over(self, canvas: np.ndarray,
                          gx: Optional[int], gy: Optional[int], now: float):
        overlay = canvas.copy()
        overlay[:] = (211, 226, 204)
        alpha = 0.58
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

        font = cv2.FONT_HERSHEY_SIMPLEX
        panel_w = min(int(self.screen_w * 0.74), 860)
        panel_h = min(int(self.screen_h * 0.64), 610)
        x1 = (self.screen_w - panel_w) // 2
        y1 = (self.screen_h - panel_h) // 2
        x2 = x1 + panel_w
        y2 = y1 + panel_h
        draw_card(canvas, x1, y1, x2, y2, UI["panel"], 30)

        # Title
        title = "GAME OVER"
        put_centered_text(canvas, title, self.screen_w // 2, y1 + 120, 2.4, UI["text"], 4)

        # Final score
        score_txt = f"Final Score  {self.score}"
        put_centered_text(canvas, score_txt, self.screen_w // 2, y1 + 220, 1.55, UI["primary_dark"], 3)

        # Max combo
        combo_txt = f"Current Combo  {self.combo}"
        put_centered_text(canvas, combo_txt, self.screen_w // 2, y1 + 285, 1.05, UI["muted"], 2)

        # Instructions
        btn_w = min(300, max(220, int(self.screen_w * 0.16)))
        btn_h = 72
        gap = 30
        bx1 = self.screen_w // 2 - btn_w - gap // 2
        bx2 = self.screen_w // 2 + gap // 2
        by1 = min(y2 - 145, y1 + 365)
        draw_rounded_rect(canvas, bx1, by1, bx1 + btn_w, by1 + btn_h, UI["primary"], 20)
        draw_rounded_rect(canvas, bx2, by1, bx2 + btn_w, by1 + btn_h, UI["white"], 20)
        draw_rounded_rect(canvas, bx2, by1, bx2 + btn_w, by1 + btn_h, (214, 226, 207), 20, 2)
        put_centered_text(canvas, "R  Retry", bx1 + btn_w // 2, by1 + 47, 0.92, UI["white"], 2)
        put_centered_text(canvas, "Q  Quit", bx2 + btn_w // 2, by1 + 47, 0.92, UI["text"], 2)

    # 鈹€鈹€ Restart 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _restart(self):
        self.phase = GamePhase.MENU
        self.score = 0
        self.combo = 0
        self.dwell_start = None
        self.last_hit_time = 0.0
        self.floating_texts.clear()
        self._init_menu_dwell()
        self._gaze_smooth_x = None
        self._gaze_smooth_y = None
        self._last_gaze_raw_pos = None
        self._gaze_offset_x = 0.0
        self._gaze_offset_y = 0.0
        self._locked_on_mole = False
        self._lock_last_seen = 0.0
        print("[whack-a-mole] Restarted.")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  CLI & Entry Point
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whack-a-Mole 鈥?eye-tracking game"
    )
    parser.add_argument(
        "--filter", choices=["kalman", "kde", "none"], default="none",
        help="Smoothing filter (default: none)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--calibration", choices=["9p", "5p", "lissajous"], default="9p",
        help="Calibration method (default: 9p)",
    )
    parser.add_argument(
        "--background", type=str, default=None,
        help="Optional background image path",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="KDE confidence level (default: 0.5)",
    )
    parser.add_argument(
        "--model", default="ridge",
        help="ML model for gaze estimation (default: ridge)",
    )
    parser.add_argument(
        "--model-file", type=str, default=None,
        help="Path to pre-trained gaze model (.pkl)",
    )
    parser.add_argument(
        "--mole-image", type=str, default=DEFAULT_MOLE_IMAGE_PATH,
        help="Path to mole PNG sprite (default: project mole asset)",
    )
    parser.add_argument(
        "--dwell", type=float, default=0.35,
        help="Dwell time in seconds to whack a mole (default: 0.35)",
    )
    parser.add_argument(
        "--timed", type=int, default=None,
        help="Force timed mode with N seconds (e.g. --timed 60)",
    )
    parser.add_argument(
        "--lives", type=int, default=None,
        help="Force lives mode with N lives (e.g. --lives 5)",
    )
    parser.add_argument(
        "--mole-ttl", type=float, default=6.0,
        help="Seconds before an unhit mole disappears (default: 6.0)",
    )
    parser.add_argument(
        "--mole-radius", type=int, default=120,
        help="Mole hit-test radius in pixels (default: 120)",
    )
    parser.add_argument(
        "--music", type=str, default=DEFAULT_MUSIC_PATH,
        help="Background music path. Use mp3/wav/ogg/flac; .ncm cannot be played directly.",
    )
    return parser.parse_args()


def run_whack_a_mole():
    args = _parse_args()

    # Build config
    config = GameConfig(
        dwell_time=args.dwell,
        mole_ttl=args.mole_ttl,
        mole_radius=args.mole_radius,
        mole_image_path=args.mole_image,
        music_path=args.music,
    )
    if args.timed is not None:
        config.mode = "timed"
        config.total_time = float(args.timed)
    elif args.lives is not None:
        config.mode = "lives"
        config.total_lives = args.lives

    game = WhackAMoleGame(
        camera_index=args.camera,
        filter_method=args.filter,
        calibration_method=args.calibration,
        background_path=args.background,
        confidence_level=args.confidence,
        model_name=args.model,
        model_file=args.model_file,
        config=config,
    )
    game.run()


if __name__ == "__main__":
    run_whack_a_mole()



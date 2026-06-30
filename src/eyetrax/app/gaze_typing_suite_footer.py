import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from eyetrax.app.gaze_typing import OnScreenKeyboard, apply_key
from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, fullscreen, iter_frames


@dataclass
class ModeButton:
    label: str
    mode: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class TopButton:
    label: str
    action: str  # "exit" or "back"
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def draw_mode_select(
    canvas: np.ndarray,
    buttons: list[ModeButton],
    highlighted: Optional[ModeButton],
    dwell_progress: float,
):
    """
    Render mode selection UI on the shared window.
    """
    h, w = canvas.shape[:2]

    # Top bar
    top_bar_h = int(h * 0.2)
    cv2.rectangle(canvas, (0, 0), (w, top_bar_h), (0, 0, 0), thickness=-1)
    title = "Select Mode"
    (title_w, title_h), _ = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 2, 2
    )
    cv2.putText(
        canvas,
        title,
        ((w - title_w) // 2, int(top_bar_h * 0.6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Buttons
    for btn in buttons:
        color = (200, 200, 200)
        thickness = 3
        if highlighted is not None and highlighted.mode == btn.mode:
            color = (0, 255, 255)
            thickness = 4
        cv2.rectangle(
            canvas,
            (btn.x1, btn.y1),
            (btn.x2, btn.y2),
            color,
            thickness=thickness,
        )
        (tw, th), _ = cv2.getTextSize(
            btn.label, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 2
        )
        tx = btn.x1 + (btn.x2 - btn.x1 - tw) // 2
        ty = btn.y1 + (btn.y2 - btn.y1 + th) // 2
        cv2.putText(
            canvas,
            btn.label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Dwell ring
    if highlighted is not None and dwell_progress > 0:
        cx = (highlighted.x1 + highlighted.x2) // 2
        cy = (highlighted.y1 + highlighted.y2) // 2
        radius = int(min(highlighted.x2 - highlighted.x1, highlighted.y2 - highlighted.y1) * 0.25)
        cv2.circle(canvas, (cx, cy), radius, (90, 90, 90), 2)
        angle = int(360 * max(0.0, min(1.0, dwell_progress)))
        cv2.ellipse(
            canvas,
            (cx, cy),
            (radius, radius),
            -90,
            0,
            angle,
            (0, 255, 255),
            4,
        )


def build_mode_buttons(screen_w: int, screen_h: int) -> list[ModeButton]:
    """
    Create two centered buttons for mode selection.
    """
    btn_w = screen_w // 3
    btn_h = screen_h // 6
    gap = int(screen_h * 0.05)
    center_x = screen_w // 2
    top = int(screen_h * 0.35)

    buttons = []
    # Typing test (original)
    x1 = center_x - btn_w - gap // 2
    y1 = top
    buttons.append(
        ModeButton("Typing Test", "test", x1, y1, x1 + btn_w, y1 + btn_h)
    )
    # Free typing
    x1 = center_x + gap // 2
    y1 = top
    buttons.append(
        ModeButton("Free Type", "free", x1, y1, x1 + btn_w, y1 + btn_h)
    )
    return buttons


def build_top_buttons(screen_w: int, screen_h: int) -> list[TopButton]:
    """
    Buttons at top-left (Exit) and top-right (Back) for typing modes.
    """
    margin = 20
    # Make buttons as wide as the top bar region (same width as text box area).
    btn_w = max(220, screen_w // 6)
    btn_h = max(120, screen_h // 6)
    y1 = margin
    buttons = [
        TopButton("EXIT", "exit", margin, y1, margin + btn_w, y1 + btn_h),
        TopButton(
            "BACK",
            "back",
            screen_w - margin - btn_w,
            y1,
            screen_w - margin,
            y1 + btn_h,
        ),
    ]
    return buttons


def draw_top_buttons(
    canvas: np.ndarray,
    buttons: list[TopButton],
    highlighted: Optional[TopButton],
    dwell_progress: float,
):
    """
    Render the exit/back buttons with optional dwell ring.
    """
    for btn in buttons:
        base_color = (30, 30, 30)
        border_color = (200, 200, 200)
        fill_color = base_color
        thickness = 3 if highlighted is not None and btn.action == highlighted.action else 2
        cv2.rectangle(canvas, (btn.x1, btn.y1), (btn.x2, btn.y2), fill_color, thickness=-1)
        cv2.rectangle(canvas, (btn.x1, btn.y1), (btn.x2, btn.y2), border_color, thickness=thickness)

        (tw, th), _ = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)
        tx = btn.x1 + (btn.x2 - btn.x1 - tw) // 2
        ty = btn.y1 + (btn.y2 - btn.y1 + th) // 2
        cv2.putText(
            canvas,
            btn.label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (240, 240, 240),
            3,
            cv2.LINE_AA,
        )

    if highlighted is not None and dwell_progress > 0:
        cx = (highlighted.x1 + highlighted.x2) // 2
        cy = (highlighted.y1 + highlighted.y2) // 2
        radius = int(min(highlighted.x2 - highlighted.x1, highlighted.y2 - highlighted.y1) * 0.35)
        cv2.circle(canvas, (cx, cy), radius, (90, 90, 90), 2)
        angle = int(360 * max(0.0, min(1.0, dwell_progress)))
        cv2.ellipse(
            canvas,
            (cx, cy),
            (radius, radius),
            -90,
            0,
            angle,
            (0, 255, 255),
            4,
        )


FOOTER_TEXT = "厦门大学郭伟杰课题组研制"
FOOTER_FONT_PATH = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc")
FOOTER_FONT_SIZE = 50
FOOTER_COLOR_BGR = (200, 200, 200)
FOOTER_BG_BGR = None
FOOTER_PADDING = 20
FOOTER_BOTTOM_MARGIN = 40


def draw_footer(
    canvas: np.ndarray,
    screen_w: int,
    screen_h: int,
    text: str = FOOTER_TEXT,
    font_path: str = FOOTER_FONT_PATH,
    font_size: int = FOOTER_FONT_SIZE,
    color_bgr: tuple[int, int, int] = FOOTER_COLOR_BGR,
    bg_bgr: tuple[int, int, int] = FOOTER_BG_BGR,
    padding: int = FOOTER_PADDING,
    bottom_margin: int = FOOTER_BOTTOM_MARGIN,
):
    font_scale = 1.4
    thickness = 3

    # OpenCV's built-in font cannot render CJK; try Pillow + a CJK font first.
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_path, size=font_size)
        dummy = Image.new("RGB", (1, 1), (0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (screen_w - tw) // 2
        y = screen_h - bottom_margin - th
        #cv2.rectangle(
            #canvas,
            #(x - padding, y - padding),
            #(x + tw + padding, y + th + padding),
            #bg_bgr,
            #thickness=-1,
        #)
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
        canvas[:] = np.array(img)
        return
    except Exception:
        pass

    # Fallback: ASCII-only OpenCV text (may show as '?').
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (screen_w - tw) // 2
    y = screen_h - bottom_margin
    cv2.rectangle(
        canvas,
        (x - padding, y - th - padding),
        (x + tw + padding, y + padding),
        bg_bgr,
        thickness=-1,
    )
    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        font_scale,
        color_bgr,
        thickness,
        cv2.LINE_AA,
    )


def run_gaze_typing_suite():
    """
    Single-window gaze typing suite with mode selection and dwell-only activation.
    Modes:
      - Typing Test: same as existing test flow with target text.
      - Free Type: no target, free-form typing.
    """
    args = parse_common_args()

    #filter_method = args.filter
    filter_method = "kde"
    camera_index = args.camera
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator(model_name=args.model)

    # One-time calibration / load
    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[typing-suite] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    screen_width, screen_height = get_screen_size()

    # Smoother
    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    # Background
    if background_path and os.path.isfile(background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (screen_width, screen_height))
    else:
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        background[:] = (30, 30, 30)

    keyboard = OnScreenKeyboard(screen_width, screen_height)
    mode_buttons = build_mode_buttons(screen_width, screen_height)
    top_buttons = build_top_buttons(screen_width, screen_height)

    target_text = "HELLO"
    typed_text = ""

    # Dwell timings
    DWELL_TIME_MODE = 1.0
    DWELL_TIME_KEY = 1.0
    DWELL_TIME_BUTTON = 1.0

    last_mode_label: Optional[str] = None
    mode_dwell_start: Optional[float] = None

    last_key_label: Optional[str] = None
    key_dwell_start: Optional[float] = None

    last_top_label: Optional[str] = None
    top_dwell_start: Optional[float] = None

    current_mode = "select"  # select -> test | free

    window_name = "Gaze Typing Suite"

    with camera(camera_index) as cap, fullscreen(window_name):
        prev_time = time.time()

        for frame in iter_frames(cap):
            now = time.time()
            dt = now - prev_time
            prev_time = now

            features, blink_detected = gaze_estimator.extract_features(frame)

            if features is not None and not blink_detected:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)
                x_pred, y_pred = smoother.step(x, y)
            else:
                x_pred = y_pred = None

            canvas = background.copy()
            gaze_draw_pos = None

            # Clamp gaze point within screen
            if x_pred is not None and y_pred is not None:
                x_pred = max(0, min(screen_width - 1, x_pred))
                y_pred = max(0, min(screen_height - 1, y_pred))
                gaze_draw_pos = (x_pred, y_pred)

            dwell_progress = 0.0
            highlighted_key = None

            if current_mode == "select":
                highlighted_btn = None
                if x_pred is not None and y_pred is not None:
                    for btn in mode_buttons:
                        if btn.contains(x_pred, y_pred):
                            highlighted_btn = btn
                            break

                    if highlighted_btn is not None:
                        if last_mode_label == highlighted_btn.mode:
                            if mode_dwell_start is None:
                                mode_dwell_start = now
                            dwell_progress = (now - mode_dwell_start) / DWELL_TIME_MODE
                        else:
                            last_mode_label = highlighted_btn.mode
                            mode_dwell_start = now
                            dwell_progress = 0.0
                    else:
                        last_mode_label = None
                        mode_dwell_start = None
                        dwell_progress = 0.0
                else:
                    last_mode_label = None
                    mode_dwell_start = None

                if (
                    highlighted_btn is not None
                    and mode_dwell_start is not None
                    and (now - mode_dwell_start) >= DWELL_TIME_MODE
                ):
                    current_mode = highlighted_btn.mode
                    typed_text = ""
                    key_dwell_start = None
                    last_key_label = None
                    # small reset to avoid immediate re-trigger
                    mode_dwell_start = now

                draw_mode_select(canvas, mode_buttons, highlighted_btn, dwell_progress)

            else:
                # Typing modes share the same keyboard UI plus top buttons.
                top_highlight = None
                top_progress = 0.0
                skip_keys = False

                if x_pred is not None and y_pred is not None:

                    # Top buttons (Exit/Back)
                    for btn in top_buttons:
                        if btn.contains(x_pred, y_pred):
                            top_highlight = btn
                            break
                    if top_highlight is not None:
                        if last_top_label == top_highlight.action:
                            if top_dwell_start is None:
                                top_dwell_start = now
                            top_progress = (now - top_dwell_start) / DWELL_TIME_BUTTON
                        else:
                            last_top_label = top_highlight.action
                            top_dwell_start = now
                            top_progress = 0.0
                        skip_keys = True
                    else:
                        last_top_label = None
                        top_dwell_start = None

                    if not skip_keys:
                        highlighted_key = keyboard.find_key_at(x_pred, y_pred)

                        if highlighted_key is not None:
                            if last_key_label == highlighted_key.label:
                                if key_dwell_start is None:
                                    key_dwell_start = now
                                dwell_progress = (now - key_dwell_start) / DWELL_TIME_KEY
                            else:
                                last_key_label = highlighted_key.label
                                key_dwell_start = now
                                dwell_progress = 0.0
                        else:
                            last_key_label = None
                            key_dwell_start = None
                            dwell_progress = 0.0
                    else:
                        highlighted_key = None
                        dwell_progress = 0.0
                        last_key_label = None
                        key_dwell_start = None
                else:
                    last_key_label = None
                    key_dwell_start = None
                    last_top_label = None
                    top_dwell_start = None

                # Handle top button selection
                if (
                    top_highlight is not None
                    and top_dwell_start is not None
                    and (now - top_dwell_start) >= DWELL_TIME_BUTTON
                ):
                    if top_highlight.action == "exit":
                        break
                    if top_highlight.action == "back":
                        current_mode = "select"
                        typed_text = ""
                        highlighted_key = None
                        dwell_progress = 0.0
                        last_key_label = None
                        key_dwell_start = None
                        last_mode_label = None
                        mode_dwell_start = None
                        last_top_label = None
                        top_dwell_start = None
                        continue

                # Handle key selection
                if (
                    highlighted_key is not None
                    and key_dwell_start is not None
                    and (now - key_dwell_start) >= DWELL_TIME_KEY
                ):
                    selected_label = highlighted_key.label
                    typed_text = apply_key(selected_label, typed_text)
                    key_dwell_start = now  # reset dwell timer
                    print(f"[typing-suite] Selected key: {selected_label}, text = {typed_text!r}")

                # Render keyboard + texts
                kb_target = target_text if current_mode == "test" else ""
                keyboard.draw(
                    canvas,
                    highlighted_key,
                    kb_target,
                    typed_text,
                    dwell_progress,
                    bar_color=(30, 30, 30),  # match keyboard background
                    target_color=(30, 30, 30),
                    typed_color=(60, 60, 60),
                    target_scale=2.4,
                    typed_scale=2.4,
                    target_thickness=4,
                    typed_thickness=4,
                    text_box_border_color=(200, 200, 200),
                    text_box_border_thickness=3,
                    text_box_margin=10,
                    text_box_width_ratio=0.6,
                    text_box_height_ratio=0.8,
                    text_box_fill_color=(255, 255, 255),
                )

                # Draw top buttons after the text bar to keep them visible
                draw_top_buttons(canvas, top_buttons, top_highlight, top_progress)

                # HUD: blink, mode, FPS
                blink_txt = "Blink: YES" if blink_detected else "Blink: NO"
                blink_clr = (0, 0, 255) if blink_detected else (0, 255, 0)
                cv2.putText(
                    canvas,
                    blink_txt,
                    (40, int(screen_height * 0.23)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"Mode: {current_mode.upper()}",
                    (40, int(screen_height * 0.29)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                fps = 1.0 / max(dt, 1e-6)
                cv2.putText(
                    canvas,
                    f"FPS: {int(fps)}",
                    (40, int(screen_height * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            draw_footer(canvas, screen_width, screen_height)

            # Draw gaze point last so it stays visible above UI
            if gaze_draw_pos is not None:
                cv2.circle(canvas, gaze_draw_pos, 10, (0, 255, 0), -1)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_gaze_typing_suite()

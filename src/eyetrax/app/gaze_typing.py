import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

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


# ---------------------- 虚拟键盘定义 ---------------------- #
@dataclass
class KeyRect:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int


class OnScreenKeyboard:
    """
    屏幕中下部的大键盘：
    - 第一行：<-（撤回） + SPACE（空格）两个大键
    - 下面三行：QWERTY / ASDFG / ZXCVB...
    """

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # 键盘区域，从屏幕高度 20% 开始到底
        self.top = int(screen_h * 0.2)
        self.bottom = screen_h
        self.height = self.bottom - self.top

        # 上面功能键行
        self.func_row = ["SPACE","<-",]

        # 三行字母
        self.rows = [
            list("QWERTYUIOP"),   # 第一行 10 个字母
            list("ASDFGHJKL"),    # 第二行 9 个字母
            list("ZXCVBNM"),      # 第三行 7 个字母
        ]

        self.key_rects: List[KeyRect] = []
        self._build_layout()

    def _build_layout(self):
        # 整个键盘区域分成 4 行（1 行功能键 + 3 行字母）
        row_height = int(self.height * 0.22)
        gap = 5  # 键之间的间隙

        # ---------- 第一行：<- 和 SPACE ----------
        y1 = self.top
        y2 = y1 + row_height - gap

        cols = len(self.func_row)  # 2 个键
        # 每个大约占屏幕宽度的 1/3
        key_width = int(self.screen_w / 3)
        total_width = key_width * cols
        x_start = (self.screen_w - total_width) // 2

        for j, label in enumerate(self.func_row):
            x1 = x_start + j * key_width + gap
            x2 = x1 + key_width - gap
            self.key_rects.append(KeyRect(label, x1, y1, x2, y2))

        # ---------- 下面三行字母键盘 ----------
        for i, row in enumerate(self.rows):
            y1 = self.top + (i + 1) * row_height  # 注意 +1，往下挪一行
            y2 = y1 + row_height - gap
            cols = len(row)

            # 字母键稍微窄一些，多放几个
            key_width = int(self.screen_w / 10)
            total_width = key_width * cols
            x_start = (self.screen_w - total_width) // 2

            for j, label in enumerate(row):
                x1 = x_start + j * key_width + gap
                x2 = x1 + key_width - gap
                self.key_rects.append(KeyRect(label, x1, y1, x2, y2))

    def find_key_at(self, x: float, y: float) -> Optional[KeyRect]:
        """根据 gaze 坐标找到所在按键"""
        for key in self.key_rects:
            if key.x1 <= x <= key.x2 and key.y1 <= y <= key.y2:
                return key
        return None

    def draw(
        self,
        canvas: np.ndarray,
        highlighted: Optional[KeyRect],
        target_text: str,
        typed_text: str,
        dwell_progress: float,
        bar_color=(0, 0, 0),
        target_color=(255, 255, 255),
        typed_color=(0, 255, 255),
        target_scale: float = 2.0,
        typed_scale: float = 2.0,
        target_thickness: int = 3,
        typed_thickness: int = 3,
        text_box_border_color=(50, 50, 50),
        text_box_border_thickness: int = 0,
        text_box_margin: int = 0,
        text_box_width_ratio: float = 0.7,
        text_box_height_ratio: float = 0.8,
        text_box_fill_color: tuple[int, int, int] | None = None,
    ):
        """
        Draw top text region + keyboard and gaze progress indicator.
        Colors and scales are configurable for reuse in different UIs.
        """
        h, w = canvas.shape[:2]

        # Top text background
        top_bar_h = int(h * 0.18)
        cv2.rectangle(canvas, (0, 0), (w, top_bar_h), bar_color, thickness=-1)
        # Centered text box inside the bar
        box_w = int(w * max(0.2, min(0.9, text_box_width_ratio)))
        box_h = int(top_bar_h * max(0.3, min(1.0, text_box_height_ratio)))
        box_x1 = (w - box_w) // 2
        box_y1 = (top_bar_h - box_h) // 2
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        if text_box_margin > 0:
            box_x1 += text_box_margin
            box_x2 -= text_box_margin
            box_y1 += text_box_margin
            box_y2 -= text_box_margin
        if text_box_fill_color is not None:
            cv2.rectangle(
                canvas,
                (box_x1, box_y1),
                (box_x2, box_y2),
                text_box_fill_color,
                thickness=-1,
            )
        if text_box_border_thickness > 0:
            cv2.rectangle(
                canvas,
                (box_x1, box_y1),
                (box_x2, box_y2),
                text_box_border_color,
                thickness=text_box_border_thickness,
            )

        # Target text (optional)
        have_target = bool(target_text.strip())
        if have_target:
            target_str = f"Target: {target_text}"
            (target_w, target_h), _ = cv2.getTextSize(
                target_str, cv2.FONT_HERSHEY_SIMPLEX, target_scale, target_thickness
            )
            target_x = box_x1 + (box_w - target_w) // 2
            target_y = box_y1 + int(box_h * 0.4)
            cv2.putText(
                canvas,
                target_str,
                (target_x, target_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                target_scale,
                target_color,
                target_thickness,
                cv2.LINE_AA,
            )

        # Typed text
        typed_str = f"Typed : {typed_text}"
        (typed_w, typed_h), _ = cv2.getTextSize(
            typed_str, cv2.FONT_HERSHEY_SIMPLEX, typed_scale, typed_thickness
        )
        typed_x = box_x1 + (box_w - typed_w) // 2
        if have_target:
            typed_y = box_y1 + int(box_h * 0.78)
        else:
            typed_y = box_y1 + (box_h // 2 + typed_h // 2)
        cv2.putText(
            canvas,
            typed_str,
            (typed_x, typed_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            typed_scale,
            typed_color,
            typed_thickness,
            cv2.LINE_AA,
        )

        # Keyboard background (semi-transparent)
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (0, self.top),
            (self.screen_w, self.bottom),
            (40, 40, 40),
            thickness=-1,
        )
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)

        # Keys
        for key in self.key_rects:
            color = (200, 200, 200)
            thickness = 2
            if highlighted is not None and key.label == highlighted.label:
                color = (0, 255, 255)
                thickness = 3
            cv2.rectangle(
                canvas,
                (key.x1, key.y1),
                (key.x2, key.y2),
                color,
                thickness=thickness,
            )

            text_scale = 4
            if key.label in ["SPACE", "<-"]:
                text_scale = 4

            text_size, _ = cv2.getTextSize(
                key.label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2
            )
            text_w, text_h = text_size
            text_x = key.x1 + (key.x2 - key.x1 - text_w) // 2
            text_y = key.y1 + (key.y2 - key.y1 + text_h) // 2
            cv2.putText(
                canvas,
                key.label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Dwell progress ring
        if highlighted is not None and dwell_progress > 0:
            cx = (highlighted.x1 + highlighted.x2) // 2
            cy = (highlighted.y1 + highlighted.y2) // 2
            radius = int(
                min(highlighted.x2 - highlighted.x1, highlighted.y2 - highlighted.y1)
                * 0.35
            )
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


def apply_key(label: str, current: str) -> str:
    """按键逻辑"""
    if label == "SPACE":
        return current + " "
    if label == "ENTER":
        return current + "\n"
    if label == "<-":
        return current[:-1] if current else current
    return current + label


# ---------------------- 主逻辑：和 demo 前半段一样 ---------------------- #

def run_gaze_typing():
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator(model_name=args.model)

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[typing] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    screen_width, screen_height = get_screen_size()

    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(
            screen_width, screen_height, confidence=confidence_level
        )
    else:
        kalman = None
        smoother = NoSmoother()

    # 背景（可用和 demo 一样的）
    if background_path and os.path.isfile(background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (screen_width, screen_height))
    else:
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        background[:] = (30, 30, 30)

    keyboard = OnScreenKeyboard(screen_width, screen_height)

    # 打字测试：可以先固定一个简单单词
    target_text = "HELLO"
    typed_text = ""

    DWELL_TIME = 1          # 凝视多久自动选中（秒）
    SELECT_COOLDOWN = 0.7     # 两次选中之间的冷却时间（防止抖动重复触发）

    last_key_label: Optional[str] = None
    dwell_start_time: Optional[float] = None
    last_select_time = 0.0

    with camera(camera_index) as cap, fullscreen("Gaze Typing"):
        prev_time = time.time()

        for frame in iter_frames(cap):
            now = time.time()
            dt = now - prev_time
            prev_time = now

            features, blink_detected = gaze_estimator.extract_features(frame)

            if features is not None:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)
                x_pred, y_pred = smoother.step(x, y)
            else:
                x_pred = y_pred = None

            canvas = background.copy()

            # 画一个小的 gaze 点，用于调试（居中偏上的区域）
            dwell_progress = 0.0
            highlighted_key: Optional[KeyRect] = None

            if x_pred is not None and y_pred is not None:
                # 限制范围
                x_pred = max(0, min(screen_width - 1, x_pred))
                y_pred = max(0, min(screen_height - 1, y_pred))

                cv2.circle(
                    canvas,
                    (int(x_pred), int(y_pred)),
                    10,
                    (0, 255, 0),
                    thickness=-1,
                )

                # 找到当前 gaze 所在的键
                highlighted_key = keyboard.find_key_at(x_pred, y_pred)

                # 凝视逻辑
                if highlighted_key is not None:
                    if last_key_label == highlighted_key.label:
                        if dwell_start_time is None:
                            dwell_start_time = now
                        dwell_progress = (now - dwell_start_time) / DWELL_TIME
                    else:
                        last_key_label = highlighted_key.label
                        dwell_start_time = now
                        dwell_progress = 0.0
                else:
                    last_key_label = None
                    dwell_start_time = None
                    dwell_progress = 0.0
            else:
                last_key_label = None
                dwell_start_time = None
                dwell_progress = 0.0

            # 选键逻辑：眨眼 或 凝视达到阈值
            selected_label = None

            # 1) 眨眼选择当前键
            if blink_detected and highlighted_key is not None:
                if now - last_select_time > SELECT_COOLDOWN:
                    selected_label = highlighted_key.label
                    last_select_time = now

            # 2) 凝视时间超过阈值自动选择
            if (
                highlighted_key is not None
                and dwell_start_time is not None
                and (now - dwell_start_time) >= DWELL_TIME
            ):
                if now - last_select_time > SELECT_COOLDOWN:
                    selected_label = highlighted_key.label
                    last_select_time = now
                    dwell_start_time = now  # 重置

            # 应用按键
            if selected_label is not None:
                typed_text = apply_key(selected_label, typed_text)
                print(f"[typing] Selected key: {selected_label}, text = {typed_text!r}")

            # 画键盘 + 文本
            keyboard.draw(
                canvas,
                highlighted_key,
                target_text,
                typed_text,
                dwell_progress,
            )

            # FPS & blink 信息（放在左上角下面一点）
            fps = 1.0 / max(dt, 1e-6)
            cv2.putText(
                canvas,
                f"FPS: {int(fps)}",
                (40, int(screen_height * 0.22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            blink_txt = "Blink: YES" if blink_detected else "Blink: NO"
            blink_clr = (0, 0, 255) if blink_detected else (0, 255, 0)
            cv2.putText(
                canvas,
                blink_txt,
                (40, int(screen_height * 0.28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                blink_clr,
                2,
                cv2.LINE_AA,
            )

            # 简单的是否打对提示（绿色/红色点一下）
            if len(typed_text.strip()) > 0:
                correct_prefix = target_text.startswith(typed_text.strip())
                color = (0, 255, 0) if correct_prefix else (0, 0, 255)
                cv2.circle(canvas, (screen_width - 60, 60), 15, color, -1)

            cv2.imshow("Gaze Typing", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 退出
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_gaze_typing()

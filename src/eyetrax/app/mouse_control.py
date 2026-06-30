import os
import time

import cv2
import numpy as np
import pyautogui

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames

pyautogui.FAILSAFE = False  # 防止鼠标到左上角自动抛异常退出


def map_to_system_screen(x, y, model_screen_size):
    """
    把 gaze 模型输出的屏幕坐标 (x, y) 映射到当前系统主屏分辨率。

    如果你的 gaze 模型就是用当前这块主屏标定的，理论上分辨率会一样，
    这里做一个比例映射，兼容分辨率不一致/多屏的情况。
    """
    model_w, model_h = model_screen_size
    sys_w, sys_h = pyautogui.size()

    model_w = max(1, model_w)
    model_h = max(1, model_h)

    mx = int(x / model_w * sys_w)
    my = int(y / model_h * sys_h)

    mx = max(0, min(sys_w - 1, mx))
    my = max(0, min(sys_h - 1, my))
    return mx, my


def run_mouse_control():
    # ========= 完全复用 demo.py 的“前半段”：参数解析 + 校准 + 滤波器 =========
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator(model_name=args.model)

    # 模型加载 / 校准：和 demo.py 一模一样
    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[mouse_control] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    # 屏幕尺寸：这是 gaze 预测坐标的“参考尺寸”
    screen_width, screen_height = get_screen_size()
    model_screen_size = (screen_width, screen_height)

    # 滤波器：也和 demo.py 一样
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

    # ========= 从这里开始：不再画背景 + 光标，改成控制“系统鼠标” =========

    # 眨眼点击相关参数
    last_blink_time = 0.0
    BLINK_CLICK_COOLDOWN = 0.8  # 两次眨眼点击间隔（秒）

    # 开关状态
    mouse_enabled = False          # 是否启用眼控鼠标
    blink_click_enabled = True     # 是否启用眨眼点击

    print("[mouse_control] 启动成功：")
    print("  - ESC：退出程序")
    print("  - T：切换眼控鼠标 ON/OFF（需要 debug 窗口在前台）")
    print("  - B：切换眨眼点击 ON/OFF（需要 debug 窗口在前台）")


    # debug 窗口
    window_name = "Eye Mouse (debug)"

    with camera(camera_index) as cap:
        prev_time = time.time()

        for frame in iter_frames(cap):
            # 提取特征 + 眨眼检测
            features, blink_detected = gaze_estimator.extract_features(frame)

            if features is not None and not blink_detected:
                # 和 demo.py 一样，用单帧特征预测 gaze 点
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)
                x_pred, y_pred = smoother.step(x, y)
            else:
                x_pred = y_pred = None
                # 当做“眨眼中 / 无法可靠估计”，此时不更新鼠标位置

            # === 1. 只有在 mouse_enabled 为 True 时才移动系统鼠标 ===
            if mouse_enabled and x_pred is not None and y_pred is not None:
                mx, my = map_to_system_screen(x_pred, y_pred, model_screen_size)
                # duration 越小越“跟手”，但也可能有一点抖
                pyautogui.moveTo(mx, my, duration=0.01)

            # === 2. 眨眼点击（受 blink_click_enabled 和 mouse_enabled 控制） ===
            now = time.time()
            if mouse_enabled and blink_click_enabled and blink_detected:
                if now - last_blink_time > BLINK_CLICK_COOLDOWN:
                    pyautogui.click()
                    last_blink_time = now

            # === 3. Debug 窗口：显示 FPS / 眨眼状态 / 开关状态 ===
            debug_frame = frame.copy()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                debug_frame,
                f"FPS: {int(fps)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            blink_txt = "Blinking" if blink_detected else "Not Blinking"
            blink_clr = (0, 0, 255) if blink_detected else (0, 255, 0)
            cv2.putText(
                debug_frame,
                blink_txt,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                blink_clr,
                2,
                cv2.LINE_AA,
            )

            # 显示开关状态
            mouse_txt = f"Mouse: {'ON' if mouse_enabled else 'OFF'}"
            blink_click_txt = f"BlinkClick: {'ON' if blink_click_enabled else 'OFF'}"
            cv2.putText(
                debug_frame,
                mouse_txt,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255) if mouse_enabled else (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_frame,
                blink_click_txt,
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255) if blink_click_enabled else (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

            # 把预测的 gaze 点大致画在 debug 画面上，方便调试
            if x_pred is not None and y_pred is not None:
                dx = int(x_pred / screen_width * debug_frame.shape[1])
                dy = int(y_pred / screen_height * debug_frame.shape[0])
                cv2.circle(debug_frame, (dx, dy), 5, (0, 255, 0), -1)

            cv2.imshow(window_name, debug_frame)

            # === 4. 处理键盘输入：ESC / T / B ===
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC 退出
                print("[mouse_control] ESC pressed, exiting.")
                break

            # T：切换眼控鼠标
            if key == ord("t"):
                mouse_enabled = not mouse_enabled
                print(f"[mouse_control] Mouse control: {'ON' if mouse_enabled else 'OFF'}")

            # B：切换眨眼点击
            if key == ord("b"):
                blink_click_enabled = not blink_click_enabled
                print(f"[mouse_control] Blink click: {'ON' if blink_click_enabled else 'OFF'}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_mouse_control()

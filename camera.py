#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import subprocess
import time
import os
from datetime import datetime


DEVICE = "/dev/video0"
CAMERA_INDEX = 0

# 摄像头工作模式
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

# True：尝试使用 MJPG，通常比 YUYV 更节省 USB 带宽
USE_MJPG = True


# ============================================================
# 滑轨参数配置
#
# 格式：
# 显示名称: {
#     "control": v4l2 控制项名称,
#     "min": 最小值,
#     "max": 最大值,
#     "default": 初始值
# }
# ============================================================

CONTROL_CONFIG = {
    "Brightness": {
        "control": "brightness",
        "min": -64,
        "max": 64,
        "default": 0,
    },
    "Contrast": {
        "control": "contrast",
        "min": 0,
        "max": 64,
        "default": 32,
    },
    "Saturation": {
        "control": "saturation",
        "min": 0,
        "max": 128,
        "default": 60,
    },
    "Hue": {
        "control": "hue",
        "min": -40,
        "max": 40,
        "default": 0,
    },
    "Gamma": {
        "control": "gamma",
        "min": 72,
        "max": 500,
        "default": 100,
    },
    "Gain": {
        "control": "gain",
        "min": 0,
        "max": 100,
        "default": 20,
    },
    "Sharpness": {
        "control": "sharpness",
        "min": 0,
        "max": 6,
        "default": 2,
    },
    "Backlight": {
        "control": "backlight_compensation",
        "min": 0,
        "max": 2,
        "default": 1,
    },
    "Exposure": {
        "control": "exposure_absolute",
        "min": 1,
        "max": 5000,
        "default": 157,
    },
    "White Balance": {
        "control": "white_balance_temperature",
        "min": 2800,
        "max": 6500,
        "default": 4600,
    },
}


WINDOW_CAMERA = "Camera Preview"
WINDOW_CONTROL = "Camera Controls"

# 避免程序初始化滑轨时重复发送命令
initializing = True

# 防止滑轨连续拖动时过于频繁执行命令
last_update_time = {}
UPDATE_INTERVAL = 0.03


def run_command(command):
    """
    执行 Linux 命令。
    返回：
        True  表示执行成功
        False 表示执行失败
    """
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if result.returncode != 0:
            error_message = result.stderr.strip()
            if error_message:
                print("[命令执行失败]", " ".join(command))
                print(error_message)
            return False

        return True

    except Exception as error:
        print("[异常]", error)
        return False


def set_v4l2_control(control_name, value, quiet=False):
    """
    使用 v4l2-ctl 设置摄像头参数。
    """
    command = [
        "v4l2-ctl",
        "--device={}".format(DEVICE),
        "--set-ctrl={}={}".format(control_name, int(value))
    ]

    success = run_command(command)

    if success and not quiet:
        print("{} = {}".format(control_name, int(value)))

    return success


def get_v4l2_control(control_name):
    """
    读取指定摄像头参数。
    """
    command = [
        "v4l2-ctl",
        "--device={}".format(DEVICE),
        "--get-ctrl={}".format(control_name)
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if result.returncode != 0:
            return None

        # 输出形式通常为：
        # brightness: 0
        output = result.stdout.strip()

        if ":" not in output:
            return None

        value_text = output.split(":")[-1].strip()
        return int(value_text)

    except Exception:
        return None


def slider_to_real_value(slider_value, minimum):
    """
    OpenCV 滑轨本身不能使用负数。
    因此将滑轨值转换成实际控制值。
    """
    return int(slider_value) + int(minimum)


def real_value_to_slider(real_value, minimum):
    """
    将摄像头真实参数转换成 OpenCV 滑轨位置。
    """
    return int(real_value) - int(minimum)


def create_callback(display_name):
    """
    为每个滑轨生成独立的回调函数。
    """
    def callback(slider_value):
        global initializing

        if initializing:
            return

        config = CONTROL_CONFIG[display_name]
        real_value = slider_to_real_value(
            slider_value,
            config["min"]
        )

        current_time = time.time()
        previous_time = last_update_time.get(display_name, 0)

        if current_time - previous_time < UPDATE_INTERVAL:
            return

        last_update_time[display_name] = current_time

        set_v4l2_control(
            config["control"],
            real_value,
            quiet=True
        )

    return callback


def create_switch_callback(control_name):
    """
    为自动曝光、自动白平衡等开关创建回调函数。
    """
    def callback(value):
        global initializing

        if initializing:
            return

        if control_name == "white_balance_temperature_auto":
            # 0：关闭自动白平衡
            # 1：开启自动白平衡
            set_v4l2_control(control_name, value)

        elif control_name == "exposure_auto":
            # 当前摄像头定义：
            # 1：手动曝光
            # 3：自动曝光
            exposure_value = 3 if value == 1 else 1
            set_v4l2_control(control_name, exposure_value)

    return callback


def update_manual_control_state():
    """
    根据自动曝光和自动白平衡状态，在画面上给出提示。
    OpenCV 滑轨不能真正禁用，因此这里只进行状态说明。
    """
    auto_wb = cv2.getTrackbarPos("Auto White Balance", WINDOW_CONTROL)
    auto_exposure = cv2.getTrackbarPos("Auto Exposure", WINDOW_CONTROL)

    return auto_wb, auto_exposure


def apply_initial_values():
    """
    将所有参数恢复到程序中设定的初始值。
    """
    global initializing
    initializing = True

    # 先关闭自动模式，确保手动值能够设置
    set_v4l2_control(
        "white_balance_temperature_auto",
        0,
        quiet=True
    )

    set_v4l2_control(
        "exposure_auto",
        1,
        quiet=True
    )

    cv2.setTrackbarPos(
        "Auto White Balance",
        WINDOW_CONTROL,
        0
    )

    cv2.setTrackbarPos(
        "Auto Exposure",
        WINDOW_CONTROL,
        0
    )

    for display_name, config in CONTROL_CONFIG.items():
        value = config["default"]

        set_v4l2_control(
            config["control"],
            value,
            quiet=True
        )

        slider_position = real_value_to_slider(
            value,
            config["min"]
        )

        cv2.setTrackbarPos(
            display_name,
            WINDOW_CONTROL,
            slider_position
        )

    initializing = False
    print("参数已恢复为程序默认值。")


def read_current_values():
    """
    从摄像头读取当前参数，并同步到滑轨。
    """
    global initializing
    initializing = True

    auto_wb_value = get_v4l2_control(
        "white_balance_temperature_auto"
    )

    exposure_auto_value = get_v4l2_control(
        "exposure_auto"
    )

    if auto_wb_value is not None:
        cv2.setTrackbarPos(
            "Auto White Balance",
            WINDOW_CONTROL,
            1 if auto_wb_value else 0
        )

    if exposure_auto_value is not None:
        cv2.setTrackbarPos(
            "Auto Exposure",
            WINDOW_CONTROL,
            1 if exposure_auto_value == 3 else 0
        )

    for display_name, config in CONTROL_CONFIG.items():
        current_value = get_v4l2_control(config["control"])

        if current_value is None:
            current_value = config["default"]

        current_value = max(
            config["min"],
            min(config["max"], current_value)
        )

        slider_position = real_value_to_slider(
            current_value,
            config["min"]
        )

        cv2.setTrackbarPos(
            display_name,
            WINDOW_CONTROL,
            slider_position
        )

    initializing = False


def print_current_values():
    """
    在终端中打印全部参数。
    """
    print("\n" + "=" * 55)
    print("当前摄像头参数")
    print("=" * 55)

    auto_wb = get_v4l2_control(
        "white_balance_temperature_auto"
    )

    exposure_auto = get_v4l2_control(
        "exposure_auto"
    )

    print(
        "white_balance_temperature_auto = {}".format(auto_wb)
    )
    print(
        "exposure_auto = {}".format(exposure_auto)
    )

    for display_name, config in CONTROL_CONFIG.items():
        value = get_v4l2_control(config["control"])
        print(
            "{:<32} = {}".format(
                config["control"],
                value
            )
        )

    print("=" * 55 + "\n")


def save_current_values(filename="camera_settings.txt"):
    """
    将摄像头参数保存为文本文件。
    """
    values = []

    values.append(
        "white_balance_temperature_auto={}".format(
            get_v4l2_control(
                "white_balance_temperature_auto"
            )
        )
    )

    values.append(
        "exposure_auto={}".format(
            get_v4l2_control("exposure_auto")
        )
    )

    for display_name, config in CONTROL_CONFIG.items():
        value = get_v4l2_control(config["control"])
        values.append(
            "{}={}".format(config["control"], value)
        )

    try:
        with open(filename, "w") as file:
            file.write("\n".join(values))
            file.write("\n")

        print("参数已保存到：{}".format(
            os.path.abspath(filename)
        ))

    except Exception as error:
        print("参数保存失败：", error)


def draw_text(frame, text, x, y):
    """
    在画面上绘制带阴影的文字。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        font,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


def open_camera():
    """
    打开摄像头并设置分辨率、帧率和编码格式。
    """
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    if not camera.isOpened():
        return None

    if USE_MJPG:
        fourcc = cv2.VideoWriter_fourcc(
            "M", "J", "P", "G"
        )
        camera.set(cv2.CAP_PROP_FOURCC, fourcc)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        FRAME_FPS
    )

    return camera


def main():
    global initializing

    if not os.path.exists(DEVICE):
        print("错误：没有找到摄像头设备 {}".format(DEVICE))
        return

    print("正在打开摄像头……")

    camera = open_camera()

    if camera is None:
        print("错误：无法打开摄像头。")
        print("请检查 /dev/video0 是否被其他程序占用。")
        return

    actual_width = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    actual_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print("摄像头打开成功")
    print(
        "实际分辨率：{} x {}".format(
            actual_width,
            actual_height
        )
    )
    print("摄像头报告帧率：{:.2f}".format(actual_fps))

    cv2.namedWindow(
        WINDOW_CAMERA,
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        WINDOW_CONTROL,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_CAMERA,
        actual_width,
        actual_height
    )

    cv2.resizeWindow(
        WINDOW_CONTROL,
        600,
        700
    )

    # 创建自动控制开关
    cv2.createTrackbar(
        "Auto White Balance",
        WINDOW_CONTROL,
        0,
        1,
        create_switch_callback(
            "white_balance_temperature_auto"
        )
    )

    cv2.createTrackbar(
        "Auto Exposure",
        WINDOW_CONTROL,
        0,
        1,
        create_switch_callback("exposure_auto")
    )

    # 创建摄像头参数滑轨
    for display_name, config in CONTROL_CONFIG.items():
        slider_maximum = (
            config["max"] - config["min"]
        )

        initial_position = real_value_to_slider(
            config["default"],
            config["min"]
        )

        cv2.createTrackbar(
            display_name,
            WINDOW_CONTROL,
            initial_position,
            slider_maximum,
            create_callback(display_name)
        )

    # 读取摄像头当前值，而不是立即覆盖
    read_current_values()

    print("\n操作说明：")
    print("拖动 Camera Controls 窗口中的滑轨调整参数")
    print("q 或 ESC：退出")
    print("r：恢复程序默认参数")
    print("p：打印当前参数")
    print("w：将参数保存到 camera_settings.txt")
    print("s：保存当前画面")
    print()

    previous_time = time.time()
    display_fps = 0.0
    fps_counter = 0
    fps_start_time = time.time()

    consecutive_failures = 0

    while True:
        success, frame = camera.read()

        if not success or frame is None:
            consecutive_failures += 1
            print(
                "\r摄像头读取失败，连续失败次数：{}".format(
                    consecutive_failures
                ),
                end=""
            )

            if consecutive_failures >= 30:
                print("\n摄像头连续读取失败，程序退出。")
                break

            time.sleep(0.02)
            continue

        consecutive_failures = 0

        # 计算实际显示帧率
        fps_counter += 1
        current_time = time.time()

        if current_time - fps_start_time >= 1.0:
            display_fps = fps_counter / (
                current_time - fps_start_time
            )
            fps_counter = 0
            fps_start_time = current_time

        auto_wb, auto_exposure = \
            update_manual_control_state()

        draw_text(
            frame,
            "FPS: {:.1f}".format(display_fps),
            10,
            25
        )

        draw_text(
            frame,
            "Resolution: {}x{}".format(
                frame.shape[1],
                frame.shape[0]
            ),
            10,
            48
        )

        exposure_mode_text = (
            "Auto" if auto_exposure else "Manual"
        )

        white_balance_mode_text = (
            "Auto" if auto_wb else "Manual"
        )

        draw_text(
            frame,
            "Exposure: {}".format(
                exposure_mode_text
            ),
            10,
            71
        )

        draw_text(
            frame,
            "White Balance: {}".format(
                white_balance_mode_text
            ),
            10,
            94
        )

        draw_text(
            frame,
            "Q/ESC Exit | R Reset | P Print | W Save settings | S Snapshot",
            10,
            frame.shape[0] - 15
        )

        cv2.imshow(WINDOW_CAMERA, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif key == ord("r"):
            apply_initial_values()

        elif key == ord("p"):
            print_current_values()

        elif key == ord("w"):
            save_current_values()

        elif key == ord("s"):
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = "camera_{}.jpg".format(
                timestamp
            )

            if cv2.imwrite(filename, frame):
                print(
                    "画面已保存到：{}".format(
                        os.path.abspath(filename)
                    )
                )
            else:
                print("画面保存失败。")

    camera.release()
    cv2.destroyAllWindows()

    print("\n程序已退出。")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实时动态手语识别 — 带手部骨架可视化 (6种: 你好/再见/对不起/我爱你/没关系/谢谢)"""
import argparse
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from utils import CvFpsCalc
from dynamic_recognizer import DynamicHandGestureRecognizer


def get_args():
    parser = argparse.ArgumentParser(description='实时动态手语识别')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--backend", choices=['any', 'dshow', 'msmf'], default='any')
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.35,
                        help='置信度阈值, 越低越容易触发识别 (默认0.35)')
    parser.add_argument("--stable_count", type=int, default=20,
                        help='连续多少帧结果一致才输出 (默认20)')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════
# 中文字体 & PIL 文字渲染
# ═══════════════════════════════════════════════════════

def _find_chinese_font():
    """查找系统可用的中文字体"""
    import os
    candidates = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑 Bold
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/simkai.ttf",     # 楷体
        "C:/Windows/Fonts/STKAITI.TTF",    # 华文楷体
        # 用户字体目录
        os.path.expanduser("~/.fonts/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


_FONT_PATH = _find_chinese_font()
_FONT_CACHE = {}  # 缓存不同字号的字体对象


def _get_font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def put_chinese_text(img, text, position, font_size=32, color=(0, 255, 255)):
    """在 OpenCV 图像上用 PIL 绘制中文文字"""
    if not text:
        return img

    # 转为 PIL Image
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)

    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # 画半透明背景
    px, py = position
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [px - 5, py - 2, px + tw + 15, py + th + 8],
        fill=(0, 0, 0, 180),
    )
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')

    # 画文字
    draw = ImageDraw.Draw(pil_img)
    draw.text((px + 5, py), text, font=font, fill=color)

    # 转回 OpenCV
    return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════
# 手部绘制
# ═══════════════════════════════════════════════════════

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, np.array([[x, y]]), axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])
    return landmark_point


def draw_landmarks(image, landmark_point):
    if len(landmark_point) == 0:
        return image

    connections = [
        (2, 3), (3, 4),           # 拇指
        (5, 6), (6, 7), (7, 8),   # 食指
        (9, 10), (10, 11), (11, 12),  # 中指
        (13, 14), (14, 15), (15, 16),  # 无名指
        (17, 18), (18, 19), (19, 20),  # 小指
        (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌心
    ]
    for s, e in connections:
        cv.line(image, tuple(landmark_point[s]), tuple(landmark_point[e]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[s]), tuple(landmark_point[e]),
                (255, 255, 255), 2)

    for i, (x, y) in enumerate(landmark_point):
        r = 8 if i in (4, 8, 12, 16, 20) else 5  # 指尖大圆, 关节点小圆
        cv.circle(image, (x, y), r, (255, 255, 255), -1)
        cv.circle(image, (x, y), r, (0, 0, 0), 1)

    return image


def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_handedness(image, brect, handedness):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    text = handedness.classification[0].label[0:]  # Left / Right
    cv.putText(image, text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


# ═══════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════

def main():
    args = get_args()
    print(f'中文字体: {_FONT_PATH or "未找到, 用英文显示"}', flush=True)

    # ── 摄像头 ───────────────────────────────────
    backend_map = {'any': cv.CAP_ANY, 'dshow': cv.CAP_DSHOW, 'msmf': cv.CAP_MSMF}
    cap = cv.VideoCapture(args.device, backend_map[args.backend])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f'Unable to open camera device: {args.device}', flush=True)
        return
    ret, first_image = cap.read()
    if not ret:
        print('Unable to read first frame', flush=True)
        cap.release()
        return

    print(f'Camera: device={args.device}, {args.width}x{args.height}', flush=True)
    cv.namedWindow('Dynamic Sign Language Recognition', cv.WINDOW_NORMAL)
    cv.imshow('Dynamic Sign Language Recognition', first_image)
    cv.waitKey(1)

    # ── MediaPipe ───────────────────────────
    hands = mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # ── 识别器 ──────────────────────────────
    dynamic_recognizer = DynamicHandGestureRecognizer(
        score_threshold=args.score_threshold,
        stable_count=args.stable_count,
    )

    print(f'Score threshold={args.score_threshold}, stable_count={args.stable_count}', flush=True)
    print('Ready! Press ESC to quit.', flush=True)

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    last_label_id = -1
    last_output_time = 0.0
    pending_image = first_image

    while True:
        fps = cvFpsCalc.get()

        if cv.waitKey(10) == 27:  # ESC
            break

        if pending_image is not None:
            image = pending_image
            pending_image = None
        else:
            ret, image = cap.read()
            if not ret:
                break

        image = cv.flip(image, 1)
        debug_image = image.copy()

        # ── 手语识别 ────────────────────────
        dynamic_result = dynamic_recognizer.recognize(image)
        label_id = dynamic_result["label_id"]
        label = dynamic_result["label"]
        score = dynamic_result["score"]

        now = time.monotonic()
        if label_id >= 0 and label_id != last_label_id and now - last_output_time >= 1.0:
            print(f'[识别] {label}  (置信度={score:.3f})', flush=True)
            last_output_time = now
        last_label_id = label_id if label_id >= 0 else last_label_id

        # ── 手部骨架 ────────────────────────
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_handedness(debug_image, brect, handedness)

        # ── 绘制识别结果(中文) ───────────────
        if label_id >= 0:
            display_text = f'{label}  {score:.2f}'
            color = (0, 255, 255)  # 黄色
        else:
            display_text = f'等待手势...'
            color = (150, 150, 150)

        debug_image = put_chinese_text(debug_image, display_text, (10, 75),
                                       font_size=40, color=color)

        # ── FPS ─────────────────────────────
        cv.putText(debug_image, f'FPS: {fps}', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(debug_image, f'FPS: {fps}', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow('Dynamic Sign Language Recognition', debug_image)

    cap.release()
    dynamic_recognizer.close()
    cv.destroyAllWindows()
    print('Finished.', flush=True)


if __name__ == '__main__':
    main()

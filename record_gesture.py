#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手势数据采集工具 — 录制你自己的手语关键点序列

用法:
  python record_gesture.py --label 你好      # 录制"你好"
  python record_gesture.py --label 谢谢      # 录制"谢谢"

操作:
  按 空格键  开始录制 (做完整手势, 约2-3秒)
  按 ESC    退出
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ═══════════════════════════════════════════════════════
# 中文字体
# ═══════════════════════════════════════════════════════

def _find_chinese_font():
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

_FONT_PATH = _find_chinese_font()
_FONT_CACHE = {}

def _get_font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def cv_put_chinese(img, text, pos, font_size=32, color=(0, 255, 0)):
    """用 PIL 在 OpenCV 图像上画中文"""
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    draw.text(pos, text, font=font, fill=color)
    return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)


HAND_FEATURE_DIM = 63
HAND_SLOT_LABELS = ('Left', 'Right')
CLASSES = ['你好', '再见', '对不起', '没关系', '谢谢',
           '上课', '下课', '不舒服', '厉害', '多少钱']


def get_args():
    parser = argparse.ArgumentParser(description='录制手势数据')
    parser.add_argument('--label', required=True, choices=CLASSES,
                        help='手语词名称')
    parser.add_argument('--output', default='my_dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--countdown', type=int, default=3,
                        help='按下空格后倒计时秒数')
    parser.add_argument('--record_seconds', type=float, default=2.5,
                        help='录制时长(秒)')
    return parser.parse_args()


def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]
    normalized = []
    for x, y, z in landmarks:
        normalized.extend([x - base_x, y - base_y, z - base_z])
    max_value = max(abs(v) for v in normalized)
    if max_value == 0:
        return normalized
    return [v / max_value for v in normalized]


def make_frame_features(results):
    features = {label: [0.0] * HAND_FEATURE_DIM for label in HAND_SLOT_LABELS}
    if not results.multi_hand_landmarks:
        return None

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = ''
        if results.multi_handedness and index < len(results.multi_handedness):
            handedness = results.multi_handedness[index].classification[0].label
        if handedness not in features:
            continue
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        features[handedness] = normalize_landmarks(landmarks)

    return features['Left'] + features['Right']


def resample_sequence(sequence, target_length):
    if not sequence or len(sequence) < 3:
        return None
    if len(sequence) == target_length:
        return sequence
    indexes = np.linspace(0, len(sequence) - 1, target_length).astype(int)
    return [sequence[i] for i in indexes]


def main():
    args = get_args()

    output_dir = Path(args.output) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找已有文件编号
    existing = list(output_dir.glob('*.csv'))
    next_id = len(existing) + 1

    # ── 摄像头 ──
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        print('摄像头打不开!', flush=True)
        return

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    print(f'录制手势: {args.label}')
    print(f'  按 空格键 开始录制 ({args.countdown}s 倒计时, 录制 {args.record_seconds}s)')
    print(f'  按 ESC 退出')
    print(f'  已录制 {len(existing)} 条, 下一条编号: {next_id}')
    print(f'  保存到: {output_dir}')

    state = 'idle'  # idle | countdown | recording
    state_start = 0.0
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        display = frame.copy()

        # MediaPipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # ── 状态机 ──
        now = time.time()

        if state == 'countdown':
            elapsed = now - state_start
            remaining = args.countdown - int(elapsed)
            if remaining <= 0:
                # 开始录制
                state = 'recording'
                state_start = now
                sequence = []
                print('  录制中...', end='', flush=True)
            else:
                cv.putText(display, str(remaining), (display.shape[1]//2 - 30, display.shape[0]//2),
                           cv.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 5, cv.LINE_AA)

        elif state == 'recording':
            elapsed = now - state_start
            # 采集特征
            feat = make_frame_features(results)
            if feat is not None:
                sequence.append(feat)

            # 进度条
            progress = min(elapsed / args.record_seconds, 1.0)
            bar_w = int(display.shape[1] * 0.6)
            cv.rectangle(display, (display.shape[1]//2 - bar_w//2, display.shape[0] - 60),
                         (display.shape[1]//2 + bar_w//2, display.shape[0] - 40),
                         (255, 255, 255), 2)
            cv.rectangle(display, (display.shape[1]//2 - bar_w//2, display.shape[0] - 60),
                         (display.shape[1]//2 - bar_w//2 + int(bar_w * progress), display.shape[0] - 40),
                         (0, 255, 0), -1)

            if elapsed >= args.record_seconds:
                # 录制完成 → 重采样 → 保存
                state = 'idle'
                resampled = resample_sequence(sequence, args.sequence_length)
                if resampled:
                    csv_path = output_dir / f'{next_id:04d}.csv'
                    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        flattened = np.array(resampled, dtype=np.float32).flatten()
                        label_id = CLASSES.index(args.label)
                        writer.writerow([label_id, *flattened])
                    print(f'\r  已保存: {csv_path} ({len(sequence)}帧 -> {args.sequence_length}帧)', flush=True)
                    next_id += 1
                    # 更新 label 文件
                    label_csv = output_dir.parent / 'label.csv'
                    with open(label_csv, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        for c in CLASSES:
                            writer.writerow([c])
                else:
                    print('\r  录制失败: 未检测到手', flush=True)

        # 绘制手部
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * display.shape[1])
                    y = int(lm.y * display.shape[0])
                    points.append([x, y])
                for s, e in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
                              (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),
                              (17,18),(18,19),(19,20),(0,17)]:
                    cv.line(display, tuple(points[s]), tuple(points[e]), (0,255,0), 2)
                for x, y in points:
                    cv.circle(display, (x, y), 3, (0, 255, 0), -1)

        # 状态提示
        if state == 'idle':
            display = cv_put_chinese(display, f'{args.label} | 已录{next_id-1}条 | 空格=录制',
                                     (10, 8), font_size=28, color=(0, 255, 0))

        cv.imshow('Record Gesture', display)

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        if key == 32 and state == 'idle':  # 空格
            state = 'countdown'
            state_start = time.time()

    cap.release()
    hands.close()
    cv.destroyAllWindows()
    print(f'完成! 共录制 {next_id - 1} 条 {args.label} 样本')
    print(f'下一步: python train_from_my_data.py')


if __name__ == '__main__':
    main()

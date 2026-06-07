#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
HAND_FEATURE_DIM = 63
HAND_SLOT_LABELS = ('Left', 'Right')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--output', default='model/dynamic_gesture_classifier')
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--max_num_hands', type=int, default=2)
    parser.add_argument('--min_detection_confidence', type=float, default=0.7)
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5)
    return parser.parse_args()


def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]
    normalized = []

    for x, y, z in landmarks:
        normalized.extend([x - base_x, y - base_y, z - base_z])

    max_value = max(abs(value) for value in normalized)
    if max_value == 0:
        return normalized

    return [value / max_value for value in normalized]


def make_frame_features(results):
    frame_features = {
        label: [0.0] * HAND_FEATURE_DIM
        for label in HAND_SLOT_LABELS
    }
    if not results.multi_hand_landmarks:
        return None

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = ''
        if results.multi_handedness and index < len(results.multi_handedness):
            handedness = results.multi_handedness[index].classification[0].label

        if handedness not in frame_features:
            continue

        landmarks = [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks.landmark
        ]
        frame_features[handedness] = normalize_landmarks(landmarks)

    return frame_features['Left'] + frame_features['Right']


def extract_video_sequence(video_path, hands):
    cap = cv.VideoCapture(str(video_path))
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = hands.process(rgb_image)
        rgb_image.flags.writeable = True

        frame_features = make_frame_features(results)
        if frame_features is not None:
            sequence.append(frame_features)

    cap.release()
    return sequence


def resample_sequence(sequence, sequence_length):
    if not sequence:
        return None

    indexes = np.linspace(0, len(sequence) - 1, sequence_length).astype(int)
    return [sequence[index] for index in indexes]


def list_label_dirs(dataset_dir):
    return sorted([
        path for path in dataset_dir.iterdir()
        if path.is_dir() and not path.name.startswith('.')
    ], key=lambda path: path.name)


def main():
    args = get_args()
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dirs = list_label_dirs(dataset_dir)
    labels = {str(index): path.name for index, path in enumerate(label_dirs)}

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    csv_path = output_dir / 'dynamic_gesture_dataset.csv'
    total_count = 0
    skipped_count = 0

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for label_id, label_dir in enumerate(label_dirs):
            video_paths = [
                path for path in label_dir.iterdir()
                if path.suffix.lower() in VIDEO_EXTENSIONS
            ]

            for video_path in sorted(video_paths, key=lambda path: path.name):
                sequence = extract_video_sequence(video_path, hands)
                sequence = resample_sequence(sequence, args.sequence_length)
                if sequence is None:
                    skipped_count += 1
                    print(f'Skipped(no hand): {video_path}', flush=True)
                    continue

                flattened = np.array(sequence, dtype=np.float32).flatten()
                writer.writerow([label_id, *flattened])
                total_count += 1
                print(f'Processed: {video_path}', flush=True)

    hands.close()

    with open(output_dir / 'dynamic_gesture_label.csv',
              'w',
              newline='',
              encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for index in range(len(labels)):
            writer.writerow([labels[str(index)]])

    with open(output_dir / 'labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(
        f'Finished. samples={total_count}, skipped={skipped_count}, output={csv_path}',
        flush=True,
    )


if __name__ == '__main__':
    main()

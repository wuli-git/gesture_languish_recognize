#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp

from model.dynamic_gesture_classifier import DynamicGestureClassifier


HAND_FEATURE_DIM = 63
HAND_SLOT_LABELS = ('Left', 'Right')

NO_RESULT = {
    "label_id": -1,
    "label": "未识别",
    "score": 0.0,
    "type": "dynamic",
}


def load_labels(path='model/dynamic_gesture_classifier/dynamic_gesture_label.csv'):
    with open(path, encoding='utf-8-sig') as f:
        return [row[0] for row in csv.reader(f) if row]


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


def collect_landmarks(results):
    """Flatten all detected hands into a list of normalized [x, y] points.

    Each hand contributes 21 consecutive points, so downstream consumers can
    chunk the list by 21 to redraw per-hand skeletons.
    """
    points = []
    if not results.multi_hand_landmarks:
        return points

    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            points.append([landmark.x, landmark.y])

    return points


class DynamicHandGestureRecognizer(object):
    def __init__(
        self,
        sequence_length=30,
        score_threshold=0.25,
        stable_count=8,
        label_path='model/dynamic_gesture_classifier/dynamic_gesture_label.csv',
    ):
        self.sequence_length = sequence_length
        self.score_threshold = score_threshold
        self.stable_count = stable_count
        self.sequence = deque(maxlen=sequence_length)
        self.prediction_history = deque(maxlen=stable_count)
        self.labels = load_labels(label_path)
        self.classifier = DynamicGestureClassifier()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def recognize(self, frame):
        if frame is None:
            return dict(NO_RESULT)

        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.hands.process(rgb_image)
        rgb_image.flags.writeable = True

        landmarks = collect_landmarks(results)

        frame_features = make_frame_features(results)
        if frame_features is None:
            self.sequence.clear()
            self.prediction_history.clear()
            return self._no_result(landmarks)

        self.sequence.append(frame_features)

        if len(self.sequence) < self.sequence_length:
            return self._no_result(landmarks)

        label_id, score = self.classifier.predict(list(self.sequence))
        if score < self.score_threshold:
            self.prediction_history.clear()
            return self._no_result(landmarks)

        self.prediction_history.append(label_id)
        most_common_label_id, count = Counter(self.prediction_history).most_common(1)[0]
        if count < self.stable_count:
            return self._no_result(landmarks)

        label = (
            self.labels[most_common_label_id]
            if most_common_label_id < len(self.labels)
            else str(most_common_label_id)
        )

        return {
            "label_id": int(most_common_label_id),
            "label": label,
            "score": float(score),
            "type": "dynamic",
            "landmarks": landmarks,
        }

    @staticmethod
    def _no_result(landmarks):
        result = dict(NO_RESULT)
        result["landmarks"] = landmarks
        return result

    def close(self):
        self.hands.close()


_default_recognizer = None


def recognize(frame):
    global _default_recognizer
    if _default_recognizer is None:
        _default_recognizer = DynamicHandGestureRecognizer()
    return _default_recognizer.recognize(frame)

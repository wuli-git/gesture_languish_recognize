#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    OpResolverType = None


class DynamicGestureClassifier(object):
    def __init__(
        self,
        model_path='model/dynamic_gesture_classifier/dynamic_gesture_classifier.tflite',
        num_threads=1,
    ):
        interpreter_kwargs = {
            'model_path': model_path,
            'num_threads': num_threads,
        }
        if OpResolverType is not None:
            interpreter_kwargs['experimental_op_resolver_type'] = (
                OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )

        self.interpreter = Interpreter(**interpreter_kwargs)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, sequence):
        label_id, _ = self.predict(sequence)
        return label_id

    def predict(self, sequence):
        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_index,
            np.array([sequence], dtype=np.float32),
        )
        self.interpreter.invoke()

        output_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_index)
        scores = np.squeeze(result)
        label_id = int(np.argmax(scores))
        score = float(scores[label_id])

        return label_id, score

"""Shared numerically-stable softmax used by temperature-scaling calibration
tools (find_optimal_temperature.py, calibrate_by_group.py)."""

import numpy as np


def softmax(logits, axis=-1):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)

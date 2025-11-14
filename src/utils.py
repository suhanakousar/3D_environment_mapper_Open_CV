import yaml
import time
import numpy as np


def load_intrinsics(path: str):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    K = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1.0]])
    return d, K


class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = time.perf_counter()

    def elapsed_ms(self):
        return (time.perf_counter() - self.t0) * 1000.0

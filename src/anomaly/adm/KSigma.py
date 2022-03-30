from typing import Any

import numpy as np
from anomaly.base import Detector


class KSigma(Detector):
    """The naive detector detects all anomaly above a threshold"""

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.fitted = False

        self.ts: Any = None
        self.ts_predicted: Any = None
        self.mu = 0
        self.sigma = 0

    def fit(self, ts, ts_predicted):
        self.fitted = True
        self.ts = ts
        self.ts_predicted = ts_predicted
        data = ts - ts_predicted
        self.mu = data.mean()
        self.sigma = np.sqrt((data**2).mean() - data.mean() ** 2)

    def detect(self):
        assert self.fitted
        return np.abs(self.ts - self.ts_predicted - self.mu) > self.k * self.sigma

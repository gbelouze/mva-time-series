from typing import Any

import numpy as np
from anomaly.base import Detector, Filter


class NaiveDetector(Detector):
    """The naive detector detects all anomaly above a threshold"""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.fitted = False
        self.ts: Any = None
        self.ts_predicted: Any = None

    def fit(self, ts, ts_predicted):
        self.fitted = True
        self.ts = ts
        self.ts_predicted = ts_predicted

    def detect(self):
        assert self.fitted
        return np.abs(self.ts - self.ts_predicted) > self.threshold


class NaiveFilter(Filter):
    """The naive filter does not filter anything"""

    def __init__(self):
        self.fitted = False

    def fit(self, ts, ts_predicted, anomalies, anomalies_predicted):
        self.fitted = True

    def filter(self, ts, ts_predicted, anomalies_predicted):
        assert self.fitted
        return anomalies_predicted

import numpy as np
from anomaly.base import Detector, Filter


class NaiveDetector(Detector):
    """The naive detector detects all anomaly above a threshold"""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.fitted = False

    def fit(self, ts, ts_predicted):
        self.fitted = True

    def detect(self, ts, ts_predicted):
        assert self.fitted
        return np.abs(ts - ts_predicted) > self.threshold


class NaiveFilter(Filter):
    """The naive filter does not filter anything"""

    def __init__(self):
        self.fitted = False

    def fit(self, ts, ts_predicted, anomalies, anomalies_predicted):
        self.fitted = True

    def filter(self, ts, ts_predicted, anomalies_predicted):
        assert self.fitted
        return anomalies_predicted

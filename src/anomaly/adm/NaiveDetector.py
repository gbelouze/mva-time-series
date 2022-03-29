import numpy as np
from anomaly.base import Detector, Predictor


class NaiveDetector(Detector):
    """The naive detector detects all anomaly above a threshold"""

    def __init__(self, predictor: Predictor, threshold: float = 0.1) -> None:
        self._predictor = predictor
        self.threshold = threshold

    @property
    def predictor(self):
        return self._predictor

    def detect(self, ts):
        pred = self._predictor.predict(ts)
        return np.abs(ts - pred) > self.threshold

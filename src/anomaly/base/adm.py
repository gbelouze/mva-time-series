import abc

import numpy as np
from anomaly.base.tmm import Predictor


class Detector(abc.ABC):
    @property
    @abc.abstractmethod
    def predictor(self) -> Predictor:
        pass

    @abc.abstractmethod
    def detect(self, ts: np.ndarray):
        pass

import abc

import numpy as np
from anomaly.base.tmm import Predictor
from numpy.typing import NDArray


class Detector(abc.ABC):
    @property
    @abc.abstractmethod
    def predictor(self) -> Predictor:
        pass

    @abc.abstractmethod
    def detect(self, ts: NDArray[np.float64]) -> NDArray[np.bool_]:
        pass

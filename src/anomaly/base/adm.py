import abc

import numpy as np
from numpy.typing import NDArray


class Detector(abc.ABC):

    fitted: bool

    @abc.abstractmethod
    def fit(self, ts: NDArray[np.float64], ts_predicted: NDArray[np.float64]) -> None:
        pass

    @abc.abstractmethod
    def detect(self) -> NDArray[np.bool_]:
        """
        Notes
        -----
            Detector instance must be fitted first
        """
        pass


class Filter(abc.ABC):

    fitted: bool

    @abc.abstractmethod
    def fit(
        self,
        ts: NDArray[np.float64],
        ts_predicted: NDArray[np.float64],
        anomalies: NDArray[np.bool_],
        anomalies_predicted: NDArray[np.bool_],
    ) -> None:
        pass

    @abc.abstractmethod
    def filter(
        self, ts: NDArray[np.float64], ts_predicted: NDArray[np.float64], anomalies_predicted: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        """
        Notes
        -----
            Filter instance must be fitted first
        """
        pass

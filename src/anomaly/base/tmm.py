import abc

import numpy as np


class Predictor(abc.ABC):
    @property
    @abc.abstractmethod
    def bias(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def mad(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def mape(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def mse(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def sae(self) -> float:
        pass

    @abc.abstractmethod
    def fit(self, ts: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, ts: np.ndarray) -> np.ndarray:
        pass

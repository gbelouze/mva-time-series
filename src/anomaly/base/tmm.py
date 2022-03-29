import abc

import numpy as np
from numpy.typing import NDArray


class Predictor(abc.ABC):

    _bias: float
    _mad: float
    _mape: float
    _mse: float
    _sae: float

    fitted: bool

    @property
    def bias(self) -> float:
        r"""The arithmetic mean of the errors
        .. math:: b = \frac{1}{N}\sum_{t=1}^N \hat{S}_t - S_t
        """
        return self._bias

    @property
    def mad(self) -> float:
        r"""The mean absolute deviation
        .. math:: b = \frac{1}{N}\sum_{t=1}^N |\hat{S}_t - S_t|
        """
        return self._mad

    @property
    def mape(self) -> float:
        r"""The mean absolute percentage error
        .. math:: b = \frac{1}{N}\sum_{t=1}^N |\frac{\hat{S}_t - S_t}{S_t}|
        """
        return self._mape

    @property
    def mse(self) -> float:
        r"""The mean square error
        .. math:: b = \frac{1}{N}\sum_{t=1}^N (\hat{S}_t - S_t)^2
        """
        return self._mse

    @property
    def sae(self) -> float:
        r"""The sum of absolute errors
        .. math:: b = \sum_{t=1}^N |\hat{S}_t - S_t|
        """
        return self._sae

    def set_features(self, ts: NDArray[np.float64], predicted: NDArray[np.float64]) -> None:
        epsilon = 1e-6
        sum_err = 0.0
        sum_abs_err = 0.0
        sum_abs_err_relative = 0.0
        sum_err_squared = 0.0

        n = len(ts)
        for t, pred in zip(ts, predicted):
            error = pred - t
            sum_err += error
            sum_abs_err += np.abs(error)
            sum_abs_err_relative += np.abs(error) / max(np.abs(t), epsilon)
            sum_err_squared += error**2

        self._bias = sum_err / n
        self._mad = sum_abs_err / n
        self._mape = sum_abs_err_relative / n
        self._mse = sum_err_squared / n
        self._sae = sum_abs_err

    @abc.abstractmethod
    def fit(self, ts: NDArray[np.float64]) -> None:
        pass

    @abc.abstractmethod
    def predict(self, ts: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Notes
        -----
            Predictor instance must be fitted first
        """
        pass

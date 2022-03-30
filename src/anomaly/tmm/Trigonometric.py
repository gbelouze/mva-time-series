from typing import Any

import numpy as np
from anomaly.base import Predictor
from scipy.signal import argrelmax, periodogram


def get_largest_local_max(signal1D: np.ndarray, n_largest: int = 3, order: int = 1) -> [np.ndarray, np.ndarray]:
    """Return the largest local max and the associated index in a tuple.

    This function uses `order` points on each side to use for the comparison.
    """
    all_local_max_indexes = argrelmax(signal1D, order=order)[0]
    all_local_max = np.take(signal1D, all_local_max_indexes)
    largest_local_max_indexes = all_local_max_indexes[all_local_max.argsort()[::-1]][:n_largest]

    return (
        np.take(signal1D, largest_local_max_indexes),
        largest_local_max_indexes,
    )


class Trigonometric(Predictor):
    r"""The trigonometric predictor fits a trigonometric regression"""

    def __init__(self) -> None:
        self._bias = -1
        self._mad = -1
        self._mape = -1
        self._mse = -1
        self._sae = -1
        self.fitted = False
        self.ts: Any = None
        self.ts_predicted: Any = None

    def fit(self, ts):
        self.fitted = True
        self.ts = ts

        time = np.arange(len(ts))
        # compute the periodogram
        freqs, Pxx_spec = periodogram(
            x=ts,
        )
        spectral_density = np.sqrt(Pxx_spec)

        # find the main frequencies
        values, (f_1_ind, f_2_ind) = get_largest_local_max(spectral_density, n_largest=2)
        (f_1, f_2) = np.take(freqs, (f_1_ind, f_2_ind))

        regressors = [np.ones(ts.shape)]
        for k in range(1, 5):
            regressors += [
                np.cos(2 * np.pi * f_1 * k * time),
                np.sin(2 * np.pi * f_1 * k * time),
                np.cos(2 * np.pi * f_2 * k * time),
                np.sin(2 * np.pi * f_2 * k * time),
            ]
        regressors = np.c_[regressors]
        beta, *_ = np.linalg.lstsq(regressors.T, ts, rcond=None)
        self.ts_predicted = regressors.T @ beta

    def predict(self, start=0, end=None):
        assert self.fitted
        if end is None:
            end = len(self.ts)

        if end > len(self.ts) or start > len(self.ts):
            raise ValueError("Naive model cannot do forecasting")
        return self.ts_predicted[start:end]

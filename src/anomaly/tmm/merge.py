from typing import Any, List

import numpy as np
from anomaly.base import Predictor


class Sequential(Predictor):
    r"""The sequential predictor applies sequentially predictors on the remaining residuals
    .. math:: \hat{S}_t = \phi_p \circ \ldots \circ \phi_1 S_{t}
    """

    def __init__(self, predictors: List[Predictor]) -> None:
        self._bias = -1
        self._mad = -1
        self._mape = -1
        self._mse = -1
        self._sae = -1
        self.fitted = False
        self.predictors = predictors
        self.ts: Any = None
        self.ts_predicted: Any = None

    def fit(self, ts):
        self.fitted = True
        self.ts = ts

        for predictor in self.predictors:
            predictor.fit(ts)
            ts -= predictor.predict()
        self.ts_predicted = self.ts - ts

    def predict(self, start=0, end=None):
        assert self.fitted
        if end is None:
            end = len(self.ts)

        if end > len(self.ts) or start > len(self.ts):
            raise ValueError("Cannot do forecasting")
        return self.ts_predicted[start:end]


class Best:
    r"""The best predictor retains the prediction from several predictors that gives the smallest residual
    .. math:: \hat{S}_t = S_t + \min_{||\cdot||}(\phi_1(S_t) - S_t, \ldots, \phi_p(S_t) - S_t)
    """

    def __init__(self, predictors: List[Predictor]) -> None:
        self._bias = -1
        self._mad = -1
        self._mape = -1
        self._mse = -1
        self._sae = -1
        self.fitted = False
        self.predictors = predictors
        self.ts: Any = None
        self.ts_predicted: Any = None

    def fit(self, ts):
        self.fitted = True
        self.ts = ts

        for predictor in self.predictors:
            predictor.fit(ts)

    def predict(self, start=0, end=None):
        assert self.fitted
        if end is None:
            end = len(self.ts)

        if end > len(self.ts) or start > len(self.ts):
            raise ValueError("Cannot do forecasting")

        predictions = np.stack([predictor.predict(start, end) for predictor in self.predictors], axis=1)
        best = np.argmin(np.abs(predictions - self.ts[start:end, None]), axis=1)
        return predictions[np.arange(len(predictions)), best]

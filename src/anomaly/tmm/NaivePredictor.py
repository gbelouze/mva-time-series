from typing import Any

from anomaly.base import Predictor


class NaivePredictor(Predictor):
    r"""The naive predictor predicts the last value
    .. math:: \hat{S}_t = S_{t-1}
    """

    def __init__(self) -> None:
        self._bias = -1
        self._mad = -1
        self._mape = -1
        self._mse = -1
        self._sae = -1
        self.fitted = False
        self.ts: Any = None

    def fit(self, ts):
        self.fitted = True
        self.ts = ts
        self.ts_predicted = ts.copy()
        self.ts_predicted[1:] = ts[:-1]

    def predict(self, start=0, end=None):
        assert self.fitted
        if end is None:
            end = len(self.ts)

        if end > len(self.ts) or start > len(self.ts):
            raise ValueError("Naive model cannot do forecasting")
        return self.ts_predicted[start:end]


if __name__ == "__main__":
    # test that the class implements all abstract methods
    naive = NaivePredictor()

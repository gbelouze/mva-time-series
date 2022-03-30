from typing import Any

import numpy as np
from anomaly.base import Predictor


class Polynomial(Predictor):
    """The polynomial predictor fits a polynomial regression"""

    def __init__(self, degree: int = 5) -> None:
        self._bias = -1
        self._mad = -1
        self._mape = -1
        self._mse = -1
        self._sae = -1
        self.fitted = False

        self.degree = degree
        self.poly: Any = None
        self.ts: Any = None

    def fit(self, ts):
        self.fitted = True
        self.ts = ts
        self.poly = np.polynomial.Polynomial.fit(np.arange(len(ts)), ts, self.degree)  # type: ignore

    def predict(self, start=0, end=None):
        assert self.fitted
        if end is None:
            end = len(self.ts)
        return self.poly(np.arange(start, end))


if __name__ == "__main__":
    # test that the class implements all abstract methods
    arima = Polynomial(degree=2)

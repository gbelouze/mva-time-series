from typing import Any, Tuple

import statsmodels.tsa.arima.model as model  # type: ignore
from anomaly.base import Predictor


class ARIMA(Predictor):
    """Autoregressive model"""

    def __init__(self, order: Tuple[int, int, int]) -> None:
        self._bias = -1
        self._mad = -1
        self._mape = -1
        self._mse = -1
        self._sae = -1
        self.fitted = False

        self.arma: Any = None
        self.order = order
        self.ts: Any = None

    def fit(self, ts):
        self.fitted = True
        self.ts = ts
        self.arma = model.ARIMA(ts, order=self.order, enforce_invertibility=False, enforce_stationarity=False).fit()

    def predict(self, start=0, end=None):
        assert self.fitted
        if end is None:
            end = len(self.ts)

        return self.arma.predict(start=start, end=(end - 1))


class AR(ARIMA):
    def __init__(self, order: int = 5) -> None:
        super().__init__(order=(order, 0, 0))


class MA(ARIMA):
    def __init__(self, order: int = 5) -> None:
        super().__init__(order=(0, 0, order))


class ARMA(ARIMA):
    def __init__(self, order_ar: int = 5, order_ma: int = 5) -> None:
        super().__init__(order=(order_ar, 0, order_ma))


if __name__ == "__main__":
    # test that the class implements all abstract methods
    arima = ARIMA(order=(2, 0, 2))

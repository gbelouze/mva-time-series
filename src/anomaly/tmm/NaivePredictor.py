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

    def fit(self, ts):
        self.fitted = True

    def predict(self, ts):
        assert self.fitted
        ts[1:] = ts[:-1]
        return ts


if __name__ == "__main__":
    # test that the class implements all abstract methods
    naive = NaivePredictor()

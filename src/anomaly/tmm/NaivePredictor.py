from anomaly.base.tmm import Predictor


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

    def fit(self, ts):
        pass

    def predict(self, ts):
        ts[1:] = ts[:-1]
        return ts


if __name__ == "__main__":
    # test that the class implements all abstract methods
    naive = NaivePredictor()

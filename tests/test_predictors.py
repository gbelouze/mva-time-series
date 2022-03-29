import anomaly.tmm as tmm
import numpy as np

naive_predictor = tmm.NaivePredictor()


def _test_fit(predictor):
    ts = np.random.random(100)
    predictor.fit(ts)


def _test_predict(predictor):
    ts = np.random.random(50)
    pred = predictor.predict(ts)
    assert ts.shape == pred.shape


def test_fit_naive():
    _test_fit(naive_predictor)


def test_predict_naive():
    _test_predict(naive_predictor)

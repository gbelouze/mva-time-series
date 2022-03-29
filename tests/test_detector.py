import numpy as np
from anomaly import adm

naive_detector = adm.NaiveDetector()


def _test_fit(detector):
    ts = np.random.random(100)
    ts_predicted = np.random.random(100)
    detector.fit(ts, ts_predicted)


def _test_detect(detector):
    ts = np.random.random(50)
    ts_predicted = np.random.random(50)
    anomalies = detector.detect(ts, ts_predicted)
    assert anomalies.shape == ts.shape


def test_fit_naive():
    _test_fit(naive_detector)


def test_predict_naive():
    _test_detect(naive_detector)

import numpy as np
from anomaly import adm

naive_filter = adm.NaiveFilter()


def _test_fit(filter):
    ts = np.random.random(100)
    ts_predicted = np.random.random(100)
    anomalies = np.random.random(100) > 0.9
    anomalies_predicted = anomalies & (np.random.random(100) > 0.9)
    filter.fit(ts, ts_predicted, anomalies, anomalies_predicted)


def _test_filter(filter):
    ts = np.random.random(50)
    ts_predicted = np.random.random(50)
    anomalies_predicted = np.random.random(50) > 0.9
    anomalies_filtered = filter.filter(ts, ts_predicted, anomalies_predicted)
    assert anomalies_filtered.shape == ts.shape


def test_fit_naive():
    _test_fit(naive_filter)


def test_predict_naive():
    _test_filter(naive_filter)

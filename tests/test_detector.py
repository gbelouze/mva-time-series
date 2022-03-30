import numpy as np
from anomaly import adm, io, tmm

naive_detector = adm.NaiveDetector()
ksigma_detector = adm.KSigma()


def _test_fit(detector):
    ts = np.array(io.read(1, 1).value)

    naive_predictor = tmm.NaivePredictor()
    naive_predictor.fit(ts)
    ts_predicted = naive_predictor.predict()

    detector.fit(ts, ts_predicted)


def _test_detect(detector):
    detector.detect()


def test_fit_naive():
    _test_fit(naive_detector)


def test_predict_naive():
    _test_detect(naive_detector)


def test_fit_ksigma():
    _test_fit(ksigma_detector)


def test_predict_ksigma():
    _test_detect(ksigma_detector)

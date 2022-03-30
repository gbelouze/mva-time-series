import numpy as np
from anomaly import io, tmm

naive_predictor = tmm.NaivePredictor()
ar_predictor = tmm.AR()
ma_predictor = tmm.MA()
arma_predictor = tmm.ARMA()
poly_predictor = tmm.Polynomial()
trigo_predictor = tmm.Trigonometric()
sequential_predictor = tmm.Sequential([poly_predictor, arma_predictor])
best_predictor = tmm.Best([poly_predictor, arma_predictor, naive_predictor])


def _test_fit(predictor):
    ts = np.array(io.read(1, 1).value)
    predictor.fit(ts)


def _test_predict(predictor):
    pred = predictor.predict(start=25, end=75)
    assert pred.shape == (50,)


def test_fit_naive():
    _test_fit(naive_predictor)


def test_predict_naive():
    _test_predict(naive_predictor)


def test_fit_ar():
    _test_fit(ar_predictor)


def test_predict_ar():
    _test_predict(ar_predictor)


def test_fit_ma():
    _test_fit(ma_predictor)


def test_predict_ma():
    _test_predict(ma_predictor)


def test_fit_arma():
    _test_fit(arma_predictor)


def test_predict_arma():
    _test_predict(arma_predictor)


def test_fit_poly():
    _test_fit(poly_predictor)


def test_predict_poly():
    _test_predict(poly_predictor)


def test_fit_trigo():
    _test_fit(trigo_predictor)


def test_predict_trigo():
    _test_predict(trigo_predictor)


def test_fit_sequential():
    _test_fit(sequential_predictor)


def test_predict_sequential():
    _test_predict(sequential_predictor)


def test_fit_best():
    _test_fit(best_predictor)


def test_predict_best():
    _test_predict(best_predictor)

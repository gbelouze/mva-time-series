from anomaly import io, tmm

naive_predictor = tmm.NaivePredictor()
ar_predictor = tmm.AR()
ma_predictor = tmm.MA()
arma_predictor = tmm.ARMA()
poly_predictor = tmm.Polynomial()


def _test_fit(predictor):
    ts = io.read(1, 1).value
    predictor.fit(ts)


def _test_predict(predictor):
    pred = predictor.predict(start=25, end=75)
    assert pred.shape == (50,)


def test_fit_naive():
    _test_fit(naive_predictor)


def test_predict_naive():
    _test_predict(naive_predictor)

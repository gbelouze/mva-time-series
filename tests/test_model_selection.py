from anomaly import io, tmm, adm
import pandas as pd

import anomaly.utils.statsutils as su


bench = io.BenchmarkDataset(1)
print(bench.read(1).columns)

bench = io.BenchmarkDataset(3)
print(bench.read(1).columns)

naive_predictor = tmm.NaivePredictor()
ar_predictor = tmm.AR()
ma_predictor = tmm.MA()
arma_predictor = tmm.ARMA()
poly_predictor = tmm.Polynomial()



predictor = tmm.ARMA()
detector = adm.KSigma()
data = io.read(2, 1)
ts = data.value
predictor.fit(ts)

ts_predicted = predictor.predict()
detector.fit(ts, ts_predicted)
predicted_anomalies = detector.detect()



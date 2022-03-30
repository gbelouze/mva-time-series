import numpy as np
import pandas as pd
import tqdm
from anomaly import adm, io
from sklearn.metrics import f1_score


def compute_predictor_scores(predictor_dict, benchmark_index, detector=adm.KSigma()):
    bench = io.BenchmarkDataset(benchmark_index)

    score_names = ["bias", "mad", "mape", "mse", "sae", "f1"]
    score_dict_np = dict.fromkeys(predictor_dict.keys(), np.empty((bench.len, len(score_names))))

    for i in tqdm.trange(bench.len):
        df = bench.read(i)
        ts = df.value
        ts_label = df.is_anomaly

        for predictor_name, predictor in predictor_dict.items():
            predictor.fit(ts)
            ts_predicted = predictor.predict(end=len(ts))

            predictor.set_features(ts, ts_predicted)

            detector.fit(ts, ts_predicted)
            predicted_anomalies = detector.detect()
            f1 = f1_score(ts_label, predicted_anomalies)

            score_dict_np[predictor_name][i] = np.array(
                [predictor.bias, predictor.mad, predictor.mape, predictor.mse, predictor.sae, f1]
            )

    score_dict_np = dict.fromkeys(predictor_dict.keys(), np.empty(bench.len, len(score_names)))

    score_dict = {
        predictor_name: pd.DataFrame(data=scores, columns=score_names)
        for predictor_name, scores in score_dict_np.items()
    }

    return score_dict

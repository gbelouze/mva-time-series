import sys

import anomaly.utils.statsutils as su
import numpy as np
import pandas as pd
import tqdm
from anomaly import adm, io, tmm  # noqa: F401
from sklearn.metrics import f1_score, recall_score


def compute_predictor_scores(predictor_dict, bench, detector=adm.KSigma()):

    score_names = ["bias", "mad", "mape", "mse", "sae", "f1", "recall"]
    score_dict_np = {k: np.empty((len(bench), len(score_names))) for k in predictor_dict}

    for i in tqdm.trange(len(bench), file=sys.stdout):
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
            recall = recall_score(ts_label, predicted_anomalies)
            scores = np.array([predictor.bias, predictor.mad, predictor.mape, predictor.mse, predictor.sae, f1, recall])

            score_dict_np[predictor_name][i] = scores

    score_dict = {
        predictor_name: pd.DataFrame(data=scores, columns=score_names)
        for predictor_name, scores in score_dict_np.items()
    }

    return score_dict


def compute_benchmark_features(bench):
    feature_names = su.TS_Features.list_features()

    features_np = np.empty((bench.len, len(feature_names)))

    for i in tqdm.trange(bench.len, file=sys.stdout):
        df = bench.read(i)
        ts = df.value

        features_np[i] = su.TS_Features(ts).features

    features = pd.DataFrame(data=features_np, columns=feature_names)
    return features

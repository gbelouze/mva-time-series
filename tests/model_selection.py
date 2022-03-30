import anomaly.utils.modelselect_utils as mu
import anomaly.utils.statsutils as su  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from anomaly import adm, io, tmm  # noqa: F401
from sklearn.metrics import f1_score  # noqa: F401

predictor_dict = {
    "naive_predictor": tmm.NaivePredictor(),
    "ar_predictor": tmm.AR(),
    "ma_predictor": tmm.MA(),
    "arma_predictor": tmm.ARMA(),
    "poly_predictor": tmm.Polynomial(),
}

score_dict = mu.compute_predictor_scores(predictor_dict, io.BenchmarkDataset(1), detector=adm.KSigma())

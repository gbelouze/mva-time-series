from anomaly import io, tmm, adm
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

import anomaly.utils.modelselect_utils as mu

import anomaly.utils.statsutils as su


predictor_dict = {
"naive_predictor" : tmm.NaivePredictor(),
"ar_predictor" : tmm.AR(),
"ma_predictor" : tmm.MA(),
"arma_predictor" : tmm.ARMA(),
"poly_predictor" : tmm.Polynomial(),
}

score_dict = mu.compute_predictor_scores(predictor_dict, 1, detector=adm.KSigma())



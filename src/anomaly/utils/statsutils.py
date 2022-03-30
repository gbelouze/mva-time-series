import hurst
import nolds
import numpy as np
import scipy
from patsy import dmatrix
from statsmodels.api import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

EPSILON = 1e-9
"""
compute the time series features as described in

"Rule induction for forecasting method selection: meta-learning the
characteristics of univariate time series", Xiaozhe Wang, Kate Smith-Miles, Rob Hyndman

These features will be used for model selection in step 1/ and anomaly filtering in step 3/.
"""


class TS_Features:
    def __init__(self, ts):
        trend, seasonality, self.periodicity = decompose_trend_and_seasonality(ts)
        self.trend_score, self.seasonality_score = compute_trend_and_seasonality_score(ts, trend, seasonality)
        self.nonlinearity = compute_nonlinearity(ts)
        self.skew = compute_skew(ts)
        self.kurtosis = compute_kurtosis(ts)
        self.lyapunov = compute_lyapunov(ts)

        # TODO: should be normalized to [0,1]?
        self.features = np.array(
            [
                self.periodicity,
                self.trend_score,
                self.seasonality_score,
                self.nonlinearity,
                self.skew,
                self.kurtosis,
                self.lyapunov,
            ]
        )


def spline_regression(ts, n_knots=3):
    n = len(ts)
    knots = [n // (n_knots + 2) * i for i in range(1, n_knots + 1)]
    knots_s = "(" + ",".join((str(knot) for knot in knots)) + ")"
    basis_x = dmatrix(
        f"bs(timestamp, knots={knots_s}, degree=3, include_intercept=False)",
        {"timestamp": np.arange(n)},
        return_type="dataframe",
    )
    fit = GLM(ts, basis_x).fit()

    predict = fit.predict(
        dmatrix(
            f"bs(timestamp, knots={knots_s}, include_intercept=False)",
            {"timestamp": np.arange(n)},
            return_type="dataframe",
        )
    )

    return predict


def compute_periodicity(ts_value_detrend):
    autocor_ts = acf(ts_value_detrend, nlags=len(ts_value_detrend) // 3)
    peaks, _ = scipy.signal.find_peaks(autocor_ts)
    troughs, _ = scipy.signal.find_peaks(-autocor_ts)

    i_trough = 0
    min_val_trough = autocor_ts[troughs[i_trough]]

    for peak in peaks:
        # peak should have positive autocorrelation
        if ts_value_detrend[peak] <= 0:
            continue

        # there should be a trough before the peak
        if troughs[0] > peak:
            continue

        # we must have peak - trough > 0.1
        while troughs[i_trough + 1] < peak:
            i_trough += 1
            min_val_trough = min(min_val_trough, autocor_ts[troughs[i_trough]])

        if autocor_ts[peak] - min_val_trough < 0.1:
            continue

        return peak

    return 1


def decompose_trend_and_seasonality(ts):
    trend = spline_regression(ts)

    ts_value_detrend = ts - trend
    periodicity = compute_periodicity(ts_value_detrend)

    if periodicity == 1:
        trend, seasonality = trend, None

    else:
        if periodicity % 2 == 0:
            periodicity += 1

        stl = STL(ts, period=periodicity)
        res = stl.fit()

        trend, seasonality = res.trend, res.seasonal

    return trend, seasonality, periodicity


def compute_trend_and_seasonality_score(ts, trend, seasonality):
    if seasonality is None:
        trend_score = 1.0
        seasonality_score = 0.0
    else:
        v_Y = np.var(ts)
        v_X = np.var(ts - trend)
        v_Z = np.var(ts - seasonality)

        trend_score = 1 - v_Y / (v_Z + EPSILON)
        seasonality_score = 1 - v_Y / (v_X + EPSILON)

    return trend_score, seasonality_score


def compute_nonlinearity(ts):
    res = OLS(ts, np.arange(len(ts))).fit()
    F, p = linear_rainbow(res)
    return F


def compute_skew(ts):
    return scipy.stats.skew(ts)


def compute_kurtosis(ts):
    return scipy.stats.kurtosis(ts)


def compute_hurst(ts):
    H, c, val = hurst.compute_Hc(ts)
    return H


def compute_lyapunov(ts):
    return nolds.lyap_r(ts)

"""The Timeseries Modeling Module"""

from .ARIMA import AR as AR  # noqa
from .ARIMA import ARMA as ARMA  # noqa
from .ARIMA import MA as MA  # noqa
from .merge import Best as Best  # noqa
from .merge import Sequential as Sequential  # noqa
from .NaivePredictor import NaivePredictor as NaivePredictor  # noqa
from .Polynomial import Polynomial as Polynomial  # noqa
from .Trigonometric import Trigonometric as Trigonometric  # noqa

all_predictors = {
    "ar": AR,
    "ma": MA,
    "arma": ARMA,
    "naive": NaivePredictor,
    "polynomial": Polynomial,
    "trigonometric": Trigonometric,
    "poly+arma": lambda: Sequential(predictors=[Polynomial(), ARMA()]),
    "poly+trigo+arma": lambda: Sequential(predictors=[Polynomial(), Trigonometric(), ARMA()]),
}

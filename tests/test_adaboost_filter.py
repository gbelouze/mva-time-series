import numpy as np

import anomaly.utils.statsutils as su


ts = np.random.randn(100)

ts_ft = su.TS_Features(ts)


print(ts_ft.features)
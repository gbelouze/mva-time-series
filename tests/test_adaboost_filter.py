import anomaly.utils.statsutils as su
import numpy as np

ts = np.random.randn(100)

ts_ft = su.TS_Features(ts)


print(ts_ft.features)

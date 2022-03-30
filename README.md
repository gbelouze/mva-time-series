# Automated Time-Series Anomaly Detection 

Code for the 2021 MVA course 'ML for time series'

## Installation

To install, run

```
$ pip install -e .
```

To install in a `venv` (in particular if the default `pip` is too old):

```bash
python3.9 -m venv ./venv/
source ./venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Codebase description

The `anomaly` module is implemented in `src/anomaly`:

- `base/` specifies abstract modules which will be implemented in the following modules
- `tmm/` is the time-series modelisation module, which implements models (ARIMA, polynomial, naive...) to fit the data
- `adm/` contains the anomaly detection module which computes the anomalies from the residuals of the data 
    - `Naive.py` is a simple threshold over the residuals
    - `Ksigma.py` is an adaptive threshold
- `utils/` contains utility functions
    - `statsutils.py` implements the functions to compute the features from the time-series
    - `modelselect_utils.py` automates the computation of the features and of the scores of a model on a benchmark dataset
- `io.py` provides convenience functions to read the benchmark datasets
- `tests/` contains scripts to check if the code is working correctly

The `data/` folder contains the datasets provided by the author (we give the datasets here rather than a link to download them, as downloading them requires approval from Yahoo which takes several days).


The `notebooks/` folder containes a Jupyter Notebook to run the experiments. Computing the anomaly detection on the datasets is fairly long (~15min per dataset) so we have saved some results in `notebooks/saved_data/` which can be used directly in the notebook.

# Copyright (c) 2024 Jacopo Ventura

import numpy as np


def hurst(ts: np.array, max_lag: int) -> float:
    """
    Returns the Hurst Exponent of the time series vector ts. The time series is in original scale.
    :param ts: input time series in original scale
    :param max_lag: maximum value of the time lag variable
    :return hurst coefficient
    """
    # Transform original time series into log time series
    log_ts = np.log(ts)

    # create a set of possible time lag values (independent variable)
    lags = range(2, max_lag)
    # lags = range(2, int(len(ts)*0.5))

    # Calculate the array of the variances of the lagged differences
    tau = [np.var(np.subtract(log_ts[lag:], log_ts[:-lag])) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] / 2.0

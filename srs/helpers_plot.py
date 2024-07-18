# Copyright (c) 2024 Jacopo Ventura

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

def plot_price_series(ts1: pd.DataFrame, ts2: pd.DataFrame,
                      residuals: np.array, predictions: np.array,
                      name_ts1: str, name_ts2: str) -> None:
    """
    Plot data for cointegrated augmented Dickey-Fuller test
    :param ts1: first time series (x-axis)
    :param ts2: second time series (y-axis
    :param residuals: residuals
    :param predictions: predictions
    :param name_ts1: name of first time series
    :param name_ts2: name of second time series
    """
    months = mdates.MonthLocator()

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(ts1.index, ts1, label=name_ts1)
    axs[0].plot(ts2.index, ts2, label=name_ts2)
    axs[0].xaxis.set_major_locator(months)
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs[0].grid(True)
    axs[0].set(xlabel="Month/Year", ylabel="Price ($)",
               title="%s and %s Daily Prices" % (name_ts1, name_ts2))
    axs[0].legend()

    # Scatter plot
    axs[1].scatter(ts1, ts2)
    axs[1].plot(ts1, predictions, color="k")
    axs[1].set(xlabel="%s Price($)" % name_ts1, ylabel="%s Price($)" % name_ts2,
                  title="%Scatterplot")

    # Plot of residuals
    axs[2].plot(ts1.index, residuals, label=name_ts1)
    axs[2].xaxis.set_major_locator(months)
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs[2].set(xlabel="Month/Year", ylabel="%s Price($)" % name_ts2,
                  title="Residuals")
    plt.show()


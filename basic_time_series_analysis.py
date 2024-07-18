from numpy import cumsum
from numpy.random import randn
import datetime
import yfinance as yf
import statsmodels.tsa.stattools as ts
import numpy as np
from sklearn.linear_model import LinearRegression

from srs.helpers_quant import hurst
from srs.helpers_plot import plot_price_series

# ======================================= TIME SERIES ANALYSIS ===============================

# Mean reverting: hurst exponent
MAX_LAG = 300

# Create a Geometric Brownian Motion, Mean-Reverting and Trending Series
gbm = cumsum(randn(100000))+1000
mr = randn(100000)+1000
tr = cumsum(randn(100000)+1)+1000
print(f"Hurst of brownian {hurst(gbm, MAX_LAG)}")
print(f"Hurst of mean reverting {hurst(mr, MAX_LAG)}")
print(f"Hurst of trending {hurst(tr, MAX_LAG)}")

# Test with SPY
spy = yf.Ticker("SPY")
spy = spy.history(start=datetime.datetime(2022, 10, 1), interval="1d")
hurst_spy = hurst(spy["Close"].values, MAX_LAG)
print(f"Hurst of SPY: {hurst_spy}")
x_spy = np.array(range(len(spy["Close"].values)))
model = LinearRegression().fit(x_spy.reshape((-1, 1)), spy["Close"].values)
beta = model.coef_[0]
intercept = model.intercept_
predictions = [beta * x + intercept for x in x_spy]
residuals = [spy["Close"].values[i] - predictions[i] for i in range(len(predictions))]
variance = np.var(np.abs(residuals))
print(f"Regression on SPY: beta {beta}, variance {variance}")


# Cointegration
START_DATE = datetime.datetime(2022, 10, 1)
END_DATE = datetime.datetime.now().date() - datetime.timedelta(1)
googl = yf.Ticker("GOOGL")
googl = googl.history(start=START_DATE, end=END_DATE, interval="1d")
googl_ts = googl["Close"]
msft = yf.Ticker("MSFT")
msft = msft.history(start=START_DATE, end=END_DATE, interval="1d")
msft_ts = msft["Close"]

# model = sm.OLS(googl_ts, msft_ts)
model = LinearRegression().fit(googl_ts.values.reshape((-1, 1)).copy(), msft_ts.values)
beta = model.coef_[0]

intercept = model.intercept_
predictions = [beta * googl_ts.values[i] + intercept for i in range(len(msft_ts))]
residuals = [msft_ts.values[i] - predictions[i] for i in range(len(msft_ts))]
variance = np.var(np.abs(residuals))

# Calculate and output the CADF test on the residuals
# H0: residuals is a random time series
cadf = ts.adfuller(residuals)
pval = cadf[0]
significance_value = cadf[4]["5%"]
print(f"CADF test results: \n")
if pval < significance_value:
    print("Reject H0: the cointegration of the price histories is mean reverting")
else:
    print("Fail to reject H0: the cointegration of the price histories is random")


plot_price_series(googl_ts, msft_ts, residuals, predictions,"GOOGL", "MSFT")

a = 1
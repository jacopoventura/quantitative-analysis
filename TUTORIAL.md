# Tutorial: algorithmic trading

## Copyright (c) 2024 Jacopo Ventura

This document is a detailed tutorial on how to develop an algorithmic trading system.


### General info on algo trading
#### Advantages of algo trading
1. backtesting: automated strategies can be backtested in long time periods
2. execution without human discretionary
3. high frequency

The core part of algorithmic trading is the development of the strategy, that must be identified, modelled and optimized. These three steps are 
performed via backtesting.

#### Issues of backtesting
1. Biases:
   1. optimization bias: the strategy overfits the data used for backtesting, and performs poorly live. Solutions: reduction of the number of 
         strategy parameters and extension of the data time window for backtesting.
   2. look-ahead bias: future data are used in the strategy. Example: signal using the close of the current candle. This is usually caused by a 
      bug or misuse of the data.
   3. cognitive bias: the backtest shows important drawdowns in the equity curve. When the system goes live, the trader gets scared at the first 
      drawdown and stops the system.
2. Exchange issues:
   1. limit and market orders are never executed exactly at the order price. These orders shall be modelled in the backtesting correctly.
   2. dataset precision: free datasets like the one from yahoo might have incorrect data in few candles.
   3. shorting limitations: in some periods (like 2008), shorting was blocked.
   4. fees and commissions have a huge impact, thus must be modelled in the backtest
   5. slippage: difference between the order price and execution price (shall be modelled in backtesting), due to the time latency between the 
      algo trading signal, the reception of this signal by the broker and the actual order execution by the broker.


### Hypothesis testing
We use hypothesis testing to verify a claim on a population. 
Key definitions:
1. alternative hypothesis (H1): claim we want to verify
2. null hypothesis (H0): the opposite of the claim (alternative hypothesis)
3. the statistical test aims at verifying whether we can REJECT or FAIL TO REJECT the null hypothesis
4. if the null hypothesis is rejected, we can accept the alternative hypothesis (our claim)
5. p-value: likelihood of obtaining a sample as extreme as the one we’ve obtained, if the null hypothesis is true. Thus, a small p-value means 
   that the observed population under the null-hypothesis is a rare (extreme) event due to casuality (i.e. very unlikely under H0). Consequently, 
   we can reject H0 and accept H1
6. p > a (0.05): fail to reject H0; p < a (0.05): reject H0 and accept H1
7. p-value of a one-sided hypothesis test is half the p-value of a two-sided hypothesis test.


### Financial data storage
There are three ways to store financial data:
1. flat-file data (like CSV): format provided by the data vendor. Easy to use and compress, but lack of querying capabilities.
2. document stores / NoSQL: storage of collections and documents, without a table schema.
3. relational database management system (RDMS): database composed by connected tables. Usage of SQL for complex queries. Change of table requires 
   some work.
Due to the simplicity of usage, we use flat-file data.

### Getting the price history database
Data have been purchased from: .com
Another source of historic data (not only equities, but also economic data like CPI) is data.nasdaq.com. This service is not used in this project.
Once the price history is obtained, it must be cleaned. For example, spikes (especially outside the trading session) shall be cleaned and futures 
contracts shall be adjusted at expirations (back/forward or proportional adjustment, rollover). Since the goal is to trade only intraday signals 
(no swing), these adjustments are automatically performed by yfinance by setting the following flags: back_adjust=False/True.


### Statistical Learning
Given the vector of features X = (x_1, x_2, ..., x_p), the response Y can be estimated from X with 

Y = f(X) + e

where f() is the unknown function and e the error (zero mean noise).

In Statistical Learning we must isolate the following concepts:
1. prediction: given a newly observed vector X, we estimate Y using the known function f()
2. inference: given a set of observations (X_i, Y_i), we model their relationship


### Time series analysis
In quant finance, continuous time series models are used. 
One of the most important and used key concepts in time series analysis is mean reversion: the price moves proportionally to the difference with 
the mean price and tends to returns to the mean value. This concepts is the opposite of the random walk model (each step is random and not related 
to the previous), which can be considered the H0 hypotheses.
The model of a continuous mean-reverting time series is called Ornstein-Uhlenbeck:

dx_t = theta * (x_t - u) * dt + sigma * dW_t

where x_t is the current price (at time t), u is the mean, theta is the scaling factor, sigma the variance and W_t is the brownian motion (adds a 
random component). This equation states that the change of the price series in the next continuous time is proportional to the difference 
between the mean price and the current price, with the addition of Gaussian noise.

#### Augmented Dickey-Fuller (ADF) Test
The ADF tests uses the following model for a mean reverting time series: dy_t = a + b * t + g * y_(t-1) + d * d(t-1) ... + e
The null hypothesis is that the time series is not mean reverting, rather casual, thus a=b=g=d=..=0.
The test is performed through statsmodels.tsa.stattools.adfuller(amzn[’Adj Close’], 1)

#### Hurst exponent
We define the variance of the logarithmic price over a time window T as:
var(T) = ⟨ | log(t + T) − log(t) |^2 ⟩ 
where T is the independent variable. Then, we check how the function var(T) is proportional to T^2H. The estimation of the 
Hurst exponent H provides the 
type 
of time series:
- H ≈ 0.5 -> Brownian motion
- H < 0.5 -> mean reverting (≈0: strong mean reverting)
- H > 0.5 -> trending (≈1: strong trending)

In practice, given a time series ts:
1. we compute the log price
2. we generate the independent variable T as 2:T_max, where T_max depends on the time length of the time series. 
3. for each value of T, we first calculate the series [log(t + T) − log(t)], then its variance
4. we apply the log property to estimate H: log(var(T)) = log(T^2h) = 2H log(T), meaning log(var(T)) / log(T) = 2H constant, thus (log(T), log(var
   (T)) is a linear function
5. we linearly fit the curve (log(T), log(var(T)) and we get H = slope / 2

#### Cointegrated Augmented Dickey-Fuller (CADF) Test
The previous two tests (ADF and Hurst exponent) test a single time series only. In the praxis, it is rare to find a price history being mean 
reverting.
For this reason, a new strategy consists of selecting two stocks of the same industry, like MSFT and GOOGL belonging to NASDAQ / Tech.
In this approach, we plot one price history against each other (same day data build one point in the plot). Then we obtain the linear 
interpolation curve and then we build the residuals. 
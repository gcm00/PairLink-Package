import numpy as np
import pandas as pd
import statsmodels.api as sm
from .processing import _transform_series_pct_change


def estimate_half_life(y: pd.Series, x: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion of the spread between two price series (classic method).

    Parameters
    ----------
    y : pd.Series
        First price series.
    x : pd.Series
        Second price series.

    Returns
    -------
    float
        Estimated half-life (in days). Returns np.inf if spread is not mean-reverting.
    """

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    beta = model.params[1]
    print(beta)                 # we ll see by printing it if it is the same than the one we found in 'cointegration.py'
    alpha = model.params[0]

    spread = y - (alpha + beta * x)
    spread_lag = spread.shift(1)
    delta_spread = spread - spread_lag

    valid = spread_lag.notna() & delta_spread.notna()
    spread_lag = spread_lag[valid]
    delta_spread = delta_spread[valid]

    x_lag = sm.add_constant(spread_lag)
    model_ar1 = sm.OLS(delta_spread, x_lag).fit()
    lambda_hat = model_ar1.params[1]

    if lambda_hat >= 0:
        return np.inf

    # Half-Life : -ln(2) / lambda
    half_life = -np.log(2) / lambda_hat
    return half_life


#------------------------------------------



def empirical_mean_reversion(y: pd.Series, x: pd.Series):
    """
    The function calculates how many periods it takes on average for the cumulative
    difference in returns (y - x) to revert to 1 (i.e., no outperformance).

    Parameters
    ----------
    y : pd.Series
        First price series.
    x : pd.Series
        Second price series.

    Returns
    -------
    float
        Average number of days to mean reversion. Returns np.inf if no reversion is observed.
    """
    y = _transform_series_pct_change(y)
    x = _transform_series_pct_change(x)
    diff_returns = y - x
    days_reversion = []
    current = 1 + diff_returns.iloc[0]
    periods = 1

    if current > 1:
        situation = '+'
    else:
        situation = '-'

    for r in range(1, diff_returns):
        current *= (1 + r)
        periods += 1
        if situation == '+':
            if current <= 1:
                days_reversion.append(periods)
                periods = 0
                current = 1
        else:
            if current >= 1:
                days_reversion.append(periods)
                periods = 0
                current = 1

    if not days_reversion:
        return np.inf

    return np.mean(days_reversion)



#------------------------------------------



def hurst_exponent(ts: pd.Series) -> float:
    """
    Estimate the Hurst exponent of a time series to evaluate its mean-reverting or trending behavior.

    Parameters
    ----------
    ts: pd.Series
        Input time series (e.g., price or spread).

    Returns
    -------
    float
        - H ≈ 0.5 -> random walk (no memory),
        - H < 0.5 -> mean-reverting (anti-persistent),
        - H > 0.5 -> trending (persistent).
    """

    lags = range(2, len(ts)//10)
    tau = [np.std(ts.diff(lag).dropna()) for lag in lags]
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    log_lags = sm.add_constant(log_lags)
    slope = sm.OLS(log_tau, log_lags).fit()
    hurst = slope.params[1]
    return hurst



import numpy as np
import pandas as pd
import statsmodels.api as sm
from .processing import _transform_series_pct_change


def half_life(y: pd.Series, x: pd.Series):
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

    exog = sm.add_constant(x)
    model = sm.OLS(y, exog).fit()
    residual = model.resid
    residual_lag = residual.shift(1)
    delta_residual = residual - residual_lag

    valid = residual_lag.notna() & delta_residual.notna()
    residual_lag = residual_lag[valid]
    delta_residual = delta_residual[valid]

    x_lag = sm.add_constant(residual_lag)
    model_ar1 = sm.OLS(delta_residual, x_lag).fit()
    lambda_hat = model_ar1.params.iloc[1]

    if lambda_hat >= 0:
        return np.inf

    half_life = -np.log(2) / lambda_hat

    return round(half_life,2)




#------------------------------------------


def segmented_half_life(y: pd.Series, x: pd.Series):
    """
    Segment the two price series in 10 groups, estimate the half-life of mean reversion of the spread for each group.

    Parameters
    ----------
    y : pd.Series
        First price series.
    x : pd.Series
        Second price series.

    Returns
    -------
    list
        Estimated half-life for each segment
    """
    half_lives = []
    start = len(y)//10
    stop = len(y)
    step = len(y)//10
    for i in range(start, stop, step):
        hl = half_life(y[i-start:i], x[i-start:i])
        half_lives.append(hl)
    return round(np.var(half_lives),1)


#------------------------------------------


def rsquared (y: pd.Series, x: pd.Series):
    """
    Calculate the coefficient of determination (R²) from a linear regression of y on x.

    Parameters
    ----------
    y : pd.Series
        Dependent variable (target price series).
    x : pd.Series
        Independent variable (regressor price series).

    Returns
    -------
    float
        R² value indicating the proportion of variance in y explained by x,
        rounded to three decimal places.
    """
    r2 = sm.OLS(y, sm.add_constant(x)).fit().rsquared
    return round(r2,3)


#------------------------------------------



def empirical_mean_reversion(y: pd.Series, x: pd.Series):
    """
    Calculates how many periods it takes on average for the cumulative
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
    list_reversions = []
    periods = 1

    for i in range(0,20,4):
        current = 1 + diff_returns.iloc[i]
        if current > 1:
            situation = '+'
        else:
            situation = '-'
        for r in diff_returns.iloc[i+1:]:
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
            break
        list_reversions.append(round(np.mean(days_reversion),2))
    mean = np.mean(list_reversions)
    stdev = np.std(list_reversions)
    cv = stdev/mean

    return round(cv*100, 2)



def expected_time_to_mean(y: pd.Series, x: pd.Series) -> float:
    """
    Empirically estimate how many steps it takes on average for the spread to revert to its mean.

    Returns
    -------
    float
        Expected number of periods to mean reversion.
    """
    model = sm.OLS(y, sm.add_constant(x)).fit()
    spread = model.resid
    deviations = spread - spread.mean()
    signs = np.sign(deviations)

    times = []
    current_sign = signs.iloc[0]
    steps = 1

    for s in signs.iloc[1:]:
        if s == current_sign and s != 0:
            steps += 1
        else:
            if current_sign != 0:
                times.append(steps)
            steps = 1
            current_sign = s

    if not times:
        return np.inf

    return round(np.mean(times), 2)


#------------------------------------------



def hurst_exponent(y: pd.Series, x: pd.Series):
    """
    Estimate the Hurst exponent of a time series to evaluate its mean-reverting or trending behavior.

    Parameters
    ----------
    y : pd.Series
        First price series.
    x : pd.Series
        Second price series.

    Returns
    -------
    float
        - H ≈ 0.5 -> random walk (no memory),
        - H < 0.5 -> mean-reverting (anti-persistent),
        - H > 0.5 -> trending (persistent).
    """

    exog = sm.add_constant(x)
    linest = sm.OLS(y, exog).fit()
    residuals = linest.resid
    lags = range(2, len(residuals)//10)
    tau = [np.std(residuals.diff(lag).dropna()) for lag in lags]
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    log_lags = sm.add_constant(log_lags)
    slope = sm.OLS(log_tau, log_lags).fit()
    hurst = slope.params[1]
    return  round(hurst,3)




def estimate_kappa(y: pd.Series, x: pd.Series) -> float:
    """
    Estimate the speed of mean reversion (kappa) of the spread between two price series,
    based on an Ornstein-Uhlenbeck process approximation.

    Parameters
    ----------
    y : pd.Series
        First price series.
    x : pd.Series
        Second price series.

    Returns
    -------
    float
        Estimated speed of mean reversion (kappa), 0 if no mean-reverting.
    """

    model = sm.OLS(y, sm.add_constant(x)).fit()
    residual = model.resid

    residual_lag = residual.shift(1)
    delta_residual = residual - residual_lag
    valid = residual_lag.notna() & delta_residual.notna()
    model_ar1 = sm.OLS(delta_residual[valid], sm.add_constant(residual_lag[valid])).fit()
    lambda_hat = model_ar1.params.iloc[1]

    if lambda_hat >= 0:
        return 0.0

    kappa = -np.log(1 + lambda_hat)
    return round(kappa, 3)




def count_mean_crossings(y: pd.Series, x: pd.Series) -> int:
    """
    Count how many times the spread between two price series crosses its mean level.

    Parameters
    ----------
    y : pd.Series
        First price series.
    x : pd.Series
        Second price series.

    Returns
    -------
    int
        Number of mean crossings (upward or downward).
    """

    model = sm.OLS(y, sm.add_constant(x)).fit()
    spread = model.resid

    centered = spread - spread.mean()
    sign_change = centered.shift(1) * centered < 0
    crossings = sign_change.sum()
    return int(crossings)




def sup_wald_test(y: pd.Series, x: pd.Series, split_ratio: float = 0.5):
    """
    Perform sup-Wald (Chow-type) test for structural break in the regression of y ~ x.
    Split the sample into two periods and compare coefficients.

    Parameters
    ----------
    y : pd.Series
        Dependent variable (spread).
    x : pd.Series
        Independent variable.
    split_ratio : float
        Ratio to split the dataset (default: 0.5).

    Returns
    -------
    float
        F-statistic of the Chow test.
    """
    n = len(y)
    split = int(n * split_ratio)

    x_full = sm.add_constant(x)
    x1 = sm.add_constant(x.iloc[:split])
    x2 = sm.add_constant(x.iloc[split:])

    y1 = y.iloc[:split]
    y2 = y.iloc[split:]

    model_pooled = sm.OLS(y, x_full).fit()
    model1 = sm.OLS(y1, x1).fit()
    model2 = sm.OLS(y2, x2).fit()

    RSS_pooled = np.sum(model_pooled.resid**2)
    RSS1 = np.sum(model1.resid**2)
    RSS2 = np.sum(model2.resid**2)

    k = x1.shape[1]  # number of parameters
    F = ((RSS_pooled - (RSS1 + RSS2)) / k) / ((RSS1 + RSS2) / (n - 2 * k))
    return round(F, 4)




def avg_time_outside_band(y: pd.Series, x: pd.Series, sigma_level: float = 1) -> float:
    """
    Compute the average duration the spread stays outside the ±σ band.

    Returns
    -------
    float
        Average time spent outside the band.
    """
    model = sm.OLS(y, sm.add_constant(x)).fit()
    spread = model.resid
    mean = spread.mean()
    std = spread.std()
    upper = mean + sigma_level * std
    lower = mean - sigma_level * std

    out = (spread > upper) | (spread < lower)

    durations = []
    count = 0
    for val in out:
        if val:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0

    if count > 0:
        durations.append(count)

    if not durations:
        return np.inf
    return round(np.mean(durations), 2)


#------------------------------------------




"""
def approximate_entropy(u: np.ndarray, m: int, r: float) -> float:
    def _phi(m):
        x = np.array([u[i:i + m] for i in range(len(u) - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (len(u) - m + 1)
        return np.sum(np.log(C)) / (len(u) - m + 1)
    return abs(_phi(m) - _phi(m + 1))
"""

"""
def cusum_test(y: pd.Series, x: pd.Series):

    Perform CUSUM test on residuals of regression y ~ x.

    Returns
    -------
    Tuple[float, float]
        Test statistic and p-value.

    model = sm.OLS(y, sm.add_constant(x)).fit()
    test_stat, p_value, _ = breaks_cusumolsresid(model.resid, ddof=model.df_model)
    return round(test_stat, 4), round(p_value, 4)
"""


# --- Mean Reversion Characteristics ---
"""
def asymmetry_of_reversion(spread: pd.Series) -> float:
    
    Compute the asymmetry in mean reversion speed above vs. below the mean.

    Returns
    -------
    float
        Difference in average reversion time between top and bottom.
    
    mean = spread.mean()
    above = spread[spread > mean]
    below = spread[spread < mean]

    time_above = (above - mean).mean()
    time_below = (mean - below).mean()

    return round(abs(time_above - time_below), 4)
"""










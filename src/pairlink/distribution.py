import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera, shapiro, anderson


def _find_residuals(y: pd.Series, x: pd.Series):
    model = sm.OLS(y, sm.add_constant(x)).fit()
    residuals = model.resid
    return residuals

def jarque_bera_test(y: pd.Series, x: pd.Series) -> dict:
    """
    Evaluates whether the residuals have skewness and kurtosis that significantly deviate
    from a normal distribution.

    Parameters
    ----------
    y : pd.Series
        Dependent variable (target price series).
    x : pd.Series
        Independent variable (regressor price series).

    Returns
    -------
    dict
        - statistic: Smaller than 6 if skewness & kurtosis close to normal.
        - p_value: A value below .05 suggests non-normality.
    """
    residuals = _find_residuals(y,x)
    statistic, p_value = jarque_bera(residuals.dropna())
    return round(statistic, 2)


def shapiro_wilk_test(y: pd.Series, x: pd.Series) -> dict:
    """
    Focusing on sample ordering, particularly powerful for small samples. (Residuals Analysis)

    Parameters
    ----------
    y : pd.Series
        Dependent variable (target price series).
    x : pd.Series
        Independent variable (regressor price series).

    Returns
    -------
    dict
        - statistic: Close to 1 if the series is normally distributed.
        - p_value: A value below .05 suggests non-normality.
    """
    if len(y) < 5000 or len(x) < 5000:
        residuals = _find_residuals(y, x)
        statistic, p_value = shapiro(residuals.dropna())
        return round(statistic, 3)
    else:
        return "Unknown (N > 5000)"


def anderson_darling_test(y: pd.Series, x: pd.Series) -> dict:
    """
    Sensitive to outliers and deviations in the extremes (on residuals).

    Parameters
    ----------
    y : pd.Series
        Dependent variable (target price series).
    x : pd.Series
        Independent variable (regressor price series).

    Returns
    -------
    dict
        - statistic: The Anderson-Darling test statistic.
        - critical_values: Thresholds for rejection at common confidence levels.
    """
    residuals = _find_residuals(y, x)
    result = anderson(residuals.dropna(), dist='norm')
    return {
        "statistic": round(result.statistic, 3),
        "critical_values": dict(zip(result.significance_level, result.critical_values))
    }




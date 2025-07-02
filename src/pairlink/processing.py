import pandas as pd
import numpy as np


def _validate_series(series1, series2):
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        raise TypeError("Both inputs must be pandas Series.")
    if not isinstance(series1.index, pd.DatetimeIndex) or not isinstance(series2.index, pd.DatetimeIndex):
        raise ValueError("Both Series must have a DatetimeIndex.")


def _align_series(series1, series2):
    same_index = series1.index.equals(series2.index)
    if not same_index:
        s1, s2 = series1.align(series2, join='inner')
    else:
        s1, s2 = series1, series2
    df = pd.concat([s1, s2], axis=1).dropna()
    return df.iloc[:, 0], df.iloc[:, 1]



def _transform_series_log(series):
    series = np.log(series)
    series = series.dropna()
    return series

def _transform_series_pct_change(series):
    series = series.pct_change()
    series = series.dropna()
    return series


def preprocess_series(series1, series2, log=False, pct_change=False):
    _validate_series(series1, series2)
    series1, series2 = _align_series(series1, series2)
    if log and pct_change:
        raise ValueError("Please choose either logarithmic returns OR percentage returns, not both.")
    if log:
        s1 = _transform_series_log(series1)
        s2 = _transform_series_log(series2)
        s1, s2 = _align_series(s1, s2)
        return s1, s2
    if pct_change:
        s1 = _transform_series_pct_change(series1)
        s2 = _transform_series_pct_change(series2)
        s1, s2 = _align_series(s1, s2)
        return s1, s2
    return series1, series2
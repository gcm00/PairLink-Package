from .cointegration import*
from .mean_reversion import*
from .distribution import*

def pairlink_test (dependent: pd.Series, independent:pd.Series):
    """
    Assess the presence of stationarity, cointegration, and mean reversion between two time series.

    Parameters
    ----------
    dependent : pd.Series
        The dependent time series.
    independent : pd.Series
        The independent time series.

    Returns
    -------
    dict
        A dictionary summarizing the binary status (✅ or ❌) for the following conditions:

            - 'Stationarity': Whether both series are individually stationary.
            - 'Cointegration': Whether the series are cointegrated in both directions.
            - 'Distribution': Whether the residuals of the spread follow a normal distribution.

    Examples
    --------
    >>> from pairlink import preprocess_series, pairlink_test
    >>> y, x = preprocess_series(series1=df['AAPL'], series2=df['MSFT'])
    >>> pairlink_test(y,x)
    Print test results
    """

    serie1, serie2 = preprocess_series(series1=dependent, series2=independent)
    engle_granger = engle_granger_global(series1=serie1, series2=serie2)
    empirical_mr = empirical_mean_reversion(y=serie1, x=serie2)
    time_to_mean= expected_time_to_mean(y=serie1, x=serie2)
    hl = half_life(y=serie1, x=serie2)
    segmented_hl = segmented_half_life(y=serie1, x=serie2)
    r2 = rsquared(y=serie1, x=serie2)
    hust = hurst_exponent(y=serie1, x=serie2)
    kappa = estimate_kappa(y=serie1, x=serie2)
    mean_crossings = count_mean_crossings(y=serie1, x=serie2)
    time_outside_band = avg_time_outside_band(y=serie1, x=serie2)
    jbera = jarque_bera_test(y=serie1, x=serie2)
    swilk = shapiro_wilk_test(y=serie1, x=serie2)
    adarling = anderson_darling_test(y=serie1, x=serie2)

    good = "✅"
    bad = "❌"

    try:
        status_stationnarity = good if all(engle_granger[col] == 1 for col in [
            'Integration Series 1 - ADF',
            'Integration Series 2 - ADF',
            'Integration Series 1 - KPSS - c',
            'Integration Series 1 - KPSS - ct',
            'Integration Series 2 - KPSS - c',
            'Integration Series 2 - KPSS - ct'
        ]) else bad
    except:
        status_stationnarity = bad

    try:
        status_coint =  good if engle_granger['Engle Granger'] < .05 else bad
    except:
        status_coint = bad

    try:
        status_distribution =  good if jbera < 6 and swilk > .95 and adarling['statistic'] < adarling['critical_values'][5.0] else bad
    except:
        status_distribution = bad
    pairlink_results = {
        '--- Individual Stationarity ---': '',
        'ADF Series 1': f"I({engle_granger['Integration Series 1 - ADF']})",
        'ADF Series 2': f"I({engle_granger['Integration Series 2 - ADF']})",
        'KPSSc Series 1': f"I({engle_granger['Integration Series 1 - KPSS - c']})",
        'KPSSc Series 2': f"I({engle_granger['Integration Series 2 - KPSS - c']})",
        'KPSSct Series 1': f"I({engle_granger['Integration Series 1 - KPSS - ct']})",
        'KPSSct Series 2': f"I({engle_granger['Integration Series 2 - KPSS - ct']})",
        '':'',
        '--- Check Cointegration ---': '',
        'Engle Granger  (< 0.05)': f"{engle_granger['Engle Granger']}",
        ' ': '',
        '--- Distribution Analysis ---': '',
        "Jarque Bera  (< 6.0)": f"{jbera}",
        'Shapiro Wilk  (> 0.95)': f"{swilk}",
        f"Anderson Darling  (< {round(adarling['critical_values'][5.0],2)})": f"{adarling['statistic']}",
        '  ': '',
        '--- Mean Reversion Analysis ---': '',
        'Expected time to M.': time_to_mean,
        'Mean Crossing': mean_crossings,
        'Time Outside of σ':time_outside_band,
        'Half Life': hl,
        'Segmented Half Life': segmented_hl,
        'Hurst Exponent  (< 0.5)': hust,
        '   ': '',
        '--- Spread Stability & Fit ---': '',
        'R²': r2,
        'Coef. Variation M.R.': f"{empirical_mr}%",
        'Kappa  (> 0.1)': kappa,
    }



    #presentation
    key_width = max(len(k) for k in pairlink_results.keys()) + 2
    val_width = max(len(str(v)) for v in pairlink_results.values()) + 5
    total_width = key_width + val_width + 5
    line = '-' * total_width
    print(line)
    print(f"| {'Test  (Targeted Value)'.ljust(key_width)}| {'Result'.ljust(val_width)}|")
    print(line)
    for k, v in pairlink_results.items():
        print(f"| {k.ljust(key_width)}| {str(v).ljust(val_width)}|")
    print(line)


    print(f"\n___ Status ___ "
          f"\nStationarity: {status_stationnarity}"
          f"\nCointegration: {status_coint}"
          f"\nDistribution: {status_distribution}")

    return {
        'Stationarity': status_stationnarity,
        'Cointegration': status_coint,
        'Distribution': status_distribution
    }
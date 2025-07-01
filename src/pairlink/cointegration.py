import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from .processing import*
from statsmodels.tools.sm_exceptions import InterpolationWarning
import statsmodels.api as sm

def engle_granger_global(series1: pd.Series, series2: pd.Series,autolag: str = "AIC"):
    """
    Verifier que les 2 series ont la meme intégration via ADF + KPSS
    Effectue le test de cointégration d'Engle-Granger dans les deux directions :
    - series1 ~ series2
    - series2 ~ series1
    """
    eg_1 = None
    eg_2 = None

    #-----------------------------------

    def test_adf(series, max_diff=2, signif_level=0.05):
        current_series = series.copy()
        for d in range(max_diff + 1):
            try:
                p_value = adfuller(current_series.dropna())[1]
            except Exception as e:
                print(f"[ERROR] ADF test failed: {e} - put default p-value = 1")
                p_value = 1
            if p_value < signif_level:
                return d
            current_series = current_series.diff()
        return None

    #-----------------------------------

    def test_kpss_c(series, max_diff=2, signif_level=0.05):
        current = series.copy()
        for d in range(max_diff + 1):
            try:
                p_value_c = kpss(current.dropna(), regression='c', nlags="auto")[1]
            except Exception as e:
                print(f"[ERROR] KPSS test (regression='c') failed: {e} - put default p-value = 0")
                p_value_c = 0
            if p_value_c > signif_level:
                return d
            current = current.diff()
        return None

    #-----------------------------------

    def test_kpss_ct(series, max_diff=2, signif_level=0.05):
        current = series.copy()
        for d in range(max_diff + 1):
            try:
                p_value_ct = kpss(current.dropna(), regression='ct', nlags="auto")[1]
            except Exception as e:
                print(f"[ERROR] KPSS (regression='ct') test failed: {e} - put default p-value = 0")
                p_value_ct = 0
            if p_value_ct > signif_level:
                return d
            current = current.diff()
        return None

    #-----------------------------------

    def test_engle_granger(dep: pd.Series, indep: pd.Series, direction_label: str):
        x = sm.add_constant(indep)
        model = sm.OLS(dep, x).fit()
        hedge_ratio = model.params.iloc[1]
        residuals = model.resid

        adf_result = adfuller(residuals, autolag=autolag)
        adf_stat, pvalue, _, _, crit_values, _ = adf_result

        return {
            'is_cointegrated': pvalue < 0.05,
            'p-value': round(pvalue,4),
            'hedge_ratio': hedge_ratio,
        }

    #-----------------------------------

    warnings.simplefilter("ignore", InterpolationWarning)
    integration_1_adf = test_adf(series=series1)
    integration_2_adf = test_adf(series=series2)

    if integration_1_adf == 1 and integration_2_adf == 1:
        integration_1_kpss_c = test_kpss_c(series=series1)
        integration_1_kpss_ct = test_kpss_ct(series=series1)
        integration_2_kpss_c = test_kpss_c(series=series2)
        integration_2_kpss_ct = test_kpss_ct(series=series2)

        if (integration_1_kpss_c == 1 and integration_2_kpss_c == 1) or (integration_1_kpss_ct == 1 and integration_2_kpss_ct == 1):
            eg_1 = test_engle_granger(series1, series2, "linest:s1_on_s2")
            eg_2 = test_engle_granger(series2, series1, "linest:s2_on_s1")

        else:
            print(
                f"According to the KPSS Test (regression='c'): "
                f"Series 1 is I({integration_1_kpss_c}), Series 2 is I({integration_2_kpss_c})\n"
                f"According to the KPSS Test (regression='ct'): "
                f"Series 1 is I({integration_1_kpss_ct}), Series 2 is I({integration_2_kpss_ct})\n"
                f"However, both series must be I(1) with at least one method (c or ct) to perform the cointegration test."
            )

            return {
                "Integration Series 1 - ADF": integration_1_adf,
                "Integration Series 2 - ADF": integration_2_adf,
                "Integration Series 1 - KPSS - c": integration_1_kpss_c,
                "Integration Series 2 - KPSS - c": integration_2_kpss_c,
                "Integration Series 1 - KPSS - ct": integration_1_kpss_ct,
                "Integration Series 2 - KPSS - ct": integration_2_kpss_ct,
                "Cointegration Status": "Not Tested - Non I(1)"
            }

    else:
        print(f"According to ADF Test: 1st Series is I({integration_1_adf}) and 2nd is I({integration_2_adf}), "
              f"However, both series must be I(1) to perform the cointegration test.")
        return {
            "Integration Series 1 - ADF": integration_1_adf,
            "Integration Series 2 - ADF": integration_2_adf,
            "Cointegration Status": "Not Tested - Non I(1)"
        }

    return {
        "Integration Series 1 - ADF": integration_1_adf,
        "Integration Series 2 - ADF": integration_2_adf,
        "Integration Series 1 - KPSS - c": integration_1_kpss_c,
        "Integration Series 2 - KPSS - c": integration_2_kpss_c,
        "Integration Series 1 - KPSS - ct": integration_1_kpss_ct,
        "Integration Series 2 - KPSS - ct": integration_2_kpss_ct,
        "Engle_linest_1/2": eg_1['p-value'],
        "Engle_linest_2/1": eg_2['p-value']
    }

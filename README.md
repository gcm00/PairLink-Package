# PairLink

## Description

`PairLink` is a diagnostic tool designed to evaluate the statistical relationship between two time series. 

It provides an assessment of their stationarity, cointegration, distributional characteristics, and mean-reverting behavior.

## Installation

You can install `PairLink` via pip:

```bash
pip install pairlink
```

## Usage
Here is a simple example to identify frequent price levels from your price data:
```python
import yfinance as yf
from pairlink import preprocess_series, pairlink_test

df = yf.download(tickers=['AAPL','MSFT'], interval='1D')

y, x = preprocess_series(series1=df['Close']['AAPL'], series2=df['Close']['MSFT'])
results = pairlink_test(y,x)
```

## Author

Anthony Gocmen - [email](mailto:anthony.gocmen@gmail.com)  
WebSite: [developexx](https://www.developexx.com)


## License


This project is licensed under the MIT License - see [MIT](https://choosealicense.com/licenses/mit/) for details.
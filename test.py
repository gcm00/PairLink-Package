from src.pairlink.core import*


df = pd.read_csv("", index_col=0, parse_dates=True)
data1 = df['']
data2 = df['']

data1, data2 = preprocess_series(series1=data1, series2=data2)
test1 = pairlink_test(data1,data2)
print(test1)

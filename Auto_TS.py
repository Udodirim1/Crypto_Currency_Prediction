# import libraries

import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling
import sns as sns
import yfinance as yf

data = pd.read_csv('latestcrypto.csv')

# data = data[["Date", "Close"]]
# data["Date"] = pd.to_datetime(data.Date)
# data["Close"].plot(figsize=(12, 8), title="Cryptocurrency", fontsize=20, label="Close Price")
# plt.legend()
# plt.grid()
# plt.show()

from autots import AutoTS
model = AutoTS(forecast_length=10, frequency='infer',
               ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

prediction = model.predict()
forecast = prediction.forecast
print("Cryptocurrency Price Prediction")
print(forecast)

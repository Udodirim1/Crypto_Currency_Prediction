import streamlit as st
from PIL import Image
import yfinance as yf
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split



#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

image = Image.open('logo.jpg')

st.image(image, width = 500)

st.title('SOLIGENCE TRADING APP')

col1 = st.sidebar
col2, col3 = st.columns((2,1))

col1.header('User Selections')

crypto_symbols = ['BTC', 'ETH', 'USDT','USDC', 'BNB',
                  'BUSD', 'XRP', 'ADA', 'SOL', 'DOGE',
                  'DAI', 'DOT', 'WTRX', 'HEX', 'TRX',
                  'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC',
                  'MATIC', 'STETH', 'UNI1', 'LTC', 'FTT',
                  'LINK', 'CRO', 'XLM', 'NEAR', 'ATOM']

coin_predict = col1.multiselect('Pick your assets', crypto_symbols, crypto_symbols[0])
#print(coin_predict)

start = col1.date_input('Start', value=pd.to_datetime('2018-01-01'))
end = col1.date_input('End', value=pd.to_datetime('today'))
print(start,end)

#coin_predict = col1.selectbox('Choose coin', tuple(crypto_symbols))
#print(coin_predict)

days_predict = col1.selectbox('Future Days Prediction', (1,7,30))
print(days_predict)



@st.cache
def predict_coins(coin, days):
    crypto_dataset = yf.download(f'{coin}-USD', start=start, end=end, interval="1d")
    crypto_clean = crypto_dataset.copy().dropna()

    crypto_clean[f'{days}-Prediction'] = crypto_clean[['Close']].shift(-days)

    predict_dataset = crypto_clean.tail(days)
    predict_dataset = np.array(predict_dataset[['Open', 'High', 'Low', 'Close', 'Volume']])

    X = np.array(crypto_clean[['Open', 'High', 'Low', 'Close', 'Volume']])
    X = X[:-days]

    y = np.array(crypto_clean[f'{days}-Prediction'])
    y = y[:-days]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    x_model = xgb.XGBRegressor(objective='reg:squarederror')
    x_model.fit(X_train, y_train)

    y_pred = x_model.predict(X_test)

    return x_model.predict(predict_dataset), y_pred, y_test

print(coin_predict)
prediction_output = {}

col2.subheader('Future Prediction')
prediction, predict_values, test_values = predict_coins(coin_predict[0], days_predict)

for coin in coin_predict:
    prediction, predict_values, test_values = predict_coins(coin, days_predict)
    prediction_output[f'{coin} Predict'] = prediction

col2.dataframe(prediction_output)

if len(coin_predict) > 0:
    prediction, predict_values, test_values = predict_coins(coin, days_predict)
    prediction_output[f'{coin} Predict'] = prediction

plt.plot(figsize=(16, 8))
plt.plot(test_values, label='Original Value')
plt.plot(predict_values, label='Predicted Value')
plt.legend()
col2.pyplot(plt)





